"""The board matrix that sits between the vision code and chess semantics.

A :class:`BoardMatrix` holds what the *picture* shows: eight rows of eight
label characters in display order, row 0 being the top of the printed diagram.
Turning that into a FEN needs one extra fact the picture does not carry --
which side is at the bottom -- so orientation lives here too, together with the
heuristics that guess it and the side to move.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import quote

from .labels import EMPTY, LABEL_GLYPHS, LABEL_TO_INDEX, is_black, is_white

WHITE_AT_BOTTOM = "white"
BLACK_AT_BOTTOM = "black"

#: Rough material weights, used only to guess which way round the diagram is.
_PULL_WEIGHTS = {"p": 1.0, "n": 3.0, "b": 3.0, "r": 5.0, "q": 9.0, "k": 4.0}


class BoardError(ValueError):
    """Raised when a board matrix is malformed."""


@dataclass
class BoardMatrix:
    """An 8x8 grid of label characters in display order (row 0 = top)."""

    rows: list[list[str]]
    orientation: str = WHITE_AT_BOTTOM
    side_to_move: str = "w"
    castling: str | None = None  # None means "infer from the placement"
    confidence: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.rows) != 8 or any(len(r) != 8 for r in self.rows):
            raise BoardError("a board matrix must be 8 rows of 8 squares")
        for row in self.rows:
            for label in row:
                if label not in LABEL_TO_INDEX:
                    raise BoardError(f"not a piece label: {label!r}")
        if self.orientation not in (WHITE_AT_BOTTOM, BLACK_AT_BOTTOM):
            raise BoardError(f"unknown orientation: {self.orientation!r}")
        if self.side_to_move not in ("w", "b"):
            raise BoardError(f"side to move must be 'w' or 'b', got {self.side_to_move!r}")

    # -- construction ----------------------------------------------------

    @classmethod
    def empty(cls, **kwargs) -> "BoardMatrix":
        return cls([[EMPTY] * 8 for _ in range(8)], **kwargs)

    @classmethod
    def from_labels(cls, labels: list[str] | str, **kwargs) -> "BoardMatrix":
        """Build from 64 labels in reading order, or from a 64-character string."""
        flat = list(labels)
        if len(flat) != 64:
            raise BoardError(f"expected 64 labels, got {len(flat)}")
        return cls([flat[i * 8:(i + 1) * 8] for i in range(8)], **kwargs)

    @classmethod
    def from_fen(cls, fen: str, orientation: str = WHITE_AT_BOTTOM) -> "BoardMatrix":
        """Inverse of :meth:`to_fen`, handy for tests and for round-tripping."""
        parts = fen.split()
        placement = parts[0]
        side = parts[1] if len(parts) > 1 else "w"
        castling = parts[2] if len(parts) > 2 else None
        ranks = placement.split("/")
        if len(ranks) != 8:
            raise BoardError(f"FEN placement needs 8 ranks, got {len(ranks)}")
        grid: list[list[str]] = []
        for rank in ranks:
            row: list[str] = []
            for ch in rank:
                if ch.isdigit():
                    row.extend([EMPTY] * int(ch))
                else:
                    row.append(ch)
            if len(row) != 8:
                raise BoardError(f"FEN rank does not add up to 8 squares: {rank!r}")
            grid.append(row)
        if orientation == BLACK_AT_BOTTOM:
            grid = [list(reversed(r)) for r in reversed(grid)]
        return cls(grid, orientation=orientation, side_to_move=side, castling=castling)

    # -- access ----------------------------------------------------------

    def __getitem__(self, rc: tuple[int, int]) -> str:
        row, col = rc
        return self.rows[row][col]

    def __setitem__(self, rc: tuple[int, int], label: str) -> None:
        row, col = rc
        if label not in LABEL_TO_INDEX:
            raise BoardError(f"not a piece label: {label!r}")
        self.rows[row][col] = label

    def flat(self) -> list[str]:
        return [label for row in self.rows for label in row]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for label in self.flat():
            if label != EMPTY:
                out[label] = out.get(label, 0) + 1
        return out

    def square_name(self, row: int, col: int) -> str:
        """Algebraic name of a display cell, taking orientation into account."""
        if self.orientation == WHITE_AT_BOTTOM:
            file_index, rank_index = col, 7 - row
        else:
            file_index, rank_index = 7 - col, row
        return "abcdefgh"[file_index] + str(rank_index + 1)

    def flipped(self) -> "BoardMatrix":
        """The same physical diagram read from the other side of the table."""
        other = BLACK_AT_BOTTOM if self.orientation == WHITE_AT_BOTTOM else WHITE_AT_BOTTOM
        return BoardMatrix(
            [list(r) for r in self.rows],
            orientation=other,
            side_to_move=self.side_to_move,
            castling=self.castling,
            confidence=[list(r) for r in self.confidence],
        )

    # -- export ----------------------------------------------------------

    def placement(self) -> str:
        """The first FEN field: ranks 8 down to 1, files a to h."""
        grid = self.rows
        if self.orientation == BLACK_AT_BOTTOM:
            grid = [list(reversed(r)) for r in reversed(grid)]
        ranks: list[str] = []
        for row in grid:
            out, run = [], 0
            for label in row:
                if label == EMPTY:
                    run += 1
                    continue
                if run:
                    out.append(str(run))
                    run = 0
                out.append(label)
            if run:
                out.append(str(run))
            ranks.append("".join(out))
        return "/".join(ranks)

    def infer_castling(self) -> str:
        """Castling rights implied by the placement.

        A diagram cannot say whether a king has already moved, so we do what
        board editors do: grant the right whenever king and rook are both still
        sitting on their home squares.
        """
        board = BoardMatrix.from_fen(self.placement() + " w - - 0 1")
        squares = {board.square_name(r, c): board[r, c] for r in range(8) for c in range(8)}
        rights = ""
        if squares.get("e1") == "K":
            if squares.get("h1") == "R":
                rights += "K"
            if squares.get("a1") == "R":
                rights += "Q"
        if squares.get("e8") == "k":
            if squares.get("h8") == "r":
                rights += "k"
            if squares.get("a8") == "r":
                rights += "q"
        return rights or "-"

    def to_fen(self) -> str:
        castling = self.castling if self.castling is not None else self.infer_castling()
        return f"{self.placement()} {self.side_to_move} {castling or '-'} - 0 1"

    def lichess_url(self, kind: str = "analysis") -> str:
        """A Lichess URL that opens this position.

        ``kind`` is ``analysis`` for the analysis board or ``editor`` for the
        board editor, which is the better landing place when the position still
        needs a tweak.
        """
        fen = self.to_fen().replace(" ", "_")
        path = quote(fen, safe="/_-")
        if kind == "editor":
            return f"https://lichess.org/editor/{path}"
        if kind == "analysis":
            return f"https://lichess.org/analysis/standard/{path}"
        raise ValueError(f"unknown lichess target: {kind!r}")

    # -- sanity checks ---------------------------------------------------

    def problems(self) -> list[str]:
        """Human readable complaints about the position, worst first.

        Diagrams in books are often puzzle fragments rather than legal games,
        so these are warnings for the reviewer, never hard errors.
        """
        out: list[str] = []
        counts = self.counts()
        for label, name, expected in (("K", "white king", 1), ("k", "black king", 1)):
            got = counts.get(label, 0)
            if got != expected:
                out.append(f"{got} {name}s in the diagram, expected {expected}")
        for label, name, limit in (("P", "white pawns", 8), ("p", "black pawns", 8)):
            if counts.get(label, 0) > limit:
                out.append(f"{counts[label]} {name}, more than {limit}")
        board = BoardMatrix.from_fen(self.placement() + " w - - 0 1")
        for row, rank in ((0, 8), (7, 1)):
            for col in range(8):
                label = board.rows[row][col]
                if label in ("P", "p"):
                    out.append(f"a pawn on {board.square_name(row, col)} (rank {rank})")
        white = sum(v for k, v in counts.items() if is_white(k))
        black = sum(v for k, v in counts.items() if is_black(k))
        if white > 16:
            out.append(f"{white} white pieces, more than 16")
        if black > 16:
            out.append(f"{black} black pieces, more than 16")
        return out

    def ascii(self) -> str:
        """A compact rendering for terminals and log files."""
        lines = []
        for row in range(8):
            cells = " ".join(LABEL_GLYPHS[self.rows[row][col]] for col in range(8))
            lines.append(f"{cells}")
        lines.append(f"[{self.orientation} at the bottom, {self.side_to_move} to move]")
        return "\n".join(lines)


def guess_orientation(rows: list[list[str]]) -> str:
    """Guess which colour is at the bottom of a diagram.

    Pieces cluster on their own side of the board, so we compare the weighted
    centre of mass of each colour: the side whose men sit lower in the picture
    is the side at the bottom.  Endgames can defeat this, which is why the
    review UI puts a flip on one keypress.
    """
    white_mass = white_rows = black_mass = black_rows = 0.0
    for row_index, row in enumerate(rows):
        for label in row:
            if label == EMPTY:
                continue
            weight = _PULL_WEIGHTS[label.lower()]
            if is_white(label):
                white_mass += weight
                white_rows += weight * row_index
            else:
                black_mass += weight
                black_rows += weight * row_index
    if not white_mass or not black_mass:
        return WHITE_AT_BOTTOM
    return WHITE_AT_BOTTOM if (white_rows / white_mass) >= (black_rows / black_mass) else BLACK_AT_BOTTOM


_WHITE_TO_MOVE = re.compile(
    r"(white\s+(?:to\s+(?:move|play)|on\s+(?:move|play))|^\s*w\s*[:.]|\bwhite\s+moves\b)",
    re.IGNORECASE | re.MULTILINE,
)
_BLACK_TO_MOVE = re.compile(
    r"(black\s+(?:to\s+(?:move|play)|on\s+(?:move|play))|^\s*b\s*[:.]|\bblack\s+moves\b)",
    re.IGNORECASE | re.MULTILINE,
)


def guess_side_to_move(caption: str | None) -> str | None:
    """Read the side to move out of the text printed around a diagram.

    Books say it in words -- "Black to play" under the board, or a bare "W:"
    beside it.  Returns ``None`` when the caption settles nothing, so the caller
    can fall back to its own default instead of inventing one.
    """
    if not caption:
        return None
    white = bool(_WHITE_TO_MOVE.search(caption))
    black = bool(_BLACK_TO_MOVE.search(caption))
    if white == black:
        return None
    return "w" if white else "b"
