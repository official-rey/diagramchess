"""Synthetic diagrams for the cold start.

The classifier's first training set has to come from somewhere, and it cannot
come from your book, because nothing has labelled your book yet.  So we draw
diagrams whose contents we already know, degrade them the way printing and
scanning degrade a real page, and cut them into labelled squares.  Later, once
you have verified real diagrams, :mod:`diagramchess.dataset` mixes those in --
and they are worth far more per example than these are.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess
import cv2
import numpy as np

from .board import BoardMatrix
from .grid import GridFit, extract_squares
from .labels import EMPTY, LABEL_TO_INDEX
from .pieces import PieceSet, available_piece_sets
from .render import DiagramStyle, random_style, render_diagram

#: Pieces that turn up in composed endgame studies, with rough frequencies.
_ENDGAME_PIECES = "PPPPPPNNBBRRQ"


def random_position(rng: random.Random) -> BoardMatrix:
    """Sample a position of the kind a book would actually print.

    Two thirds come from playing random legal moves, which produces realistic
    pawn structures and piece density; one third are sparse endgame-like
    scatters, because books are full of those and a game walk almost never
    reaches them.
    """
    if rng.random() < 0.66:
        board = chess.Board()
        plies = rng.randint(0, 90)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        fen = board.board_fen()
    else:
        fen = _random_endgame(rng)
    matrix = BoardMatrix.from_fen(fen + " w - - 0 1")
    if rng.random() < 0.5:
        # Books print plenty of diagrams from Black's side of the table.
        matrix = BoardMatrix([list(reversed(r)) for r in reversed(matrix.rows)])
    return matrix


def _random_endgame(rng: random.Random) -> str:
    grid = [[EMPTY] * 8 for _ in range(8)]
    free = [(r, c) for r in range(8) for c in range(8)]
    rng.shuffle(free)
    men = ["K", "k"]
    for _ in range(rng.randint(0, 10)):
        piece = rng.choice(_ENDGAME_PIECES)
        men.append(piece if rng.random() < 0.5 else piece.lower())
    for symbol in men:
        row, col = free.pop()
        if symbol in ("P", "p") and row in (0, 7):
            row = rng.randint(1, 6)
        grid[row][col] = symbol
    return BoardMatrix(grid).placement()


def degrade(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make a clean render look like it came off a page and through a scanner.

    Every step here corresponds to something that really happens to a diagram
    between the typesetter and the classifier: halftone screening, ink spread,
    rescaling to the PDF's resolution, sensor noise, and the slight rotation of
    a book held down on a flatbed.
    """
    out = image.astype(np.float32)
    noise = np.random.default_rng(rng.getrandbits(64))

    if rng.random() < 0.35:  # ink spread or starvation in the print
        kernel = np.ones((2, 2), np.uint8)
        op = cv2.MORPH_ERODE if rng.random() < 0.5 else cv2.MORPH_DILATE
        out = cv2.morphologyEx(out.astype(np.uint8), op, kernel).astype(np.float32)

    if rng.random() < 0.30:  # a page never sits perfectly straight on the glass
        angle = rng.uniform(-0.8, 0.8)
        h, w = out.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.5:  # resampled on its way into and out of the PDF
        scale = rng.uniform(0.55, 1.4)
        h, w = out.shape
        small = cv2.resize(out, (max(16, int(w * scale)), max(16, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    blur = rng.uniform(0.0, 1.1)
    if blur > 0.15:
        out = cv2.GaussianBlur(out, (0, 0), blur)

    gain = rng.uniform(0.82, 1.12)
    bias = rng.uniform(-18, 14)
    out = out * gain + bias

    if rng.random() < 0.75:
        out += noise.normal(0, rng.uniform(1.5, 9.0), out.shape).astype(np.float32)

    if rng.random() < 0.18:  # photocopier speckle
        mask = noise.random(out.shape) < rng.uniform(0.0005, 0.004)
        out[mask] = rng.choice([0.0, 255.0])

    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < 0.30:  # JPEG, the usual way a scanned book is stored
        quality = rng.randint(30, 85)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out


def jitter_grid(grid: GridFit, rng: random.Random, amount: float = 0.06) -> GridFit:
    """Nudge a grid the way an imperfect fit on a real page would.

    Training on perfectly cut squares produces a classifier that falls apart
    the first time the fitter is half a pixel out, so we teach it to expect that.
    """
    dx = rng.uniform(-amount, amount) * grid.step_x
    dy = rng.uniform(-amount, amount) * grid.step_y
    scale = 1.0 + rng.uniform(-amount, amount) * 0.5
    return GridFit(
        grid.x0 + dx, grid.y0 + dy,
        grid.step_x * scale, grid.step_y * scale,
        grid.line_score, grid.checker_score,
    )


@dataclass
class SyntheticBoard:
    """One rendered, degraded board together with its 64 labelled squares."""

    squares: np.ndarray   # (64, size, size) uint8
    labels: np.ndarray    # (64,) int64
    image: np.ndarray     # the degraded board picture, for eyeballing
    grid: GridFit
    style: DiagramStyle
    board: BoardMatrix


def synth_board(
    rng: random.Random,
    piece_sets: list[PieceSet],
    square_size: int = 48,
    pad_frac: float = 0.14,
) -> SyntheticBoard:
    """Draw one training board end to end."""
    board = random_position(rng)
    style = random_style(rng, rng.choice(piece_sets))
    rendered = render_diagram(board, style)
    image = degrade(rendered.image, rng)
    grid = jitter_grid(rendered.grid, rng)
    squares = extract_squares(image, grid, size=square_size, pad_frac=pad_frac)
    labels = np.array([LABEL_TO_INDEX[s] for s in board.flat()], dtype=np.int64)
    return SyntheticBoard(squares, labels, image, rendered.grid, style, board)


def synth_squares(
    count: int,
    rng: random.Random | None = None,
    piece_sets: list[PieceSet] | None = None,
    square_size: int = 48,
    balance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate roughly ``count`` labelled squares.

    With ``balance`` on, empty squares are thinned out.  A real board is more
    than half empty, and a classifier trained on that distribution learns to
    guess 'empty' whenever it is unsure, which is exactly the error that costs
    the most to correct by hand.
    """
    rng = rng or random.Random()
    piece_sets = piece_sets or available_piece_sets()
    images: list[np.ndarray] = []
    labels: list[int] = []
    empty_index = LABEL_TO_INDEX[EMPTY]
    while len(images) < count:
        sample = synth_board(rng, piece_sets, square_size=square_size)
        for square, label in zip(sample.squares, sample.labels):
            if balance and label == empty_index and rng.random() < 0.72:
                continue
            images.append(square)
            labels.append(int(label))
    return np.stack(images[:count]), np.array(labels[:count], dtype=np.int64)
