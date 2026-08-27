"""Finding the 8x8 lattice inside a board picture and cutting it into squares.

Everything downstream depends on this being right to within a pixel or two: if
the lattice is off, every square crop carries a slice of its neighbour and the
classifier never recovers.  So rather than assuming the detected box is exactly
the board, we search for the nine horizontal and nine vertical lines that best
explain the picture's edge energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A board smaller than this many pixels per square is not worth reading.
MIN_CELL_PX = 6

#: The checkerboard sign pattern, +1 on light squares and -1 on dark ones.
CHECKER = np.indices((8, 8)).sum(axis=0) % 2 * -2 + 1


@dataclass(frozen=True)
class GridFit:
    """Where the board's lattice sits, in the coordinates of the crop it was fitted to."""

    x0: float
    y0: float
    step_x: float
    step_y: float
    line_score: float = 0.0
    checker_score: float = 0.0

    @property
    def x1(self) -> float:
        return self.x0 + 8 * self.step_x

    @property
    def y1(self) -> float:
        return self.y0 + 8 * self.step_y

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def score(self) -> float:
        """How board-like the geometry is, in 0..1.

        Either evidence alone is enough: plenty of books print an unshaded board
        that is nothing but ruled lines, and plenty of others shade the squares
        and draw no interior rules at all.  So the stronger signal carries the
        score and the weaker one only tops it up.
        """
        return max(self.line_score, self.checker_score) * 0.85 + min(self.line_score, self.checker_score) * 0.15

    def cell_box(self, row: int, col: int, pad_frac: float = 0.0) -> tuple[float, float, float, float]:
        """Box of one cell, optionally grown by a fraction of a cell on each side."""
        px, py = pad_frac * self.step_x, pad_frac * self.step_y
        cx0 = self.x0 + col * self.step_x - px
        cy0 = self.y0 + row * self.step_y - py
        return (cx0, cy0, cx0 + self.step_x + 2 * px, cy0 + self.step_y + 2 * py)

    @classmethod
    def from_box(cls, box: tuple[float, float, float, float], **scores) -> "GridFit":
        """The lattice you get by dividing a board's frame into eight."""
        x0, y0, x1, y1 = box
        return cls(x0, y0, (x1 - x0) / 8.0, (y1 - y0) / 8.0, **scores)

    def with_scores(self, line_score: float, checker_score: float) -> "GridFit":
        return GridFit(self.x0, self.y0, self.step_x, self.step_y, line_score, checker_score)

    def translated(self, dx: float, dy: float) -> "GridFit":
        return GridFit(self.x0 + dx, self.y0 + dy, self.step_x, self.step_y,
                       self.line_score, self.checker_score)

    def as_dict(self) -> dict:
        return {
            "x0": self.x0, "y0": self.y0, "step_x": self.step_x, "step_y": self.step_y,
            "line_score": self.line_score, "checker_score": self.checker_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GridFit":
        return cls(
            float(data["x0"]), float(data["y0"]),
            float(data["step_x"]), float(data["step_y"]),
            float(data.get("line_score", 0.0)), float(data.get("checker_score", 0.0)),
        )


def edge_profile(gray: np.ndarray, axis: int) -> np.ndarray:
    """A profile over one axis that peaks where the board's rules are.

    The naive profile -- total gradient summed along the axis -- does not work
    here, because thirty-two pieces put more ink on a page than eighteen thin
    rules do, and the fitter ends up chasing knights.  What separates a rule
    from a piece is coherence: a rule produces an edge in nearly every row it
    crosses, a piece only in the rows it occupies.  So we threshold the gradient
    and count rows rather than summing magnitudes.

    ``axis=1`` gives a profile over x that peaks at vertical rules; ``axis=0``
    gives one over y that peaks at horizontal ones.
    """
    img = gray.astype(np.float32)
    diff = np.abs(np.diff(img, axis=axis))
    if diff.size == 0:
        return np.zeros(gray.shape[1 - axis], np.float32)
    strong = max(6.0, float(np.percentile(diff, 99.0)) * 0.25)
    counts = (diff > strong).sum(axis=1 - axis).astype(np.float32)
    profile = np.concatenate([counts, counts[-1:]])
    profile -= np.median(profile)
    np.clip(profile, 0, None, out=profile)
    # Normalise against a typical rule rather than the strongest one: books
    # frame the board in a rule much heavier than the interior ones, and
    # dividing by that would flatten the interior rules into the noise.
    top = np.sort(profile)[-18:]
    scale = float(np.median(top)) or float(profile.max()) or 1.0
    profile = np.clip(profile / scale, 0.0, 1.0)
    # A gentle blur buys about a pixel of tolerance without letting a fit that
    # is three pixels out score as well as the right one.
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25], np.float32)
    kernel /= kernel.sum()
    return np.convolve(profile, kernel, mode="same")


def _fit_axis(
    profile: np.ndarray,
    search_frac: float = 0.30,
    step_hint: float | None = None,
    step_tol: float = 0.06,
) -> tuple[float, float, float]:
    """Fit nine equally spaced lines to one profile.

    Returns ``(first_line, step, score)``.  We search over where the first and
    last lines sit rather than over offset and spacing, because the board's two
    outer edges are the two features we can most reliably expect to be there.
    Positions are interpolated, so the fit lands on a fraction of a pixel.

    ``step_hint`` restricts the spacing to near a value already fitted on the
    other axis.  Cells are square, so once one axis is confident the other one
    has almost no freedom left, and saying so keeps a faint axis from wandering.
    """
    n = len(profile)
    limit = max(2, int(n * search_frac))
    firsts = np.arange(0, limit, 0.5, dtype=np.float32)
    lasts = np.arange(n - limit, n - 1, 0.5, dtype=np.float32)
    if len(firsts) == 0 or len(lasts) == 0:
        return 0.0, n / 8.0, 0.0

    steps = (lasts[None, :] - firsts[:, None]) / 8.0
    valid = steps >= MIN_CELL_PX
    if step_hint is not None:
        valid &= np.abs(steps - step_hint) <= step_tol * step_hint

    ks = np.arange(9, dtype=np.float32)
    positions = firsts[:, None, None] + steps[:, :, None] * ks[None, None, :]
    grid_x = np.arange(n, dtype=np.float32)
    sampled = np.interp(positions.ravel(), grid_x, profile).reshape(positions.shape)

    # Score on a trimmed mean of the nine lines rather than their plain mean.
    # The plain mean lets a lattice that catches two very strong edges -- the
    # frame around the diagram, or the boundary of the pasted image -- outscore
    # the true lattice, which catches nine ordinary ones, so we drop the two
    # strongest before averaging.  The weak end has to stay in: a lattice
    # shifted by exactly one cell still matches eight of the nine lines, and
    # forgiving its one miss would make that shift free.
    ordered = np.sort(sampled, axis=2)
    scores = ordered[:, :, :7].mean(axis=2)
    scores[~valid] = -1.0

    if not valid.any():
        return 0.0, n / 8.0, 0.0
    best = int(np.argmax(scores))
    fi, li = np.unravel_index(best, scores.shape)
    return float(firsts[fi]), float(steps[fi, li]), float(max(scores[fi, li], 0.0))


def cell_shades(gray: np.ndarray, grid: GridFit) -> np.ndarray:
    """Estimate each square's own printed shade, as an 8x8 array.

    Sampling the middle of a cell would mostly measure whatever piece is
    standing on it, so we sample the four corners instead: a piece is drawn
    centred and rarely reaches them, while the square's colour fills them.  The
    median across the four corner patches then survives the one or two corners a
    wide-based queen does reach.
    """
    out = np.zeros((8, 8), dtype=np.float32)
    h, w = gray.shape[:2]
    corner = 0.22  # fraction of a cell taken from each corner
    for row in range(8):
        for col in range(8):
            x0, y0, x1, y1 = grid.cell_box(row, col)
            cw, ch = (x1 - x0) * corner, (y1 - y0) * corner
            samples = []
            for cx0, cy0 in ((x0, y0), (x1 - cw, y0), (x0, y1 - ch), (x1 - cw, y1 - ch)):
                xs0, ys0 = int(max(0, round(cx0))), int(max(0, round(cy0)))
                xs1, ys1 = int(min(w, round(cx0 + cw))), int(min(h, round(cy0 + ch)))
                if xs1 > xs0 and ys1 > ys0:
                    samples.append(float(np.median(gray[ys0:ys1, xs0:xs1])))
            out[row, col] = float(np.median(samples)) if samples else 0.0
    return out


def cell_ink(gray: np.ndarray, grid: GridFit) -> np.ndarray:
    """Fraction of each cell that is markedly darker than the cell's own shade.

    This is the "is something standing here" measurement.  Comparing against
    each cell's own shade rather than a global threshold is what lets it work
    on a dark square and a light square at once.
    """
    shades = cell_shades(gray, grid)
    out = np.zeros((8, 8), dtype=np.float32)
    h, w = gray.shape[:2]
    for row in range(8):
        for col in range(8):
            x0, y0, x1, y1 = grid.cell_box(row, col, pad_frac=-0.10)
            xs0, ys0 = int(max(0, round(x0))), int(max(0, round(y0)))
            xs1, ys1 = int(min(w, round(x1))), int(min(h, round(y1)))
            if xs1 <= xs0 or ys1 <= ys0:
                continue
            patch = gray[ys0:ys1, xs0:xs1]
            out[row, col] = float((patch < shades[row, col] - 45).mean())
    return out


def checkerboard_score(gray: np.ndarray, grid: GridFit) -> float:
    """How strongly the cell shades alternate like a chessboard, in 0..1.

    Zero is the honest answer for the books that print their boards as bare
    outlines with no shading at all; those are carried by the line score.
    """
    shades = cell_shades(gray, grid)
    values = shades - shades.mean()
    spread = float(np.std(values))
    if spread < 1.0:  # a flat board tells us nothing either way
        return 0.0
    pattern = CHECKER.astype(np.float32)
    pattern = pattern - pattern.mean()
    denom = float(np.linalg.norm(values) * np.linalg.norm(pattern))
    if denom == 0:
        return 0.0
    return float(abs(np.dot(values.ravel(), pattern.ravel()) / denom))


def fit_grid(gray: np.ndarray, search_frac: float = 0.30) -> GridFit:
    """Fit the 8x8 lattice to a crop that is believed to hold a board.

    ``search_frac`` is how far in from each edge the outer board lines are
    allowed to be, as a fraction of the crop.  Leave it generous for crops that
    include captions or coordinate labels, tighten it when the crop is known to
    be snug.
    """
    if gray.ndim != 2:
        raise ValueError("fit_grid expects a grayscale image")
    if min(gray.shape) < 8 * MIN_CELL_PX:
        raise ValueError(f"crop is too small to hold a board: {gray.shape}")

    profile_x = edge_profile(gray, axis=1)
    profile_y = edge_profile(gray, axis=0)
    x0, step_x, score_x = _fit_axis(profile_x, search_frac)
    y0, step_y, score_y = _fit_axis(profile_y, search_frac)

    # Refit the less confident axis against the more confident one's spacing.
    if score_x >= score_y:
        y0, step_y, score_y = _fit_axis(profile_y, search_frac, step_hint=step_x)
    else:
        x0, step_x, score_x = _fit_axis(profile_x, search_frac, step_hint=step_y)

    grid = GridFit(x0, y0, step_x, step_y, line_score=min(score_x, score_y))
    return GridFit(x0, y0, step_x, step_y,
                   line_score=grid.line_score,
                   checker_score=checkerboard_score(gray, grid))


def extract_squares(
    gray: np.ndarray,
    grid: GridFit,
    size: int = 48,
    pad_frac: float = 0.14,
) -> np.ndarray:
    """Cut the board into 64 square crops, in reading order (top-left first).

    Each crop is grown by ``pad_frac`` of a cell so that pieces which overhang
    their square -- almost every knight in every diagram font -- stay whole.
    """
    import cv2

    h, w = gray.shape[:2]
    out = np.zeros((64, size, size), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            x0, y0, x1, y1 = grid.cell_box(row, col, pad_frac=pad_frac)
            xs0, ys0 = int(round(x0)), int(round(y0))
            xs1, ys1 = int(round(x1)), int(round(y1))
            # Pad rather than clamp, so edge squares keep the same scale as the rest.
            pad_left, pad_top = max(0, -xs0), max(0, -ys0)
            pad_right, pad_bottom = max(0, xs1 - w), max(0, ys1 - h)
            patch = gray[max(0, ys0):min(h, ys1), max(0, xs0):min(w, xs1)]
            if patch.size == 0:
                out[row * 8 + col] = 255
                continue
            if pad_left or pad_top or pad_right or pad_bottom:
                patch = cv2.copyMakeBorder(
                    patch, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_REPLICATE,
                )
            out[row * 8 + col] = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return out
