"""Finding board diagrams on a page.

The detector works in two stages, the way object detectors usually do.  First a
cheap proposal stage throws out boxes that might be boards, from three
independent sources so that a book which defeats one still gets found by
another.  Then a verification stage fits the 8x8 lattice inside each proposal
and scores how much it really looks like a chessboard, which is what decides.

The verification score is deliberately a plain geometric measurement rather
than a learned one, so the tool works on the first PDF you feed it with no
training at all.  Once you have verified some diagrams, :mod:`diagramchess.verifier`
trains a small net on your own pages and re-ranks these proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .grid import GridFit, cell_ink, checkerboard_score, edge_profile, fit_grid

#: Smallest board we will read, in page pixels.  At 200 dpi this is about a
#: two-centimetre diagram, below which the pieces are mush anyway.
MIN_BOARD_PX = 90
#: Boards are square; allow for scanner skew and sloppy typesetting.
MAX_ASPECT_SKEW = 0.18
#: Below this a proposal is not a board.  Chosen by sweeping generated books:
#: real diagrams scored 0.26 and up, page furniture 0.16 and down.  It leans
#: towards recall on purpose --
#: a false positive costs one keystroke in review, a missed diagram costs the page.
SCORE_THRESHOLD = 0.30
#: A proposal this much inside a better-scoring one is a patch of that board,
#: not a second diagram.
CONTAINMENT_LIMIT = 0.7


@dataclass
class Detection:
    """One board found on one page, in page-pixel coordinates."""

    box: tuple[float, float, float, float]
    grid: GridFit
    source: str
    score: float
    verifier_score: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def line_score(self) -> float:
        return self.grid.line_score

    @property
    def checker_score(self) -> float:
        return self.grid.checker_score

    @property
    def rank(self) -> float:
        """Score used for ordering and thresholding, learned model included."""
        if self.verifier_score is None:
            return self.score
        return 0.5 * self.score + 0.5 * self.verifier_score

    def as_dict(self) -> dict:
        return {
            "box": list(self.box), "grid": self.grid.as_dict(), "source": self.source,
            "score": self.score, "verifier_score": self.verifier_score, "meta": self.meta,
        }


def _plausible(box: tuple[float, float, float, float], shape: tuple[int, int]) -> bool:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w < MIN_BOARD_PX or h < MIN_BOARD_PX:
        return False
    if max(w, h) > 0.98 * max(shape):
        return False
    return abs(w - h) <= MAX_ASPECT_SKEW * max(w, h)


def _contour_boxes(gray: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Proposals from connected dark regions.

    A chessboard's dark squares touch at their corners, so once the page is
    thresholded and closed by a pixel or two they form a single component whose
    bounding box is the board.  Boards drawn as outlines instead give us their
    border rectangle from the same pass.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    boxes: list[tuple[float, float, float, float]] = []
    for mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, binary = cv2.threshold(blur, 0, 255, mode | cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = (float(x), float(y), float(x + w), float(y + h))
            if _plausible(box, gray.shape):
                boxes.append(box)
    return boxes


def _lattice_boxes(gray: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Proposals from long straight lines crossing each other.

    This is the path for books that print an unshaded board: no dark squares to
    merge, but eighteen ruled lines that nothing else on a text page imitates.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    span = max(12, min(gray.shape) // 40)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (span, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, span)))
    lattice = cv2.dilate(cv2.bitwise_and(horizontal, vertical), np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(lattice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = (float(x), float(y), float(x + w), float(y + h))
        if _plausible(box, gray.shape):
            boxes.append(box)
    return boxes


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _containment(inner: tuple[float, ...], outer: tuple[float, ...]) -> float:
    """What fraction of ``inner`` lies inside ``outer``."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ((ix1 - ix0) * (iy1 - iy0)) / area if area > 0 else 0.0


def _dedupe(boxes: list[tuple[float, float, float, float]], iou_threshold: float = 0.85) -> list:
    """Drop proposals that are near-copies of one we already have."""
    kept: list[tuple[float, float, float, float]] = []
    for box in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(box, other) < iou_threshold for other in kept):
            kept.append(box)
    return kept


#: A cell holding a piece is far darker than this; a cell holding a table's
#: digit is not.  This is the number that separates a diagram from a crosstable.
OCCUPIED_INK = 0.08


def content_score(gray: np.ndarray, grid: GridFit) -> tuple[float, dict]:
    """How much the lattice's contents look like a chess position, in 0..1.

    Geometry alone cannot tell a chessboard from a tournament crosstable: both
    are a ruled 8-ish by 8-ish grid of square cells, and chess books are full of
    crosstables.  What settles it is what is *in* the cells.  A diagram has a
    handful to a boardful of cells carrying a big centred glyph and the rest
    empty; a crosstable has a small mark in every cell and a lattice that keeps
    going past where the board would end.
    """
    ink = cell_ink(gray, grid)
    occupied = ink > OCCUPIED_INK
    count = int(occupied.sum())
    mean_ink = float(ink[occupied].mean()) if count else 0.0

    if count < 2 or count > 44:
        # Fewer than two men is not a position anyone would print, and more
        # than forty-four cells occupied is not a chess position at all.
        occupancy = 0.0
    else:
        occupancy = min(1.0, mean_ink / 0.15)

    # Does the lattice stop where a board's lattice stops?  Sample one step
    # beyond each outer line: a board has nothing there, a table has its next rule.
    profile_x, profile_y = edge_profile(gray, axis=1), edge_profile(gray, axis=0)
    beyond = max(
        _profile_at(profile_x, grid.x0 - grid.step_x),
        _profile_at(profile_x, grid.x1 + grid.step_x),
        _profile_at(profile_y, grid.y0 - grid.step_y),
        _profile_at(profile_y, grid.y1 + grid.step_y),
    )
    extension = float(np.clip(1.0 - (beyond - 0.35) / 0.4, 0.0, 1.0))

    meta = {
        "occupied_cells": count,
        "mean_ink": round(mean_ink, 4),
        "occupancy_score": round(occupancy, 4),
        "lattice_extends": round(beyond, 4),
    }
    return occupancy * extension, meta


def _profile_at(profile: np.ndarray, position: float) -> float:
    if position < 0 or position >= len(profile):
        return 0.0
    return float(np.interp(position, np.arange(len(profile), dtype=np.float32), profile))


def _frame_score(gray: np.ndarray, grid: GridFit) -> float:
    """Evidence that the board's outer frame is really there.

    Used for the fallback hypothesis below, where the interior rules cannot be
    seen and the frame is all we have to go on.
    """
    profile_x, profile_y = edge_profile(gray, axis=1), edge_profile(gray, axis=0)
    edges = [
        _profile_at(profile_x, grid.x0), _profile_at(profile_x, grid.x1),
        _profile_at(profile_y, grid.y0), _profile_at(profile_y, grid.y1),
    ]
    return float(min(edges))


def verify_proposal(
    gray: np.ndarray,
    box: tuple[float, float, float, float],
    margin_frac: float = 0.10,
) -> tuple[GridFit, float, dict] | None:
    """Fit the lattice inside one proposal and score it.

    The proposal is grown a little first, because a board's outermost line is
    often exactly on the box edge where the line fitter cannot see it, and
    because coordinate labels tend to sit just outside.  The returned grid is in
    page coordinates, not crop coordinates; the score is the geometry score
    tempered by what the cells contain.

    Two hypotheses are tried.  The first is the fitted lattice, which is what
    works when the interior rules survive.  The second is that the proposal's
    own box *is* the board, divided into eight -- which is how a reader handles
    a diagram whose interior rules have been washed out by a bad scan, and
    which the centring check can confirm or refute on its own.
    """
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = box
    margin = margin_frac * max(x1 - x0, y1 - y0)
    cx0, cy0 = int(max(0, x0 - margin)), int(max(0, y0 - margin))
    cx1, cy1 = int(min(w, x1 + margin)), int(min(h, y1 + margin))
    crop = gray[cy0:cy1, cx0:cx1]
    if min(crop.shape[:2]) < MIN_BOARD_PX:
        return None

    candidates: list[tuple[GridFit, float]] = []
    try:
        fitted = fit_grid(crop, search_frac=0.28)
        candidates.append((fitted, fitted.score))
    except ValueError:
        pass

    inner = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)
    if abs((inner[2] - inner[0]) - (inner[3] - inner[1])) <= MAX_ASPECT_SKEW * (inner[2] - inner[0]):
        framed = GridFit.from_box(inner)
        framed = framed.with_scores(_frame_score(crop, framed), checkerboard_score(crop, framed))
        candidates.append((framed, framed.score))

    best: tuple[GridFit, float, dict] | None = None
    for grid, geometry in candidates:
        if grid.step_x <= 0 or grid.step_y <= 0:
            continue
        if abs(grid.step_x - grid.step_y) > MAX_ASPECT_SKEW * max(grid.step_x, grid.step_y):
            continue
        content, meta = content_score(crop, grid)
        score = geometry * content
        if best is None or score > best[1]:
            best = (grid.translated(cx0, cy0), score, meta)
    return best


def detect_boards(
    gray: np.ndarray,
    extra_proposals: list[tuple[float, float, float, float]] | None = None,
    threshold: float = SCORE_THRESHOLD,
) -> list[Detection]:
    """Find every board diagram on a rendered page.

    ``extra_proposals`` carries boxes the PDF itself gave us -- embedded images
    and clusters of vector squares -- which are usually exact and cost nothing.
    """
    sources: list[tuple[str, tuple[float, float, float, float]]] = []
    for box in extra_proposals or []:
        if _plausible(box, gray.shape):
            sources.append(("pdf", box))
    for box in _contour_boxes(gray):
        sources.append(("contour", box))
    for box in _lattice_boxes(gray):
        sources.append(("lattice", box))

    seen: list[tuple[float, float, float, float]] = []
    detections: list[Detection] = []
    for source, box in sources:
        if any(_iou(box, other) > 0.9 for other in seen):
            continue
        seen.append(box)
        verified = verify_proposal(gray, box)
        if verified is None:
            continue
        grid, score, meta = verified
        if score < threshold:
            continue
        detections.append(Detection(box=grid.box, grid=grid, source=source, score=score, meta=meta))

    # Two proposal sources landing on the same board is the normal case; keep
    # the better fit of the two.  A box mostly inside a better one is dropped
    # even when their overlap is small: that is a patch of a board being read as
    # a board of its own, which a chess page never actually contains.
    detections.sort(key=lambda d: -d.score)
    kept: list[Detection] = []
    for detection in detections:
        if any(_iou(detection.box, other.box) >= 0.5 or _containment(detection.box, other.box) > CONTAINMENT_LIMIT
               for other in kept):
            continue
        kept.append(detection)
    kept.sort(key=lambda d: (d.box[1], d.box[0]))  # reading order down the page
    return kept


def crop_board(gray: np.ndarray, detection: Detection, margin_frac: float = 0.06) -> tuple[np.ndarray, GridFit]:
    """Cut the board out of the page, with the grid rebased to the cut-out."""
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = detection.box
    margin = margin_frac * max(x1 - x0, y1 - y0)
    cx0, cy0 = int(max(0, round(x0 - margin))), int(max(0, round(y0 - margin)))
    cx1, cy1 = int(min(w, round(x1 + margin))), int(min(h, round(y1 + margin)))
    return gray[cy0:cy1, cx0:cx1].copy(), detection.grid.translated(-cx0, -cy0)
