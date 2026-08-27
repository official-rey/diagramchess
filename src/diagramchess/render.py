"""Drawing chess diagrams ourselves.

This is how the tool bootstraps: before it has seen a single page of your book
it can draw hundreds of thousands of diagrams whose contents it already knows,
in a range of styles wide enough that the classifier learns 'rook' rather than
'that particular rook'.  The same renderer builds the fixture PDFs the tests
run against.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import BoardMatrix
from .grid import GridFit
from .labels import EMPTY
from .pieces import PieceSet


@dataclass
class DiagramStyle:
    """How one book prints its diagrams.

    Real books vary along every one of these axes, and a classifier trained on
    one point in this space transfers badly, so the synthesiser samples it.
    """

    piece_set: PieceSet
    cell_px: int = 48
    light: int = 245
    dark: int = 175
    checkered: bool = True
    grid_line: int | None = 110      # shade of the interior rules, None for none
    grid_width: int = 1
    border_width: int = 2
    border_shade: int = 40
    outer_margin: int = 10           # white space kept around the board
    coordinates: bool = False        # a-h and 1-8 printed outside the board
    coordinate_shade: int = 90
    piece_scale: float = 0.86
    piece_ink: int = 20              # how black the black pieces print
    white_piece_fill: int = 252      # how white the white pieces print
    background: int = 255


@dataclass
class RenderedDiagram:
    image: np.ndarray   # grayscale uint8
    grid: GridFit       # where the board sits inside ``image``
    board: BoardMatrix
    style: DiagramStyle = field(repr=False, default=None)


def _blend(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int, style: DiagramStyle) -> None:
    """Alpha-composite one piece onto the grayscale canvas at (x, y)."""
    h, w = rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + w), min(ch, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    piece = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = piece[:, :, 3:4].astype(np.float32) / 255.0
    # The artwork is drawn in colour; its luminance already distinguishes the
    # white pieces' pale bodies from the black pieces' dark ones.  We only
    # stretch that luminance to the printing darkness this style uses.
    luma = piece[:, :, :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    luma = style.piece_ink + (style.white_piece_fill - style.piece_ink) * (luma / 255.0)
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(
        region * (1 - alpha[:, :, 0]) + luma * alpha[:, :, 0], 0, 255
    ).astype(np.uint8)


def render_diagram(board: BoardMatrix, style: DiagramStyle) -> RenderedDiagram:
    """Draw a board matrix as a printed-looking diagram."""
    cell = style.cell_px
    board_px = cell * 8
    coord_pad = int(cell * 0.55) if style.coordinates else 0
    pad = style.outer_margin + style.border_width + coord_pad
    size_x = board_px + 2 * pad
    size_y = board_px + 2 * pad
    canvas = np.full((size_y, size_x), style.background, np.uint8)

    origin_x = origin_y = pad
    grid = GridFit(float(origin_x), float(origin_y), float(cell), float(cell))

    for row in range(8):
        for col in range(8):
            shade = style.light
            if style.checkered and (row + col) % 2 == 1:
                shade = style.dark
            x0, y0 = origin_x + col * cell, origin_y + row * cell
            canvas[y0:y0 + cell, x0:x0 + cell] = shade

    if style.grid_line is not None:
        for k in range(9):
            x = origin_x + k * cell
            y = origin_y + k * cell
            cv2.line(canvas, (x, origin_y), (x, origin_y + board_px), style.grid_line, style.grid_width)
            cv2.line(canvas, (origin_x, y), (origin_x + board_px, y), style.grid_line, style.grid_width)

    if style.border_width > 0:
        half = style.border_width // 2
        cv2.rectangle(
            canvas,
            (origin_x - half - 1, origin_y - half - 1),
            (origin_x + board_px + half, origin_y + board_px + half),
            style.border_shade, style.border_width,
        )

    if style.coordinates:
        scale = cell / 64.0
        for col in range(8):
            label = "abcdefgh"[col]
            cv2.putText(canvas, label,
                        (origin_x + col * cell + int(cell * 0.38), origin_y + board_px + int(coord_pad * 0.9)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, style.coordinate_shade, 1, cv2.LINE_AA)
        for row in range(8):
            label = str(8 - row)
            cv2.putText(canvas, label,
                        (origin_x - int(coord_pad * 0.8), origin_y + row * cell + int(cell * 0.62)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, style.coordinate_shade, 1, cv2.LINE_AA)

    piece_px = max(4, int(round(cell * style.piece_scale)))
    offset = (cell - piece_px) // 2
    for row in range(8):
        for col in range(8):
            symbol = board.rows[row][col]
            if symbol == EMPTY:
                continue
            art = style.piece_set.render(symbol, piece_px)
            _blend(canvas, art, origin_x + col * cell + offset, origin_y + row * cell + offset, style)

    return RenderedDiagram(image=canvas, grid=grid, board=board, style=style)


def random_style(rng: random.Random, piece_set: PieceSet, cell_px: int | None = None) -> DiagramStyle:
    """Sample a plausible book's diagram style.

    The ranges here are the tool's opinion about what printed chess diagrams
    look like: light squares from near-white to light gray, dark squares that
    stay clearly darker, rules that are sometimes there and sometimes not.
    """
    light = rng.randint(215, 255)
    checkered = rng.random() < 0.85
    dark = rng.randint(140, max(145, light - 35)) if checkered else light
    grid_line = rng.choice([None, rng.randint(60, 150), rng.randint(60, 150)])
    border_width = rng.choice([0, 1, 2, 3])
    # A board has to be visible as a board.  Shading, interior rules and a
    # frame are each optional in print, but no book leaves out all three --
    # that would be pieces floating on blank paper.
    if not checkered and grid_line is None:
        if border_width == 0:
            border_width = rng.choice([1, 2, 3])
        grid_line = rng.randint(60, 150)
    return DiagramStyle(
        piece_set=piece_set,
        cell_px=cell_px if cell_px is not None else rng.randint(20, 64),
        light=light,
        dark=dark,
        checkered=checkered,
        grid_line=grid_line,
        grid_width=1,
        border_width=border_width,
        border_shade=rng.randint(0, 90),
        outer_margin=rng.randint(4, 22),
        coordinates=rng.random() < 0.35,
        coordinate_shade=rng.randint(60, 150),
        piece_scale=rng.uniform(0.72, 0.98),
        piece_ink=rng.randint(0, 60),
        white_piece_fill=rng.randint(235, 255),
        background=rng.randint(240, 255),
    )
