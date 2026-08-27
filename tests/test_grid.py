import random

import numpy as np
import pytest

from diagramchess.board import BoardMatrix
from diagramchess.grid import (
    GridFit, MIN_CELL_PX, cell_ink, cell_shades,
    checkerboard_score, extract_squares, fit_grid,
)
from diagramchess.labels import EMPTY
from diagramchess.pieces import available_piece_sets
from diagramchess.render import DiagramStyle, random_style, render_diagram

POSITION = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"


@pytest.fixture(params=[s.name for s in available_piece_sets()])
def piece_set(request):
    return next(s for s in available_piece_sets() if s.name == request.param)


def test_fit_finds_the_true_lattice(piece_set):
    board = BoardMatrix.from_fen(POSITION)
    rendered = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=40, coordinates=True))
    fitted = fit_grid(rendered.image)
    assert abs(fitted.x0 - rendered.grid.x0) < 0.1 * rendered.grid.step_x
    assert abs(fitted.y0 - rendered.grid.y0) < 0.1 * rendered.grid.step_y
    assert abs(fitted.step_x - rendered.grid.step_x) < 0.03 * rendered.grid.step_x


def test_fit_survives_a_heavier_frame_than_the_interior_rules(piece_set):
    """The frame is the strongest line on the page; the lattice must not follow it."""
    board = BoardMatrix.from_fen(POSITION)
    style = DiagramStyle(piece_set=piece_set, cell_px=40, border_width=4,
                         border_shade=0, grid_line=150)
    rendered = render_diagram(board, style)
    fitted = fit_grid(rendered.image)
    assert abs(fitted.step_x - rendered.grid.step_x) < 0.03 * rendered.grid.step_x


def test_fit_is_accurate_across_random_styles():
    rng = random.Random(4)
    sets = available_piece_sets()
    offsets = []
    for _ in range(40):
        board = BoardMatrix.from_fen(POSITION)
        style = random_style(rng, rng.choice(sets))
        style.cell_px = rng.randint(26, 56)
        rendered = render_diagram(board, style)
        fitted = fit_grid(rendered.image)
        offsets.append(abs(fitted.x0 - rendered.grid.x0) / rendered.grid.step_x)
    assert np.median(offsets) < 0.06
    assert sum(o > 0.12 for o in offsets) <= 2


def test_checkerboard_score_is_high_on_a_shaded_board_and_zero_on_a_flat_one(piece_set):
    board = BoardMatrix.from_fen(POSITION)
    shaded = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=40))
    assert checkerboard_score(shaded.image, shaded.grid) > 0.7

    flat = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=40, checkered=False))
    assert checkerboard_score(flat.image, flat.grid) < 0.3


def test_cell_shades_read_the_square_not_the_piece(piece_set):
    """A board full of pieces must still report its own light and dark squares."""
    board = BoardMatrix.from_fen(POSITION)
    rendered = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=44,
                                                  light=245, dark=175))
    shades = cell_shades(rendered.image, rendered.grid)
    light = shades[(np.indices((8, 8)).sum(axis=0) % 2) == 0]
    dark = shades[(np.indices((8, 8)).sum(axis=0) % 2) == 1]
    assert light.mean() > dark.mean() + 40


def test_cell_ink_marks_the_occupied_squares(piece_set):
    board = BoardMatrix.from_fen(POSITION)
    rendered = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=44))
    ink = cell_ink(rendered.image, rendered.grid)
    occupied = np.array([[c != EMPTY for c in row] for row in board.rows])
    assert (ink[occupied] > 0.08).mean() > 0.95
    assert (ink[~occupied] <= 0.08).mean() > 0.95


def test_extract_squares_shape_and_order(piece_set):
    board = BoardMatrix.from_fen(POSITION)
    rendered = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=40))
    squares = extract_squares(rendered.image, rendered.grid, size=48)
    assert squares.shape == (64, 48, 48)
    # The top-left square holds a black rook; e4 in this position is empty.
    assert squares[0].std() > squares[8 * 3 + 4].std()


def test_extract_squares_pads_rather_than_clipping_at_the_edge(piece_set):
    """A grid that runs off the crop must still yield equally scaled squares."""
    board = BoardMatrix.from_fen(POSITION)
    rendered = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=40,
                                                  outer_margin=0, border_width=0))
    squares = extract_squares(rendered.image, rendered.grid, size=48, pad_frac=0.2)
    assert squares.shape == (64, 48, 48)
    assert squares.dtype == np.uint8


def test_tiny_crops_are_refused():
    with pytest.raises(ValueError):
        fit_grid(np.zeros((8 * MIN_CELL_PX - 1, 200), np.uint8))
    with pytest.raises(ValueError):
        fit_grid(np.zeros((200, 200, 3), np.uint8))


def test_grid_fit_serialises():
    grid = GridFit(1.5, 2.5, 10.0, 10.5, 0.8, 0.9)
    assert GridFit.from_dict(grid.as_dict()) == grid
    assert GridFit.from_box((0, 0, 80, 80)).step_x == 10.0
