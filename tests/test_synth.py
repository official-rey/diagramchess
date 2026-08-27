import random

import numpy as np
import pytest

from diagramchess.board import BoardMatrix
from diagramchess.dataset import VerifiedSquares, augment_crop, fixed_set, make_batches, synth_stream
from diagramchess.labels import EMPTY, LABEL_TO_INDEX, NUM_CLASSES
from diagramchess.pieces import SYMBOLS, available_piece_sets
from diagramchess.render import DiagramStyle, random_style, render_diagram
from diagramchess.synth import degrade, jitter_grid, random_position, synth_board, synth_squares


def test_every_piece_set_draws_every_piece():
    sets = available_piece_sets()
    assert sets, "no piece artwork found at all"
    for piece_set in sets:
        for symbol in SYMBOLS:
            art = piece_set.render(symbol, 48)
            assert art.shape == (48, 48, 4)
            assert art[:, :, 3].max() == 255, f"{piece_set.name} draws nothing for {symbol}"


def test_white_and_black_pieces_are_drawn_differently():
    for piece_set in available_piece_sets():
        white = piece_set.render("K", 64)
        black = piece_set.render("k", 64)
        assert not np.array_equal(white, black)


def test_random_positions_are_plausible():
    rng = random.Random(1)
    for _ in range(40):
        board = random_position(rng)
        counts = board.counts()
        assert counts.get("K", 0) <= 1 and counts.get("k", 0) <= 1
        assert counts.get("P", 0) <= 8 and counts.get("p", 0) <= 8
        assert sum(counts.values()) <= 32


def test_a_rendered_board_matches_the_grid_it_reports():
    board = BoardMatrix.from_fen("8/8/4k3/8/2K5/8/6R1/8 w - - 0 1")
    piece_set = available_piece_sets()[0]
    rendered = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=40))
    assert rendered.grid.step_x == 40
    assert rendered.image.shape[0] == rendered.image.shape[1]
    assert rendered.grid.x1 <= rendered.image.shape[1]


def test_a_random_style_always_shows_the_board():
    """Shading, rules and a frame are each optional; all three missing is not a board."""
    rng = random.Random(7)
    sets = available_piece_sets()
    for _ in range(200):
        style = random_style(rng, rng.choice(sets))
        assert style.checkered or style.grid_line is not None or style.border_width > 0


def test_degrade_keeps_the_picture_the_same_size_and_type():
    rng = random.Random(2)
    image = np.full((200, 200), 240, np.uint8)
    image[50:150, 50:150] = 30
    for _ in range(20):
        out = degrade(image, rng)
        assert out.shape == image.shape and out.dtype == np.uint8


def test_degrade_is_reproducible_from_its_seed():
    image = np.full((120, 120), 200, np.uint8)
    first = degrade(image, random.Random(9))
    second = degrade(image, random.Random(9))
    assert np.array_equal(first, second)


def test_jitter_moves_the_grid_but_not_far():
    from diagramchess.grid import GridFit

    grid = GridFit(10.0, 20.0, 40.0, 40.0)
    rng = random.Random(0)
    for _ in range(50):
        moved = jitter_grid(grid, rng, amount=0.06)
        assert abs(moved.x0 - grid.x0) <= 0.06 * grid.step_x + 1e-9
        assert abs(moved.step_x - grid.step_x) <= 0.03 * grid.step_x + 1e-9


def test_a_synthetic_board_carries_its_own_labels():
    sample = synth_board(random.Random(3), available_piece_sets())
    assert sample.squares.shape == (64, 48, 48)
    assert sample.labels.shape == (64,)
    assert sample.labels.max() < NUM_CLASSES
    flat = sample.board.flat()
    assert [LABEL_TO_INDEX[c] for c in flat] == list(sample.labels)


def test_synthetic_squares_are_not_mostly_empty():
    """A board is well over half empty, and training on that distribution teaches
    the net to answer 'empty' whenever it hesitates -- the one mistake that is
    invisible in review until you notice a missing piece.  Thinning brings it to
    roughly a third, which is still the largest class but no longer the safe guess."""
    images, labels = synth_squares(600, random.Random(4))
    assert images.shape == (600, 48, 48)
    empty_share = (labels == LABEL_TO_INDEX[EMPTY]).mean()
    assert 0.05 < empty_share < 0.45, empty_share
    assert len(set(labels.tolist())) >= 12


def test_batches_are_normalised_and_the_right_shape():
    import itertools

    for x, y in itertools.islice(make_batches(0, 64), 2):
        assert x.shape == (64, 48, 48) and y.shape == (64,)
        assert abs(float(x.mean())) < 0.2
        assert 0.5 < float(x.std()) < 2.0


def test_verified_squares_are_mixed_in_when_given():
    import itertools

    marker = np.full((5, 48, 48), 7, np.uint8)
    verified = VerifiedSquares(marker, np.full(5, 3, np.int64), np.ones(5, np.int64))
    labels = []
    for _, y in itertools.islice(make_batches(1, 128, verified=verified, verified_fraction=0.5), 4):
        labels.extend(y.tolist())
    assert labels.count(3) > 100, "verified squares are not reaching the batches"


def test_augment_keeps_a_crop_recognisable():
    rng = random.Random(1)
    crop = np.full((48, 48), 240, np.uint8)
    crop[14:34, 18:30] = 20
    for _ in range(30):
        out = augment_crop(crop, rng)
        assert out.shape == (48, 48) and out.dtype == np.uint8
        assert out.min() < 120 < out.max()


def test_fixed_set_is_reproducible():
    a_images, a_labels = fixed_set(11, 200)
    b_images, b_labels = fixed_set(11, 200)
    assert np.array_equal(a_images, b_images)
    assert np.array_equal(a_labels, b_labels)
