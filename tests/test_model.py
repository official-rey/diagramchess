import numpy as np
import pytest

from diagramchess.model import MIN_CONTRAST, SQUARE_SIZE, normalise


def _crop(fill, size=SQUARE_SIZE):
    return np.full((1, size, size), fill, np.uint8)


def test_a_flat_crop_stays_flat():
    """The bug this guards against was invisible until a degraded scan.

    Dividing every crop by its own standard deviation stretches an empty
    square -- which has almost none -- until its sensor noise fills the range,
    and the classifier reads the result as a piece.  On a poor scan of a real
    book that put 382 of its errors in the same place: empty squares called
    pieces, scoring below what answering "empty" everywhere would have got.
    """
    flat = _crop(200)
    flat[0, 0, 0] = 201
    assert np.abs(normalise(flat)).max() < 0.05


def test_scanner_noise_on_an_empty_square_is_not_amplified_to_a_piece():
    noise = np.clip(
        np.random.default_rng(0).normal(200, 15, (1, SQUARE_SIZE, SQUARE_SIZE)), 0, 255
    ).astype(np.uint8)
    piece = _crop(240)
    piece[0, 12:36, 12:36] = 30

    # The piece must still come out clearly stronger than the noise; before the
    # floor both were stretched to exactly unit variance and looked alike.
    assert normalise(noise).std() < 0.5 * normalise(piece).std()


def test_print_density_is_still_normalised_away():
    """The mean must go: the same piece printed pale or heavy is the same piece."""
    pale = _crop(250)
    pale[0, 12:36, 12:36] = 170
    heavy = _crop(230)
    heavy[0, 12:36, 12:36] = 150
    assert abs(float(normalise(pale).mean())) < 1e-5
    assert abs(float(normalise(heavy).mean())) < 1e-5


def test_a_high_contrast_crop_is_scaled_by_its_own_spread():
    """Above the floor nothing changes, so ordinary crops are unaffected."""
    piece = _crop(245)
    piece[0, 8:40, 8:40] = 20
    assert piece[0].std() > MIN_CONTRAST
    assert float(normalise(piece).std()) == pytest.approx(1.0, abs=0.01)


def test_normalise_accepts_a_single_crop_or_a_batch():
    single = normalise(np.full((SQUARE_SIZE, SQUARE_SIZE), 128, np.uint8))
    assert single.shape == (1, SQUARE_SIZE, SQUARE_SIZE)
    batch = normalise(np.full((5, SQUARE_SIZE, SQUARE_SIZE), 128, np.uint8))
    assert batch.shape == (5, SQUARE_SIZE, SQUARE_SIZE)
    assert batch.dtype == np.float32
