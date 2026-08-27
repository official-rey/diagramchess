import random

import numpy as np
import pytest

from diagramchess.board import BoardMatrix
from diagramchess.grid import extract_squares
from diagramchess.labels import LABEL_TO_INDEX, LABELS
from diagramchess.pieces import available_piece_sets
from diagramchess.predict import BoardReading, ExemplarBank, Predictor
from diagramchess.render import DiagramStyle, render_diagram
from diagramchess.synth import degrade

POSITION = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
OTHER = "2kr3r/ppp2ppp/2n1b3/3q4/3P4/2N1BN2/PPP2PPP/R2Q1RK1 w - - 0 1"


def _squares(fen, piece_set, seed=None, cell_px=44):
    board = BoardMatrix.from_fen(fen)
    rendered = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=cell_px))
    image = degrade(rendered.image, random.Random(seed)) if seed is not None else rendered.image
    return extract_squares(image, rendered.grid, size=48), board


@pytest.fixture(params=[s.name for s in available_piece_sets()])
def piece_set(request):
    return next(s for s in available_piece_sets() if s.name == request.param)


def test_an_empty_bank_says_nothing():
    bank = ExemplarBank()
    assert len(bank) == 0
    probabilities = bank.probabilities(np.zeros((3, 48, 48), np.uint8))
    assert probabilities.shape == (3, len(LABELS))
    assert np.allclose(probabilities, 1 / len(LABELS))


def test_a_bank_recognises_the_style_it_was_built_from(piece_set):
    """The whole point: one book, one figurine font, so matching is nearly exact."""
    squares, board = _squares(POSITION, piece_set)
    bank = ExemplarBank()
    bank.add(squares, np.array([LABEL_TO_INDEX[c] for c in board.flat()]))

    # A different position in the same style, printed a little worse.
    other_squares, other_board = _squares(OTHER, piece_set, seed=3)
    probabilities = bank.probabilities(other_squares)
    predicted = [LABELS[i] for i in probabilities.argmax(axis=1)]
    correct = sum(1 for a, b in zip(predicted, other_board.flat()) if a == b)
    assert correct >= 60, f"{correct}/64 from exemplars alone"


def test_a_bank_cannot_invent_a_class_it_has_never_seen():
    bank = ExemplarBank()
    images = np.random.randint(0, 255, (4, 48, 48), dtype=np.uint8)
    bank.add(images, np.array([LABEL_TO_INDEX["K"]] * 4))
    probabilities = bank.probabilities(images)
    assert probabilities[:, LABEL_TO_INDEX["q"]].max() == 0.0
    assert probabilities[:, LABEL_TO_INDEX["K"]].min() > 0.99


def test_predictor_needs_something_to_read_with():
    predictor = Predictor()
    assert not predictor.has_model
    with pytest.raises(RuntimeError, match="nothing to read with"):
        predictor.read_squares(np.zeros((64, 48, 48), np.uint8))


def test_predictor_reads_with_exemplars_alone(piece_set):
    """Without a trained model, one verified diagram is enough to read the next."""
    squares, board = _squares(POSITION, piece_set)
    bank = ExemplarBank()
    bank.add(squares, np.array([LABEL_TO_INDEX[c] for c in board.flat()]))

    other_squares, other_board = _squares(OTHER, piece_set, seed=5)
    reading = Predictor().read_squares(other_squares, bank)
    assert reading.source == "exemplars"
    correct = sum(1 for a, b in zip(reading.labels, other_board.flat()) if a == b)
    assert correct >= 58


def test_reading_rejects_the_wrong_number_of_squares():
    with pytest.raises(ValueError, match="64"):
        Predictor().read_squares(np.zeros((10, 48, 48), np.uint8))


def test_reading_to_board_carries_orientation_and_caption():
    labels = BoardMatrix.from_fen(POSITION).flat()
    reading = BoardReading(labels, [0.9] * 64, np.zeros((64, 13), np.float32))
    board = reading.to_board(caption="Diagram 3: Black to play")
    assert board.side_to_move == "b"
    assert board.orientation == "white"
    assert board.to_fen().startswith("r1bqkbnr/")


def test_min_confidence_is_the_worst_square():
    confidence = [0.99] * 64
    confidence[10], confidence[20] = 0.4, 0.2
    reading = BoardReading(["."] * 64, confidence, np.zeros((64, 13), np.float32))
    assert reading.min_confidence == pytest.approx(0.2)


def test_read_board_cuts_and_reads_in_one_step(piece_set):
    """The one-shot path: a board picture and its grid in, a position out."""
    board = BoardMatrix.from_fen(POSITION)
    rendered = render_diagram(board, DiagramStyle(piece_set=piece_set, cell_px=44))
    bank = ExemplarBank()
    squares, _ = _squares(POSITION, piece_set)
    bank.add(squares, np.array([LABEL_TO_INDEX[c] for c in board.flat()]))

    reading, crops = Predictor().read_board(rendered.image, rendered.grid, bank)
    assert crops.shape == (64, 48, 48)
    assert reading.labels == board.flat()
