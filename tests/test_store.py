import numpy as np
import pytest

from diagramchess.board import BoardMatrix
from diagramchess.dataset import VerifiedSquares
from diagramchess.labels import LABEL_TO_INDEX

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _add_diagram(workspace, book_id, page=0, index=0):
    return workspace.add_diagram(
        book_id, page, (10.0 * index, 20.0, 110.0, 120.0),
        {"x0": 0, "y0": 0, "step_x": 12, "step_y": 12}, "contour", 0.9,
        f"crops/x{index}.png", caption="Diagram 1: White to play",
    )


def test_books_are_recognised_by_content_not_by_path(workspace):
    first = workspace.add_book("/books/a.pdf", "digest-1", 100, 200)
    moved = workspace.add_book("/elsewhere/a.pdf", "digest-1", 100, 200)
    assert first == moved
    assert workspace.book(first)["path"] == "/elsewhere/a.pdf"
    assert len(workspace.books()) == 1


def test_the_same_detection_is_not_stored_twice(workspace):
    book = workspace.add_book("/books/a.pdf", "d", 10, 200)
    assert _add_diagram(workspace, book) is not None
    assert _add_diagram(workspace, book) is None       # same page and position
    assert _add_diagram(workspace, book, index=1) is not None


def test_a_prediction_does_not_overwrite_a_human_reading(workspace):
    book = workspace.add_book("/books/a.pdf", "d", 10, 200)
    diagram = _add_diagram(workspace, book)
    labels = BoardMatrix.from_fen(START).flat()

    workspace.set_prediction(diagram, ["."] * 64, [0.4] * 64, "8/8/8/8/8/8/8/8 w - - 0 1",
                             "white", "w", None)
    workspace.save_review(diagram, labels, "white", "w", START)
    assert workspace.diagram(diagram)["fen"] == START

    # A later model run must leave the verified reading alone.
    workspace.set_prediction(diagram, ["."] * 64, [0.9] * 64, "8/8/8/8/8/8/8/8 w - - 0 1",
                             "black", "b", None)
    row = workspace.diagram(diagram)
    assert row["fen"] == START
    assert row["orientation"] == "white"
    assert row["side_to_move"] == "w"
    # but it still records what the model thought, for measuring accuracy later
    assert row["predicted_fen"] == "8/8/8/8/8/8/8/8 w - - 0 1"


def test_verified_squares_come_back_with_their_book(workspace, tmp_path):
    book = workspace.add_book("/books/a.pdf", "d", 10, 200)
    diagram = _add_diagram(workspace, book)
    np.save(workspace.squares_path(diagram), np.zeros((64, 48, 48), np.uint8))
    labels = BoardMatrix.from_fen(START).flat()
    workspace.save_review(diagram, labels, "white", "w", START)

    verified = workspace.verified_squares()
    assert len(verified) == 64
    assert verified.images.shape == (64, 48, 48)
    assert verified.labels[0] == LABEL_TO_INDEX["r"]
    assert set(verified.book_ids) == {book}


def test_verified_squares_skip_diagrams_whose_crops_are_gone(workspace):
    book = workspace.add_book("/books/a.pdf", "d", 10, 200)
    diagram = _add_diagram(workspace, book)
    workspace.save_review(diagram, ["."] * 64, "white", "w", "8/8/8/8/8/8/8/8 w - - 0 1")
    assert len(workspace.verified_squares()) == 0     # no .npy on disk


def test_uncertain_ordering_puts_the_worst_first(workspace):
    book = workspace.add_book("/books/a.pdf", "d", 10, 200)
    confident = _add_diagram(workspace, book, index=0)
    unsure = _add_diagram(workspace, book, index=1)
    workspace.set_prediction(confident, ["."] * 64, [0.99] * 64, "f", "white", "w", None)
    workspace.set_prediction(unsure, ["."] * 64, [0.3] * 64, "f", "white", "w", None)
    order = [row["id"] for row in workspace.diagrams(order="uncertain")]
    assert order[0] == unsure


def test_models_are_versioned_and_one_is_active(workspace):
    first = workspace.register_model("/m/1.pt", "2026-01-01", {"val_accuracy": 0.9})
    assert workspace.active_model()["id"] == first
    second = workspace.register_model("/m/2.pt", "2026-01-02", {"val_accuracy": 0.95})
    assert workspace.active_model()["id"] == second
    assert len(workspace.models()) == 2
    third = workspace.register_model("/m/3.pt", "2026-01-03", {}, activate=False)
    assert workspace.active_model()["id"] == second


def test_stats_counts_what_is_there(workspace):
    book = workspace.add_book("/books/a.pdf", "d", 10, 200)
    _add_diagram(workspace, book, index=0)
    diagram = _add_diagram(workspace, book, index=1)
    workspace.save_review(diagram, ["."] * 64, "white", "w", "8/8/8/8/8/8/8/8 w - - 0 1")
    stats = workspace.stats()
    assert stats["books"] == 1 and stats["diagrams"] == 2
    assert stats["verified"] == 1 and stats["pending"] == 1
    assert stats["labelled_squares"] == 64


def test_verified_split_keeps_a_book_on_one_side():
    """Two crops of the same rook must not straddle the split."""
    images = np.zeros((30, 48, 48), np.uint8)
    labels = np.zeros(30, np.int64)
    books = np.array([1] * 10 + [2] * 10 + [3] * 10, np.int64)
    train, held = VerifiedSquares(images, labels, books).split(0.34, seed=1)
    assert len(train) + len(held) == 30
    assert not (set(train.book_ids) & set(held.book_ids))
