"""Where training examples come from, and how they are jittered.

Two sources feed the classifier.  Synthetic squares are unlimited and free but
only approximate your book.  Verified squares are the ones you corrected by hand
in the review UI: there are few of them and they are exactly right.  The mixing
here is deliberately tilted towards the verified ones, because a hundred squares
cut from the book you are actually reading are worth more than a hundred
thousand drawn from our guess at what books look like.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

from .model import normalise
from .pieces import PieceSet, available_piece_sets
from .synth import synth_board


def augment_crop(square: np.ndarray, rng: random.Random) -> np.ndarray:
    """Jitter one already-cut square.

    Verified crops come from real pages and are already realistic, so this is
    much gentler than the synthetic degradation: enough to stop the net
    memorising a hundred exact crops, not enough to invent artefacts.
    """
    size = square.shape[0]
    out = square

    shift = 0.05 * size
    matrix = np.array([
        [rng.uniform(0.94, 1.06), 0.0, rng.uniform(-shift, shift)],
        [0.0, rng.uniform(0.94, 1.06), rng.uniform(-shift, shift)],
    ], dtype=np.float32)
    matrix[0, 2] += (size - matrix[0, 0] * size) / 2
    matrix[1, 2] += (size - matrix[1, 1] * size) / 2
    out = cv2.warpAffine(out, matrix, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.9))
    if rng.random() < 0.6:
        noise = np.random.default_rng(rng.getrandbits(64))
        out = np.clip(out.astype(np.float32) + noise.normal(0, rng.uniform(1, 6), out.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        gain, bias = rng.uniform(0.9, 1.1), rng.uniform(-10, 10)
        out = np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    return out


@dataclass
class VerifiedSquares:
    """Hand-corrected squares, grouped by the book they came from."""

    images: np.ndarray          # (n, size, size) uint8
    labels: np.ndarray          # (n,) int64
    book_ids: np.ndarray        # (n,) int64

    def __len__(self) -> int:
        return len(self.images)

    def for_books(self, book_ids: set[int]) -> "VerifiedSquares":
        mask = np.isin(self.book_ids, list(book_ids))
        return VerifiedSquares(self.images[mask], self.labels[mask], self.book_ids[mask])

    def split(self, fraction: float, seed: int = 0) -> tuple["VerifiedSquares", "VerifiedSquares"]:
        """Split off a validation share, keeping each book on one side or the other.

        Splitting by book rather than by square is the only honest way to
        measure this: two crops of the same rook from the same page are nearly
        the same image, and letting one land in training and one in validation
        would report an accuracy the model does not have on the next book.
        """
        books = sorted(set(int(b) for b in self.book_ids))
        rng = random.Random(seed)
        rng.shuffle(books)
        cut = max(1, int(round(len(books) * fraction))) if len(books) > 1 else 0
        held = set(books[:cut])
        return self.for_books(set(books) - held), self.for_books(held)


def synth_stream(
    seed: int,
    piece_sets: list[PieceSet] | None = None,
    square_size: int = 48,
    empty_keep: float = 0.28,
):
    """Yield ``(square, label)`` pairs from freshly drawn diagrams, forever.

    ``empty_keep`` thins the empty squares out.  Better than half of a real
    board is empty, and a net trained on that distribution learns to answer
    'empty' whenever it hesitates -- which is the one mistake that is invisible
    in review until you notice a missing piece.
    """
    rng = random.Random(seed)
    sets = piece_sets or available_piece_sets()
    while True:
        board = synth_board(rng, sets, square_size=square_size)
        order = list(range(64))
        rng.shuffle(order)
        for i in order:
            label = int(board.labels[i])
            if label == 0 and rng.random() > empty_keep:
                continue
            yield board.squares[i], label


def make_batches(
    seed: int,
    batch_size: int,
    piece_sets: list[PieceSet] | None = None,
    square_size: int = 48,
    verified: VerifiedSquares | None = None,
    verified_fraction: float = 0.35,
):
    """Yield normalised ``(x, y)`` batches mixing synthetic and verified squares."""
    rng = random.Random(seed ^ 0x5EED)
    stream = synth_stream(seed, piece_sets, square_size)
    have_verified = verified is not None and len(verified) > 0

    while True:
        images: list[np.ndarray] = []
        labels: list[int] = []
        while len(images) < batch_size:
            if have_verified and rng.random() < verified_fraction:
                index = rng.randrange(len(verified))
                images.append(augment_crop(verified.images[index], rng))
                labels.append(int(verified.labels[index]))
            else:
                square, label = next(stream)
                images.append(square)
                labels.append(label)
        yield normalise(np.stack(images)), np.array(labels, dtype=np.int64)


def fixed_set(
    seed: int,
    count: int,
    piece_sets: list[PieceSet] | None = None,
    square_size: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """A reproducible evaluation set drawn from the given styles."""
    stream = synth_stream(seed, piece_sets, square_size, empty_keep=0.28)
    images, labels = [], []
    for _ in range(count):
        square, label = next(stream)
        images.append(square)
        labels.append(label)
    return np.stack(images), np.array(labels, dtype=np.int64)
