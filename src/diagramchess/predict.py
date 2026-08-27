"""Reading a board picture into a position.

Two readers are combined.  The neural net knows what chess pieces look like in
general, having been trained on synthetic diagrams and on whatever you have
verified so far.  The exemplar bank knows what pieces look like *in this book*,
because it is a nearest-neighbour lookup over the crops you corrected by hand on
earlier pages of it.

That second reader is why review pays off so quickly.  A book sets its diagrams
in one figurine font throughout, so once you have verified two or three
positions, nearly every piece in the book has an exact match on file, and the
bank is right about squares the general model finds ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .board import BoardMatrix, guess_orientation, guess_side_to_move
from .grid import GridFit, extract_squares
from .labels import NUM_CLASSES, index_label
from .model import SQUARE_SIZE, normalise

#: Crops are matched at this size; the detail above it is print noise.
EXEMPLAR_SIZE = 24


@dataclass
class BoardReading:
    """What a reader made of one board, square by square."""

    labels: list[str]                 # 64, reading order
    confidence: list[float]           # 64, calibrated 0..1
    probabilities: np.ndarray         # (64, 13)
    source: str = "model"

    @property
    def min_confidence(self) -> float:
        return float(min(self.confidence)) if self.confidence else 0.0

    def to_board(self, caption: str | None = None, orientation: str | None = None) -> BoardMatrix:
        rows = [self.labels[r * 8:(r + 1) * 8] for r in range(8)]
        board = BoardMatrix(
            rows,
            orientation=orientation or guess_orientation(rows),
            side_to_move=guess_side_to_move(caption) or "w",
            confidence=[self.confidence[r * 8:(r + 1) * 8] for r in range(8)],
        )
        return board


class ExemplarBank:
    """Nearest-neighbour lookup over crops verified in one book."""

    def __init__(self, size: int = EXEMPLAR_SIZE):
        self.size = size
        self.vectors: np.ndarray = np.zeros((0, size * size), np.float32)
        self.labels: np.ndarray = np.zeros(0, np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def classes(self) -> set[int]:
        return set(int(v) for v in np.unique(self.labels))

    def add(self, images: np.ndarray, labels: np.ndarray) -> None:
        if len(images) == 0:
            return
        self.vectors = np.concatenate([self.vectors, self._embed(images)])
        self.labels = np.concatenate([self.labels, np.asarray(labels, np.int64)])

    def _embed(self, images: np.ndarray) -> np.ndarray:
        """Standardise, shrink and unit-normalise, so matching is a dot product.

        Standardising first is what makes a match survive a darker printing or a
        greyer scan of the same figurine.
        """
        import cv2

        if images.ndim == 2:
            images = images[None, ...]
        resized = np.stack([
            cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            for img in images
        ])
        flat = normalise(resized).reshape(len(resized), -1)
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        return (flat / np.maximum(norms, 1e-6)).astype(np.float32)

    def probabilities(self, images: np.ndarray, temperature: float = 0.08) -> np.ndarray:
        """Per-class scores for each crop, as a (n, 13) distribution.

        A class's score comes from its single best match rather than an average
        over its examples: the bank holds one figurine per class per book, and
        the question is whether *this* crop is that figurine.
        """
        if len(self) == 0:
            return np.full((len(images), NUM_CLASSES), 1.0 / NUM_CLASSES, np.float32)
        similarity = self._embed(images) @ self.vectors.T      # (n, bank)
        best = np.full((len(images), NUM_CLASSES), -1.0, np.float32)
        for class_index in self.classes:
            mask = self.labels == class_index
            best[:, class_index] = similarity[:, mask].max(axis=1)
        scores = np.exp((best - best.max(axis=1, keepdims=True)) / temperature)
        scores[best < -0.5] = 0.0                              # classes not in the bank
        total = scores.sum(axis=1, keepdims=True)
        return (scores / np.maximum(total, 1e-9)).astype(np.float32)


class Predictor:
    """Reads boards, using a trained net, an exemplar bank, or both."""

    def __init__(self, checkpoint_path: str | Path | None = None, square_size: int = SQUARE_SIZE):
        self.square_size = square_size
        self.net = None
        self.checkpoint = None
        if checkpoint_path is not None:
            from .model import load_net

            self.net, self.checkpoint = load_net(checkpoint_path)
            self.square_size = self.checkpoint.square_size

    @property
    def has_model(self) -> bool:
        return self.net is not None

    def _net_probabilities(self, squares: np.ndarray) -> np.ndarray:
        import torch

        temperature = self.checkpoint.temperature if self.checkpoint else 1.0
        with torch.no_grad():
            x = torch.from_numpy(normalise(squares)).unsqueeze(1)
            logits = self.net(x) / temperature
            return torch.softmax(logits, dim=1).numpy().astype(np.float32)

    def read_squares(self, squares: np.ndarray, bank: ExemplarBank | None = None) -> BoardReading:
        """Classify 64 square crops in reading order."""
        if len(squares) != 64:
            raise ValueError(f"expected 64 squares, got {len(squares)}")
        have_bank = bank is not None and len(bank) > 0
        if not self.has_model and not have_bank:
            raise RuntimeError(
                "nothing to read with: train a model, or verify a diagram in this "
                "book first so the exemplar bank has something in it"
            )
        if not self.has_model:
            return self._reading(bank.probabilities(squares), "exemplars")

        net = self._net_probabilities(squares)
        if not have_bank:
            return self._reading(net, "net")

        # How much the bank may say on a square is its authority times the
        # net's doubt.  Authority is about coverage, not volume: the bank
        # returns zero for a class it has never seen, so what makes it risky is
        # the piece types missing from it, not how few crops it holds.
        #
        # Multiplying by the net's doubt is what makes mixing safe.  Measured
        # over full review runs on books in a figurine style the net had never
        # seen, letting the bank assert itself in proportion to its coverage
        # made readings *worse* -- it overrode a net that was already right.
        # Confined to the squares the net is unsure of, it costs nothing and
        # still settles the cases the net cannot.
        authority = 0.8 * len(bank.classes) / NUM_CLASSES
        weight = (authority * (1.0 - net.max(axis=1)))[:, None].astype(np.float32)
        combined = (1.0 - weight) * net + weight * bank.probabilities(squares)
        return self._reading(combined, "net+exemplars")

    def _reading(self, probabilities: np.ndarray, source: str) -> BoardReading:
        indices = probabilities.argmax(axis=1)
        return BoardReading(
            labels=[index_label(int(i)) for i in indices],
            confidence=[float(probabilities[i, indices[i]]) for i in range(64)],
            probabilities=probabilities,
            source=source,
        )

    def read_board(
        self,
        gray: np.ndarray,
        grid: GridFit,
        bank: ExemplarBank | None = None,
        pad_frac: float = 0.14,
    ) -> tuple[BoardReading, np.ndarray]:
        """Cut a board picture into squares and read it.  Returns the crops too."""
        squares = extract_squares(gray, grid, size=self.square_size, pad_frac=pad_frac)
        return self.read_squares(squares, bank), squares


def bank_for_book(workspace, book_id: int, size: int = EXEMPLAR_SIZE) -> ExemplarBank:
    """Build an exemplar bank from what has been verified in one book."""
    verified = workspace.verified_squares(square_size=SQUARE_SIZE)
    bank = ExemplarBank(size)
    if len(verified):
        mask = verified.book_ids == book_id
        bank.add(verified.images[mask], verified.labels[mask])
    return bank
