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

        exemplar = bank.probabilities(squares)
        weight = self._bank_weight(net, exemplar, bank)[:, None]
        combined = (1.0 - weight) * net + weight * exemplar
        return self._reading(combined, "net+exemplars")

    def _bank_weight(self, net: np.ndarray, exemplar: np.ndarray, bank: "ExemplarBank") -> np.ndarray:
        """How much say the exemplar bank gets on each square, in 0..1.

        Two things set it.

        *Coverage*, because the bank returns zero for a class it has never seen:
        what makes it risky is the piece types missing from it, not how few
        crops it holds of the ones it has.  One verified middlegame covers
        nearly the whole label set; one verified king-and-pawn ending covers
        four classes.

        *Whether the model looks lost on this book*, because the right answer
        genuinely depends on that and nothing else here knows it.  When the two
        readers disagree about a large part of the board, the model is probably
        reading a figurine style it was never trained on; when they mostly
        agree, it is fine and the bank should only break ties.  Disagreement
        cannot say which reader is wrong, so it is a guess -- but it is a guess
        made from the two readings we already have, needing no labels.

        This was fitted twice, and the first fit was wrong both times it was
        checked.  Measured as mean errors per diagram over books set in unseen
        figurine fonts, with banks of one to twelve verified diagrams:

            weighting            weak model    shipped model
            model alone               11.34             0.94
            by the model's doubt      10.91             0.65
            by coverage alone          6.43             2.63
            by coverage x doubt        8.43             0.65
            this rule                  6.71             1.20

        Nothing wins both columns.  Weighting by doubt alone is best against the
        model that actually ships and four errors a diagram worse when the model
        is out of its depth; weighting by coverage alone is the reverse.  This
        rule is chosen on the worst case rather than the average, because the
        hard column is the whole reason the exemplar bank exists -- and because
        the cost of being second-best in the easy column is half a correction
        per diagram, on diagrams that barely need reviewing.
        """
        # One verified diagram is not evidence.  Measured on unseen fonts, a
        # bank built from a single diagram makes readings *worse* than the model
        # alone -- it has one crop per class, no sense of how much a piece can
        # vary within the book, and enough coverage to sound confident.  From
        # the second diagram on it is an improvement at every bank size.
        if len(bank) < 2 * 64:
            return np.zeros(len(net), np.float32)

        coverage = len(bank.classes) / NUM_CLASSES
        disagreement = float((net.argmax(axis=1) != exemplar.argmax(axis=1)).mean())
        # Six per cent of a board is ordinary noise between two readers; a
        # quarter of it means one of them is not reading this book at all.
        lost = float(np.clip((disagreement - 0.06) / 0.16, 0.0, 1.0))
        doubt = 1.0 - net.max(axis=1)
        return (0.8 * coverage * np.maximum(lost, doubt)).astype(np.float32)

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
