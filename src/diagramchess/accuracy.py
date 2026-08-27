"""Measuring how well the tool is doing on your own books.

Synthetic validation accuracy says how well the model reads diagrams we drew.
That is not the question.  The question is how often it is right about the book
in your hands, and the only honest answer comes from the diagrams you have
already checked: for each of those we have what the model said before you looked
at it, and what you said afterwards.

This is also the number that decides when to stop checking every diagram.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .labels import LABELS, LABEL_NAMES, LABEL_TO_INDEX, NUM_CLASSES


@dataclass
class AccuracyReport:
    diagrams: int = 0
    perfect_diagrams: int = 0
    squares: int = 0
    correct_squares: int = 0
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64))
    #: (predicted confidence, was it right) for every square, for calibration.
    calibration: list[tuple[float, bool]] = field(default_factory=list)
    per_book: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def square_accuracy(self) -> float:
        return self.correct_squares / self.squares if self.squares else 0.0

    @property
    def diagram_accuracy(self) -> float:
        return self.perfect_diagrams / self.diagrams if self.diagrams else 0.0

    @property
    def corrections_per_diagram(self) -> float:
        return (self.squares - self.correct_squares) / self.diagrams if self.diagrams else 0.0

    def worst_confusions(self, limit: int = 8) -> list[tuple[str, str, int]]:
        """The mistakes it actually makes, worst first."""
        out: list[tuple[str, str, int]] = []
        for actual in range(NUM_CLASSES):
            for predicted in range(NUM_CLASSES):
                if actual != predicted and self.confusion[actual, predicted]:
                    out.append((LABELS[actual], LABELS[predicted], int(self.confusion[actual, predicted])))
        return sorted(out, key=lambda row: -row[2])[:limit]

    def missed_at_confidence(self, threshold: float) -> tuple[int, int]:
        """Squares at or above ``threshold`` and how many of those were wrong.

        This is what a review threshold would cost you: raise the bar and you
        check more squares by hand; lower it and some errors slip through
        unlooked at.
        """
        above = [ok for confidence, ok in self.calibration if confidence >= threshold]
        return len(above), sum(1 for ok in above if not ok)

    def advice(self) -> str:
        """Whether it is safe to stop checking every diagram, and why."""
        if self.diagrams < 5:
            return (f"only {self.diagrams} diagram(s) checked so far -- "
                    "verify a few more before reading anything into these numbers")
        if self.diagram_accuracy >= 0.95:
            return ("the model now reads whole diagrams correctly "
                    f"{self.diagram_accuracy * 100:.0f}% of the time; you could switch to "
                    "spot-checking the low-confidence ones only")
        if self.corrections_per_diagram <= 1.0:
            return (f"about {self.corrections_per_diagram:.1f} correction(s) per diagram; "
                    "retrain now -- the corrections you have will pay for themselves")
        return (f"about {self.corrections_per_diagram:.1f} correction(s) per diagram; "
                "keep verifying, and retrain once you have twenty or so diagrams done")

    def describe(self) -> str:
        if not self.diagrams:
            return "nothing verified yet, so there is nothing to measure"
        lines = [
            f"measured against {self.diagrams} diagram(s) you verified "
            f"({self.squares} squares)",
            f"  squares read correctly:  {self.square_accuracy * 100:6.2f}%",
            f"  diagrams read perfectly: {self.diagram_accuracy * 100:6.2f}%",
            f"  corrections per diagram: {self.corrections_per_diagram:6.2f}",
        ]
        for threshold in (0.99, 0.95, 0.90):
            count, wrong = self.missed_at_confidence(threshold)
            if count:
                lines.append(f"  above {threshold * 100:.0f}% confidence: {count} squares, "
                             f"{wrong} of them wrong ({wrong / count * 100:.2f}%)")
        confusions = self.worst_confusions()
        if confusions:
            lines.append("  mistakes it makes most:")
            for actual, predicted, count in confusions:
                lines.append(f"      {LABEL_NAMES[actual]} read as {LABEL_NAMES[predicted]}: {count}")
        lines.append(f"  -> {self.advice()}")
        return "\n".join(lines)


def measure(workspace, book_id: int | None = None) -> AccuracyReport:
    """Compare what the model said against what you said, over verified diagrams.

    Only diagrams that carry a stored prediction are counted; one you filled in
    by hand before any model existed has nothing to be measured against.
    """
    report = AccuracyReport()
    clauses = ["d.status = 'verified'"]
    params: list = []
    if book_id is not None:
        clauses.append("d.book_id = ?")
        params.append(book_id)
    rows = workspace.query(f"""
        SELECT s.diagram_id, d.book_id, s.predicted, s.label, s.confidence
        FROM squares s JOIN diagrams d ON d.id = s.diagram_id
        WHERE {' AND '.join(clauses)} AND s.label IS NOT NULL AND s.predicted IS NOT NULL
        ORDER BY s.diagram_id, s.row, s.col
    """, tuple(params))

    by_diagram: dict[int, list] = {}
    for row in rows:
        by_diagram.setdefault(int(row["diagram_id"]), []).append(row)

    for diagram_id, squares in by_diagram.items():
        if len(squares) != 64:
            continue  # a half-recorded diagram would skew the per-diagram rate
        report.diagrams += 1
        book = int(squares[0]["book_id"])
        wrong = 0
        for square in squares:
            predicted, actual = square["predicted"], square["label"]
            report.squares += 1
            correct = predicted == actual
            report.correct_squares += int(correct)
            wrong += int(not correct)
            report.confusion[LABEL_TO_INDEX[actual], LABEL_TO_INDEX[predicted]] += 1
            if square["confidence"] is not None:
                report.calibration.append((float(square["confidence"]), correct))
        report.perfect_diagrams += int(wrong == 0)
        done, total = report.per_book.get(book, (0, 0))
        report.per_book[book] = (done + int(wrong == 0), total + 1)

    return report
