"""The 13-class label set used everywhere: 12 pieces plus the empty square.

Labels are the single vocabulary shared by the classifier, the store, the
review UI and the FEN writer, so they are defined once here.  The string form
is the FEN character ('K', 'q', ...) with '.' for an empty square; the integer
form is the class index the neural net sees.
"""

from __future__ import annotations

EMPTY = "."

#: Class index -> label character.  Index 0 is empty so that an all-zeros
#: prediction is an empty board rather than a board full of kings.
LABELS: tuple[str, ...] = (
    EMPTY,
    "P", "N", "B", "R", "Q", "K",
    "p", "n", "b", "r", "q", "k",
)

NUM_CLASSES = len(LABELS)

#: Label character -> class index.
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(LABELS)}

#: Human readable names, used by the review UI and by error reports.
LABEL_NAMES: dict[str, str] = {
    EMPTY: "empty",
    "P": "white pawn", "N": "white knight", "B": "white bishop",
    "R": "white rook", "Q": "white queen", "K": "white king",
    "p": "black pawn", "n": "black knight", "b": "black bishop",
    "r": "black rook", "q": "black queen", "k": "black king",
}

#: Unicode glyphs for compact display in the terminal and in the browser.
LABEL_GLYPHS: dict[str, str] = {
    EMPTY: "·",
    "K": "♔", "Q": "♕", "R": "♖",
    "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜",
    "b": "♝", "n": "♞", "p": "♟",
}


def label_index(label: str) -> int:
    """Class index for a label character, raising on anything unknown."""
    try:
        return LABEL_TO_INDEX[label]
    except KeyError:
        raise ValueError(f"not a piece label: {label!r}") from None


def index_label(index: int) -> str:
    """Label character for a class index."""
    if not 0 <= index < NUM_CLASSES:
        raise ValueError(f"class index out of range: {index}")
    return LABELS[index]


def is_white(label: str) -> bool:
    return label != EMPTY and label.isupper()


def is_black(label: str) -> bool:
    return label != EMPTY and label.islower()
