"""Score the tool against a real chess book, using the book as its own answer key.

Many chess books typeset their diagrams in a chess font rather than pasting in
pictures, and Chess Merida is the common one.  When they do, the PDF's text
layer holds the exact position of every diagram *and* exactly where each sits on
the page -- so a book like that can grade the vision pipeline on real pages, at
the size a real book prints them, with no hand-labelling at all.

That is worth more than any amount of generated material: the two most damaging
bugs in this project were both found this way and neither was visible in the
generated benchmarks.

    python tools/real_book.py your-book.pdf

Merida encodes one square as one character: the letter picks the piece and its
case picks the square's colour.  If a book uses a different diagram font this
will decode nothing, and say so rather than reporting a false score.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pymupdf

from diagramchess import pdfio
from diagramchess.board import BoardMatrix
from diagramchess.detect import _iou, detect_boards
from diagramchess.grid import extract_squares
from diagramchess.labels import LABEL_NAMES
from diagramchess.model import bundled_model
from diagramchess.predict import Predictor

DPI = 200

#: Merida's square alphabet.  Lower case sits on a light square, upper on a
#: dark one; the piece is the same either way.
MERIDA = {
    " ": ".", "+": ".",
    "p": "P", "n": "N", "b": "B", "r": "R", "q": "Q", "k": "K",
    "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
    "o": "p", "m": "n", "v": "b", "t": "r", "w": "q", "l": "k",
    "O": "p", "M": "n", "V": "b", "T": "r", "W": "q", "L": "k",
}
#: The rank markers that open a board row, rank 8 down to rank 1.
RANK_MARKERS = ""


def decode_row(line: str) -> list[str] | None:
    """One board row of eight squares, or None if this line is not one."""
    if not line or line[0] not in RANK_MARKERS:
        return None
    squares: list[str] = []
    for ch in line[1:]:
        plain = chr(ord(ch) - 0xF000) if 0xF000 <= ord(ch) <= 0xF0FF else ch
        if plain in MERIDA:
            squares.append(MERIDA[plain])
        if len(squares) == 8:
            return squares
    return None


def diagrams_on_page(page) -> list[tuple[BoardMatrix, tuple[float, float, float, float]]]:
    """Every diagram the page's text layer describes, with its box in points."""
    found, rows, boxes = [], [], []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            row = decode_row(text)
            if row is None:
                rows, boxes = [], []
                continue
            rows.append(row)
            boxes.append(line["bbox"])
            if len(rows) == 8:
                box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                       max(b[2] for b in boxes), max(b[3] for b in boxes))
                found.append((BoardMatrix([list(r) for r in rows]), box))
                rows, boxes = [], []
    return found


def measure(path: str, model: str | None = None, dpi: int = DPI) -> dict:
    doc = pdfio.open_pdf(path)
    predictor = Predictor(model or bundled_model())

    found = missed = spurious = 0
    ok = total = perfect = read = 0
    illegal = 0
    confusions: Counter = Counter()

    for index in range(len(doc)):
        render = pdfio.render_page(doc, index, dpi=dpi)
        truth = diagrams_on_page(doc[index])
        illegal += sum(1 for board, _ in truth if board.problems())
        boxes = [render.to_pixels(pymupdf.Rect(*box)) for _, box in truth]

        proposals = pdfio.embedded_image_boxes(doc, index, render)
        proposals += pdfio.vector_drawing_boxes(doc, index, render)
        detections = detect_boards(render.image, proposals)

        matched: dict[int, object] = {}
        for detection in detections:
            best, best_iou = None, 0.0
            for i, box in enumerate(boxes):
                if i in matched:
                    continue
                score = _iou(detection.box, box)
                if score > best_iou:
                    best, best_iou = i, score
            if best is None or best_iou < 0.4:
                spurious += 1
            else:
                matched[best] = detection
        found += len(matched)
        missed += len(truth) - len(matched)

        for i, detection in matched.items():
            actual = truth[i][0].flat()
            labels = predictor.read_squares(
                extract_squares(render.image, detection.grid, size=48)
            ).labels
            right = sum(1 for a, b in zip(labels, actual) if a == b)
            ok += right
            total += 64
            perfect += int(right == 64)
            read += 1
            for a, b in zip(labels, actual):
                if a != b:
                    confusions[(b, a)] += 1
    doc.close()
    return dict(diagrams=found + missed, found=found, missed=missed, spurious=spurious,
                ok=ok, total=total, perfect=perfect, read=read, illegal=illegal,
                confusions=confusions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("book")
    parser.add_argument("--model")
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    result = measure(args.book, args.model, args.dpi)
    if not result["diagrams"]:
        print("No diagrams could be decoded from this PDF's text layer.\n"
              "That means it is not typeset in Chess Merida -- it may paste its\n"
              "diagrams in as pictures, or use a different diagram font -- so this\n"
              "harness has no answer key for it and cannot score anything.")
        return 1

    # A wrong decode would show up here long before it showed up as a score.
    print(f"answer key: {result['diagrams']} diagrams decoded from the text layer, "
          f"{result['diagrams'] - result['illegal']} of them legal positions")
    print(f"\nDETECTION  found {result['found']}  missed {result['missed']}  "
          f"spurious {result['spurious']}")
    print(f"           recall {result['found'] / max(1, result['diagrams']) * 100:.1f}%  "
          f"precision {result['found'] / max(1, result['found'] + result['spurious']) * 100:.1f}%")
    print(f"\nREADING    {result['ok']}/{result['total']} squares "
          f"({result['ok'] / max(1, result['total']) * 100:.2f}%)")
    print(f"           {result['perfect']}/{result['read']} diagrams read perfectly")
    print(f"           {(result['total'] - result['ok']) / max(1, result['read']):.2f} "
          f"corrections per diagram")
    if result["confusions"]:
        print("\n           mistakes it makes most:")
        for (actual, got), count in result["confusions"].most_common(8):
            print(f"             {LABEL_NAMES[actual]:>14} read as "
                  f"{LABEL_NAMES[got]:<14} {count:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
