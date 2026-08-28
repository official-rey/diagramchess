"""How far can a real book degrade before the tool stops reading it?

A clean PDF scoring perfectly says little on its own: plenty of chess books only
exist as photocopies, and plenty of readers photograph a page rather than scan
it.  This takes the same diagrams and the same answer key as ``real_book.py``
and puts the page through what actually happens to one -- a coarser scan, a soft
lens, sensor noise, JPEG, and a sheet that was not square on the glass.

The ground-truth boxes go through the same geometric transform as the page, so
they stay correct however it is warped.

    python tools/eval_stress.py your-book.pdf
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import pymupdf

from diagramchess import pdfio
from diagramchess.detect import _iou, detect_boards
from diagramchess.grid import extract_squares
from diagramchess.model import bundled_model
from diagramchess.predict import Predictor
from real_book import diagrams_on_page


@dataclass
class Condition:
    """One way a page can arrive in worse shape than the publisher sent it."""

    name: str
    dpi: int = 200
    blur: float = 0.0
    noise: float = 0.0
    jpeg: int = 0          # 0 means no JPEG step
    skew: float = 0.0      # degrees


CONDITIONS = [
    Condition("clean, 200 dpi"),
    Condition("150 dpi", dpi=150),
    Condition("120 dpi", dpi=120),
    Condition("100 dpi", dpi=100),
    Condition("photocopy", dpi=150, blur=0.8, noise=4, jpeg=70, skew=0.4),
    Condition("poor scan", dpi=120, blur=1.2, noise=8, jpeg=45, skew=0.8),
    Condition("phone photo", dpi=110, blur=1.6, noise=12, jpeg=35, skew=1.5),
]


def degrade(image, boxes, condition: Condition, seed: int):
    """Apply the condition to the page, and to the ground-truth boxes with it."""
    rng = np.random.default_rng(seed)
    out = image.astype(np.float32)
    corners = [np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32)
               for b in boxes]

    if condition.skew:
        h, w = out.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), condition.skew, 1.0)
        out = cv2.warpAffine(out, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        corners = [cv2.transform(c[None], matrix)[0] for c in corners]

    if condition.blur:
        out = cv2.GaussianBlur(out, (0, 0), condition.blur)
    if condition.noise:
        out = out + rng.normal(0, condition.noise, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if condition.jpeg:
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), condition.jpeg])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    moved = [(float(c[:, 0].min()), float(c[:, 1].min()),
              float(c[:, 0].max()), float(c[:, 1].max())) for c in corners]
    return out, moved


def run(book: str, condition: Condition, predictor: Predictor) -> dict:
    doc = pdfio.open_pdf(book)
    found = missed = spurious = 0
    ok = total = perfect = read = 0
    for index in range(len(doc)):
        render = pdfio.render_page(doc, index, dpi=condition.dpi)
        truth = diagrams_on_page(doc[index])
        boxes = [render.to_pixels(pymupdf.Rect(*b)) for _, b in truth]
        image, boxes = degrade(render.image, boxes, condition, seed=index)

        matched: dict[int, object] = {}
        for detection in detect_boards(image, []):
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
            labels = predictor.read_squares(extract_squares(image, detection.grid, size=48)).labels
            right = sum(1 for a, b in zip(labels, actual) if a == b)
            ok += right
            total += 64
            perfect += int(right == 64)
            read += 1
    doc.close()
    return dict(found=found, missed=missed, spurious=spurious, ok=ok,
                total=total, perfect=perfect, read=read)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("book")
    parser.add_argument("--model")
    args = parser.parse_args()

    predictor = Predictor(args.model or bundled_model())
    print(f"{'condition':<18}{'recall':>9}{'spurious':>10}{'squares':>10}"
          f"{'perfect':>12}{'corr/diag':>11}")
    for condition in CONDITIONS:
        r = run(args.book, condition, predictor)
        if not r["read"]:
            print(f"{condition.name:<18}  nothing read")
            continue
        recall = r["found"] / max(1, r["found"] + r["missed"])
        squares = r["ok"] / max(1, r["total"])
        corrections = (r["total"] - r["ok"]) / max(1, r["read"])
        print(f"{condition.name:<18}{recall * 100:8.1f}%{r['spurious']:>10}"
              f"{squares * 100:9.2f}%{r['perfect']:>7}/{r['read']:<4}{corrections:>11.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
