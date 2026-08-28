"""Cold-start accuracy per figurine style.

The synthetic benchmark the tool ships with draws its diagrams from three piece
styles, and a model trained on those reads them near-perfectly -- which tells us
almost nothing, because real books are set in figurine fonts that look nothing
like the three.  This measures the case that matters: a book set in a style the
model has never seen, one style at a time, so the failures are attributable.

Point ``--pieces`` at a directory of style subdirectories (see 'dgc pieces').
"""
import argparse, json, sys, tempfile
sys.path.insert(0, "src")
from pathlib import Path

import numpy as np
import pymupdf

from diagramchess.board import BoardMatrix
from diagramchess.demo import build_demo_book
from diagramchess.detect import _iou
from diagramchess.pieces import piece_sets_in
from diagramchess.pipeline import ingest, load_squares
from diagramchess.predict import Predictor, bank_for_book
from diagramchess.store import Workspace
from diagramchess import pdfio


def truth_for_diagrams(workspace, book_id, pdf, meta, dpi=200):
    """Match each stored diagram to the position the book actually printed."""
    doc = pdfio.open_pdf(pdf)
    truth = {}
    for row in workspace.diagrams(book_id=book_id):
        page = int(row["page"])
        render = pdfio.render_page(doc, page, dpi=dpi)
        best, best_iou = None, 0.0
        for entry in meta["diagrams"]:
            if entry["page"] != page:
                continue
            box = render.to_pixels(pymupdf.Rect(*entry["box_pt"]))
            score = _iou((row["x0"], row["y0"], row["x1"], row["y1"]), box)
            if score > best_iou:
                best, best_iou = entry, score
        if best and best_iou > 0.4:
            truth[int(row["id"])] = BoardMatrix.from_fen(best["fen"]).flat()
    doc.close()
    return truth


def run_style(model, piece_set, pages=8, seed=0):
    """Build a book in one style, read it, and score against ground truth."""
    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "book.pdf"
    build_demo_book(pdf, pages=pages, seed=seed, style_seed=seed, piece_set=piece_set)
    meta = json.loads(pdf.with_suffix(".truth.json").read_text())
    workspace = Workspace(tmp / "ws")
    report = ingest(workspace, pdf, predictor=Predictor(model))
    truth = truth_for_diagrams(workspace, report.book_id, pdf, meta)

    squares_ok = squares = perfect = 0
    for diagram_id, actual in truth.items():
        predicted = [row["predicted"] for row in workspace.squares(diagram_id)]
        ok = sum(1 for a, b in zip(predicted, actual) if a == b)
        squares_ok += ok
        squares += 64
        perfect += int(ok == 64)
    workspace.close()
    return {
        "style": piece_set.name,
        "detected": report.detected,
        "expected": len(meta["diagrams"]),
        "matched": len(truth),
        "square_accuracy": squares_ok / squares if squares else 0.0,
        "perfect": perfect,
        "diagrams": len(truth),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--pieces", default="/home/user/lichess-org/lila/public/piece")
    parser.add_argument("--styles", help="comma separated style names; default all")
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sets = {s.name: s for s in piece_sets_in(args.pieces)}
    wanted = args.styles.split(",") if args.styles else sorted(sets)
    print(f"{'style':<22}{'found':>10}{'squares':>10}{'perfect':>10}")
    rows = []
    for name in wanted:
        if name not in sets:
            print(f"{name:<22}  (no such style)")
            continue
        row = run_style(args.model, sets[name], args.pages, args.seed)
        rows.append(row)
        print(f"{row['style']:<22}{row['matched']:>4}/{row['expected']:<5}"
              f"{row['square_accuracy'] * 100:>9.2f}%{row['perfect']:>7}/{row['diagrams']}")
    if rows:
        acc = np.mean([r["square_accuracy"] for r in rows])
        perfect = sum(r["perfect"] for r in rows)
        total = sum(r["diagrams"] for r in rows)
        print(f"\n{len(rows)} styles: mean square accuracy {acc * 100:.2f}%, "
              f"{perfect}/{total} diagrams read perfectly")


if __name__ == "__main__":
    main()
