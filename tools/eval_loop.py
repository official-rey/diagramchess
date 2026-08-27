"""Does verifying diagrams actually make the tool better on the next one?

This simulates the review loop against a generated book set in a figurine style
the model was never trained on -- the situation the tool is really for.  A
simulated reviewer verifies diagrams in the order the tool asks for them, and
after each one every remaining diagram is read again.  What we want to see is
the number of corrections per diagram falling as exemplars accumulate.
"""
import json, sys, tempfile
sys.path.insert(0, "src")
from pathlib import Path

import numpy as np
import pymupdf

from diagramchess.board import BoardMatrix
from diagramchess.demo import build_demo_book
from diagramchess.detect import _iou
from diagramchess.pipeline import ingest, load_squares
from diagramchess.predict import Predictor, bank_for_book
from diagramchess.store import Workspace
from diagramchess import pdfio


def truth_for_diagrams(workspace, book_id, pdf, meta):
    """Match each stored diagram to the position the book actually printed."""
    doc = pdfio.open_pdf(pdf)
    truth = {}
    for row in workspace.diagrams(book_id=book_id):
        page = int(row["page"])
        render = pdfio.render_page(doc, page, dpi=200)
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


def run(model_path, seed=900, pages=14, style_seed=None):
    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "book.pdf"
    build_demo_book(pdf, pages=pages, seed=seed, style_seed=style_seed)
    meta = json.loads(pdf.with_suffix(".truth.json").read_text())
    workspace = Workspace(tmp / "ws")
    predictor = Predictor(model_path)
    report = ingest(workspace, pdf, predictor=predictor)
    book_id = report.book_id
    truth = truth_for_diagrams(workspace, book_id, pdf, meta)
    print(f"style: pieces={meta['piece_set']} checkered={meta['style']['checkered']} "
          f"lines={meta['style']['grid_line']} coords={meta['style']['coordinates']}")
    print(f"{len(truth)} diagrams matched to ground truth\n")

    def read_all(bank):
        """Corrections each unverified diagram would need right now."""
        out = {}
        for did, actual in truth.items():
            row = workspace.diagram(did)
            if row["status"] == "verified":
                continue
            squares = load_squares(workspace, did)
            reading = predictor.read_squares(squares, bank)
            wrong = sum(1 for a, b in zip(reading.labels, actual) if a != b)
            out[did] = (wrong, reading.min_confidence)
        return out

    print(f"{'verified':>8} {'remaining':>9} {'mean errors':>11} {'perfect':>7} {'worst':>5}")
    rows = []
    for step in range(len(truth) + 1):
        bank = bank_for_book(workspace, book_id)
        pending = read_all(bank)
        if not pending:
            break
        errors = [w for w, _ in pending.values()]
        perfect = sum(1 for e in errors if e == 0)
        print(f"{step:>8} {len(pending):>9} {np.mean(errors):>11.2f} "
              f"{perfect}/{len(errors):>5} {max(errors):>5}")
        rows.append((step, float(np.mean(errors)), perfect / len(errors)))

        # The reviewer takes whichever diagram the tool is least sure about.
        target = min(pending, key=lambda d: pending[d][1])
        labels = truth[target]
        board = BoardMatrix.from_labels(labels)
        workspace.save_review(target, labels, board.orientation, board.side_to_move, board.to_fen())
    return rows


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "models/piece-net-holdout.pt"
    style = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(model, style_seed=style)
