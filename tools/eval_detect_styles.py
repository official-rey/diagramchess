"""Does detection hold up across figurine styles, or only the ones we drew with?

Detection is meant to be the part that does not care what the pieces look like:
it fits a lattice and checks that the cells hold big centred glyphs.  That is a
claim, and this measures it, one style at a time.
"""
import argparse, json, sys, tempfile
sys.path.insert(0, "src")
from pathlib import Path

import numpy as np
import pymupdf

from diagramchess import pdfio
from diagramchess.demo import build_demo_book
from diagramchess.detect import _iou, detect_boards
from diagramchess.pieces import piece_sets_in


def run_style(piece_set, pages, seed):
    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "book.pdf"
    build_demo_book(pdf, pages=pages, seed=seed, style_seed=seed, piece_set=piece_set)
    meta = json.loads(pdf.with_suffix(".truth.json").read_text())
    doc = pdfio.open_pdf(pdf)
    found = missed = spurious = 0
    for page_index in range(len(doc)):
        render = pdfio.render_page(doc, page_index, dpi=200)
        proposals = pdfio.embedded_image_boxes(doc, page_index, render)
        proposals += pdfio.vector_drawing_boxes(doc, page_index, render)
        detections = detect_boards(render.image, proposals)
        truth = [render.to_pixels(pymupdf.Rect(*d["box_pt"]))
                 for d in meta["diagrams"] if d["page"] == page_index]
        matched = set()
        for detection in detections:
            hit = next((i for i, box in enumerate(truth)
                        if i not in matched and _iou(detection.box, box) > 0.55), None)
            if hit is None:
                spurious += 1
            else:
                matched.add(hit)
                found += 1
        missed += len(truth) - len(matched)
    doc.close()
    return found, missed, spurious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pieces", default="/home/user/lichess-org/lila/public/piece")
    parser.add_argument("--styles", help="comma separated; default every style found")
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    sets = {s.name: s for s in piece_sets_in(args.pieces)}
    names = args.styles.split(",") if args.styles else sorted(sets)
    total_found = total_missed = total_spurious = 0
    worst = []
    for name in names:
        if name not in sets:
            continue
        found = missed = spurious = 0
        for seed in range(args.seeds):
            f, m, s = run_style(sets[name], args.pages, 700 + seed)
            found, missed, spurious = found + f, missed + m, spurious + s
        recall = found / max(1, found + missed)
        total_found += found
        total_missed += missed
        total_spurious += spurious
        worst.append((recall, name, found, missed, spurious))
        print(f"{name:<22} recall {recall * 100:6.1f}%  found {found:>3} missed {missed:>3} spurious {spurious:>3}")

    recall = total_found / max(1, total_found + total_missed)
    precision = total_found / max(1, total_found + total_spurious)
    print(f"\n{len(worst)} styles: recall {recall * 100:.1f}%  precision {precision * 100:.1f}%")
    print("weakest styles:")
    for r, name, f, m, s in sorted(worst)[:5]:
        print(f"  {name:<20} {r * 100:5.1f}%  ({m} missed of {f + m})")


if __name__ == "__main__":
    main()
