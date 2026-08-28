"""Sweep the detection threshold against generated books to choose it from data."""
import json, sys, tempfile
sys.path.insert(0, "src")
from pathlib import Path
import pymupdf
from diagramchess import pdfio
from diagramchess.detect import detect_boards, _iou
from diagramchess.demo import build_demo_book

def main(books=8, pages=6, seed0=100):
    tmp = Path(tempfile.mkdtemp())
    scored = []   # (score, is_true)
    total_gt = 0
    for b in range(books):
        pdf = tmp / f"book{b}.pdf"
        build_demo_book(pdf, pages=pages, seed=seed0 + b, style_seed=seed0 + 100 + b)
        meta = json.loads(pdf.with_suffix(".truth.json").read_text())
        doc = pdfio.open_pdf(pdf)
        for pi in range(len(doc)):
            r = pdfio.render_page(doc, pi, dpi=200)
            props = pdfio.embedded_image_boxes(doc, pi, r) + pdfio.vector_drawing_boxes(doc, pi, r)
            gts = [r.to_pixels(pymupdf.Rect(*d["box_pt"])) for d in meta["diagrams"] if d["page"] == pi]
            total_gt += len(gts)
            dets = detect_boards(r.image, props, threshold=0.0)
            matched = set()
            for d in sorted(dets, key=lambda x: -x.score):
                hit = next((i for i, g in enumerate(gts) if i not in matched and _iou(d.box, g) > 0.55), None)
                if hit is None:
                    scored.append((d.score, False))
                else:
                    matched.add(hit); scored.append((d.score, True))
    print(f"{'thresh':>7} {'recall':>7} {'prec':>7} {'tp':>4} {'fp':>4} {'fn':>4}")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        tp = sum(1 for s, ok in scored if s >= t and ok)
        fp = sum(1 for s, ok in scored if s >= t and not ok)
        fn = total_gt - tp
        print(f"{t:7.2f} {tp/total_gt:7.3f} {tp/max(1,tp+fp):7.3f} {tp:4d} {fp:4d} {fn:4d}")
    misses = sorted(s for s, ok in scored if ok)[:8]
    print("lowest true-board scores:", [round(m, 3) for m in misses])
    fps = sorted((s for s, ok in scored if not ok), reverse=True)[:8]
    print("highest false-positive scores:", [round(m, 3) for m in fps])

if __name__ == "__main__":
    import sys
    main(books=int(sys.argv[1]) if len(sys.argv) > 1 else 8,
         seed0=int(sys.argv[2]) if len(sys.argv) > 2 else 100)
