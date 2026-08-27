"""End-to-end detection accuracy on generated demo books."""
import json, sys, tempfile
sys.path.insert(0, "src")
from pathlib import Path
from diagramchess import pdfio
from diagramchess.detect import detect_boards
from diagramchess.demo import build_demo_book

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0: return 0.0
    inter = (ix1-ix0)*(iy1-iy0)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def main(books=6, pages=4):
    tmp = Path(tempfile.mkdtemp())
    tp = fp = fn = 0
    for b in range(books):
        pdf = tmp / f"book{b}.pdf"
        truth = build_demo_book(pdf, pages=pages, seed=100+b, style_seed=200+b)
        doc = pdfio.open_pdf(pdf)
        meta = json.loads(pdf.with_suffix(".truth.json").read_text())
        found_total = 0
        for page_index in range(len(doc)):
            render = pdfio.render_page(doc, page_index, dpi=200)
            proposals = pdfio.embedded_image_boxes(doc, page_index, render)
            proposals += pdfio.vector_drawing_boxes(doc, page_index, render)
            dets = detect_boards(render.image, proposals)
            gts = [render.to_pixels(__import__("pymupdf").Rect(*d["box_pt"])) for d in meta["diagrams"] if d["page"] == page_index]
            matched = set()
            for d in dets:
                hit = None
                for i, g in enumerate(gts):
                    if i not in matched and iou(d.box, g) > 0.55:
                        hit = i; break
                if hit is None: fp += 1
                else: matched.add(hit); tp += 1; found_total += 1
            fn += len(gts) - len(matched)
        print(f"book{b} pieces={meta['piece_set']:10s} coords={meta['style']['coordinates']} "
              f"checkered={meta['style']['checkered']} lines={meta['style']['grid_line']} border={meta['style']['border_width']} -> found {found_total}/{len(truth)}")
    print(f"TOTAL tp={tp} fp={fp} fn={fn}  recall={tp/(tp+fn):.3f} precision={tp/max(1,tp+fp):.3f}")

if __name__ == "__main__":
    main()
