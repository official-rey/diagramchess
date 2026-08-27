"""Detection on scanned books: no text layer, the whole page is one image.

Many chess books only exist as scans, where none of the PDF-level proposals
help -- the single image on the page is the page -- so everything rests on the
image-based detector.
"""
import json, sys, tempfile
sys.path.insert(0, "src")
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from diagramchess import pdfio
from diagramchess.demo import build_demo_book
from diagramchess.detect import _iou, detect_boards


def scan(source: Path, target: Path, dpi: int = 150) -> list[tuple[int, tuple]]:
    """Re-render a PDF as image-only pages, the way a flatbed would."""
    doc = pdfio.open_pdf(source)
    out = pymupdf.open()
    boxes = []
    for page_index in range(len(doc)):
        render = pdfio.render_page(doc, page_index, dpi=dpi)
        image = render.image
        # scanner noise and a slight skew
        rng = np.random.default_rng(page_index)
        matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2),
                                         rng.uniform(-0.6, 0.6), 1.0)
        image = cv2.warpAffine(image, matrix, image.shape[::-1], borderValue=255)
        image = np.clip(image.astype(np.float32) + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        page = out.new_page(width=doc[page_index].rect.width, height=doc[page_index].rect.height)
        page.insert_image(page.rect, stream=buf.tobytes())
        boxes.append((page_index, page.rect))
    doc.close()
    out.save(str(target))
    out.close()
    return boxes


def main(books=4):
    tmp = Path(tempfile.mkdtemp())
    tp = fp = fn = 0
    for b in range(books):
        clean = tmp / f"clean{b}.pdf"
        build_demo_book(clean, pages=6, seed=500 + b, style_seed=200 + b)
        meta = json.loads(clean.with_suffix(".truth.json").read_text())
        scanned = tmp / f"scan{b}.pdf"
        scan(clean, scanned)

        doc = pdfio.open_pdf(scanned)
        assert not any(doc[i].get_text().strip() for i in range(len(doc))), "still has text"
        found = 0
        for page_index in range(len(doc)):
            render = pdfio.render_page(doc, page_index, dpi=200)
            proposals = pdfio.embedded_image_boxes(doc, page_index, render)
            proposals += pdfio.vector_drawing_boxes(doc, page_index, render)
            detections = detect_boards(render.image, proposals)
            truth = [render.to_pixels(pymupdf.Rect(*d["box_pt"]))
                     for d in meta["diagrams"] if d["page"] == page_index]
            matched = set()
            for d in detections:
                hit = next((i for i, g in enumerate(truth)
                            if i not in matched and _iou(d.box, g) > 0.5), None)
                if hit is None:
                    fp += 1
                else:
                    matched.add(hit); tp += 1; found += 1
            fn += len(truth) - len(matched)
        doc.close()
        print(f"scan{b}: found {found}/{len([d for d in meta['diagrams']])}")
    print(f"TOTAL tp={tp} fp={fp} fn={fn} recall={tp/max(1,tp+fn):.3f} precision={tp/max(1,tp+fp):.3f}")


if __name__ == "__main__":
    main()
