import numpy as np
import pytest
import pymupdf

from diagramchess import pdfio
from diagramchess.board import BoardMatrix
from diagramchess.detect import (
    MIN_BOARD_PX, Detection, _containment, _iou, content_score,
    crop_board, detect_boards, verify_proposal,
)
from diagramchess.pieces import available_piece_sets
from diagramchess.render import DiagramStyle, render_diagram

POSITION = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"


def _page_with_diagram(piece_set, margin=90):
    """A rendered diagram set into a larger sheet of blank paper."""
    rendered = render_diagram(BoardMatrix.from_fen(POSITION),
                              DiagramStyle(piece_set=piece_set, cell_px=40))
    h, w = rendered.image.shape
    page = np.full((h + 2 * margin, w + 2 * margin), 255, np.uint8)
    page[margin:margin + h, margin:margin + w] = rendered.image
    return page, rendered.grid.translated(margin, margin)


@pytest.fixture(params=[s.name for s in available_piece_sets()])
def piece_set(request):
    return next(s for s in available_piece_sets() if s.name == request.param)


def test_finds_a_diagram_on_a_page(piece_set):
    page, truth = _page_with_diagram(piece_set)
    detections = detect_boards(page)
    assert len(detections) == 1
    found = detections[0]
    assert abs(found.grid.x0 - truth.x0) < 0.15 * truth.step_x
    assert abs(found.grid.step_x - truth.step_x) < 0.03 * truth.step_x
    assert found.score > 0.5


def test_reports_nothing_on_blank_paper():
    assert detect_boards(np.full((900, 700), 250, np.uint8)) == []


def test_a_crosstable_is_not_a_diagram():
    """The board detector's own description of a chessboard also fits a crosstable."""
    page = np.full((700, 700), 252, np.uint8)
    cell = 60
    for k in range(10):
        page[100 + k * cell, 100:100 + 9 * cell] = 40
        page[100:100 + 9 * cell, 100 + k * cell] = 40
    for r in range(9):
        for c in range(9):     # a small mark in every cell, as a crosstable has
            y, x = 100 + r * cell + 26, 100 + c * cell + 26
            page[y:y + 8, x:x + 5] = 30
    detections = detect_boards(page)
    assert detections == [], [d.score for d in detections]


def test_content_score_separates_a_board_from_a_table(piece_set):
    page, truth = _page_with_diagram(piece_set)
    score, meta = content_score(page, truth)
    assert score > 0.8
    assert 20 <= meta["occupied_cells"] <= 32
    assert meta["mean_ink"] > 0.15


def test_content_score_rejects_an_empty_lattice(piece_set):
    """A board with nothing on it carries no position, so it is not worth reporting."""
    rendered = render_diagram(BoardMatrix.empty(), DiagramStyle(piece_set=piece_set, cell_px=40))
    score, meta = content_score(rendered.image, rendered.grid)
    assert meta["occupied_cells"] < 2
    assert score == 0.0


def test_a_patch_of_a_board_is_not_a_second_diagram(piece_set):
    """Containment suppression: a box inside a better one is dropped."""
    page, truth = _page_with_diagram(piece_set)
    detections = detect_boards(page)
    boxes = [d.box for d in detections]
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i != j:
                assert _containment(a, b) <= 0.7


def test_verify_rejects_a_proposal_that_is_too_small():
    tiny = (0.0, 0.0, MIN_BOARD_PX - 10.0, MIN_BOARD_PX - 10.0)
    assert verify_proposal(np.full((400, 400), 250, np.uint8), tiny) is None


def test_crop_board_rebases_the_grid(piece_set):
    page, truth = _page_with_diagram(piece_set)
    detection = detect_boards(page)[0]
    crop, grid = crop_board(page, detection)
    assert grid.x0 < detection.grid.x0
    assert 0 <= grid.x0 < crop.shape[1]
    assert abs(grid.step_x - detection.grid.step_x) < 1e-9


def test_iou_and_containment_maths():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert _containment((2, 2, 4, 4), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert _containment((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(0.04)


def test_detects_every_diagram_in_a_generated_book(demo_pdf):
    """End to end on a PDF, including the crosstable pages that must stay empty."""
    import json

    meta = json.loads(demo_pdf.with_suffix(".truth.json").read_text())
    doc = pdfio.open_pdf(demo_pdf)
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
    assert spurious == 0
    assert found >= 0.9 * (found + missed), f"found {found}, missed {missed}"


def test_finds_diagrams_in_a_scan_with_no_text_layer(tmp_path):
    """Plenty of chess books only exist as scans.

    There the PDF-level proposals are no help -- the single image on the page
    *is* the page -- so everything rests on the image-based detector, and the
    board arrives skewed, speckled and JPEG-compressed.
    """
    import cv2
    import numpy as np

    from diagramchess.demo import build_demo_book

    clean = tmp_path / "clean.pdf"
    build_demo_book(clean, pages=9, seed=500, style_seed=200)
    meta = __import__("json").loads(clean.with_suffix(".truth.json").read_text())

    source = pdfio.open_pdf(clean)
    scanned = pymupdf.open()
    for page_index in range(len(source)):
        render = pdfio.render_page(source, page_index, dpi=150)
        image = render.image
        rng = np.random.default_rng(page_index)
        matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2),
                                         rng.uniform(-0.6, 0.6), 1.0)
        image = cv2.warpAffine(image, matrix, image.shape[::-1], borderValue=255)
        image = np.clip(image.astype(np.float32) + rng.normal(0, 4, image.shape),
                        0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        assert ok
        page = scanned.new_page(width=source[page_index].rect.width,
                                height=source[page_index].rect.height)
        page.insert_image(page.rect, stream=buf.tobytes())
    source.close()
    out = tmp_path / "scan.pdf"
    scanned.save(str(out))
    scanned.close()

    doc = pdfio.open_pdf(out)
    assert not any(doc[i].get_text().strip() for i in range(len(doc))), "the scan still has text"
    found = missed = spurious = 0
    for page_index in range(len(doc)):
        render = pdfio.render_page(doc, page_index, dpi=200)
        proposals = pdfio.embedded_image_boxes(doc, page_index, render)
        detections = detect_boards(render.image, proposals)
        truth = [render.to_pixels(pymupdf.Rect(*d["box_pt"]))
                 for d in meta["diagrams"] if d["page"] == page_index]
        matched = set()
        for detection in detections:
            hit = next((i for i, box in enumerate(truth)
                        if i not in matched and _iou(detection.box, box) > 0.5), None)
            if hit is None:
                spurious += 1
            else:
                matched.add(hit)
                found += 1
        missed += len(truth) - len(matched)
    doc.close()
    assert spurious == 0
    # Not all of them: a scan of a board whose dark squares are a printed screen
    # rather than a flat tint is the hardest thing this detector faces, and
    # measured across generated books it loses about one in ten.  Asserting
    # perfection here would only mean the fixtures had gone easy again.
    assert found + missed >= 10, "too few diagrams for the rate to mean anything"
    assert found >= 0.8 * (found + missed), f"found {found}, missed {missed}"


@pytest.mark.parametrize("dark_fill", ["hatch", "stipple"])
def test_finds_a_board_whose_dark_squares_are_textured(piece_set, dark_fill):
    """Books shade dark squares with hatching or a dot screen at least as often
    as with a flat tint, because texture survives monochrome printing better.

    Measured straight, every textured square reads as full of ink and the board
    comes out with sixty of its cells occupied, which is not a chess position --
    so the diagram was thrown away.  It was resolution-dependent too: coarse
    renders blurred the texture into a tint and found the board, finer ones
    resolved the strokes and did not.
    """
    import numpy as np

    rendered = render_diagram(
        BoardMatrix.from_fen(POSITION),
        DiagramStyle(piece_set=piece_set, cell_px=40, dark_fill=dark_fill, screen_ink=35),
    )
    margin = 60
    h, w = rendered.image.shape
    page = np.full((h + 2 * margin, w + 2 * margin), 255, np.uint8)
    page[margin:margin + h, margin:margin + w] = rendered.image

    score, meta = content_score(page, rendered.grid.translated(margin, margin))
    assert meta["occupied_cells"] <= 40, "the texture is being counted as pieces"
    assert score > 0.5

    detections = detect_boards(page)
    assert len(detections) == 1, [d.score for d in detections]
