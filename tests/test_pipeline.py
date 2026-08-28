import json

import numpy as np

from diagramchess.board import BoardMatrix
from diagramchess.grid import GridFit
from diagramchess.pipeline import ingest, load_squares, repredict
from diagramchess.predict import ExemplarBank, Predictor, bank_for_book


def test_ingest_finds_stores_and_crops(workspace, demo_pdf):
    report = ingest(workspace, demo_pdf)
    assert report.stored == report.detected > 0
    assert report.predicted == 0            # no model was given

    diagrams = workspace.diagrams()
    assert len(diagrams) == report.stored
    for row in diagrams:
        assert (workspace.root / row["crop_path"]).exists()
        squares = load_squares(workspace, int(row["id"]))
        assert squares is not None and squares.shape == (64, 48, 48)
        grid = GridFit.from_dict(json.loads(row["grid"]))
        assert grid.step_x > 0


def test_ingest_reads_the_caption(workspace, demo_pdf):
    ingest(workspace, demo_pdf)
    captions = [row["caption"] for row in workspace.diagrams()]
    assert sum("to play" in c for c in captions) >= len(captions) - 2


def test_ingesting_twice_keeps_your_work(workspace, demo_pdf):
    """Re-running the detector must not throw away verified corrections."""
    ingest(workspace, demo_pdf)
    first = workspace.diagrams()[0]
    labels = BoardMatrix.from_fen("8/8/4k3/8/2K5/8/6R1/8 w - - 0 1").flat()
    workspace.save_review(int(first["id"]), labels, "white", "w", "8/8/4k3/8/2K5/8/6R1/8 w - - 0 1")

    again = ingest(workspace, demo_pdf)
    assert again.stored == 0
    assert again.skipped_existing == again.detected
    assert len(workspace.books()) == 1
    row = workspace.diagram(int(first["id"]))
    assert row["status"] == "verified"
    assert row["fen"] == "8/8/4k3/8/2K5/8/6R1/8 w - - 0 1"


def test_ingest_can_be_limited_to_a_page_range(workspace, demo_pdf):
    report = ingest(workspace, demo_pdf, pages=[0])
    assert report.pages == 1
    assert all(row["page"] == 0 for row in workspace.diagrams())


def test_repredict_leaves_verified_diagrams_alone(workspace, demo_pdf):
    """The stored prediction on a verified diagram is the record accuracy is measured against."""
    ingest(workspace, demo_pdf)
    rows = workspace.diagrams()
    target = int(rows[0]["id"])
    workspace.set_prediction(target, ["."] * 64, [0.5] * 64, "old", "white", "w", None)
    workspace.save_review(target, ["."] * 64, "white", "w", "8/8/8/8/8/8/8/8 w - - 0 1")

    # A predictor that reads every square as a white king.
    class Always:
        square_size = 48
        has_model = True

        def read_squares(self, squares, bank=None):
            from diagramchess.predict import BoardReading
            return BoardReading(["K"] * 64, [0.9] * 64, np.zeros((64, 13), np.float32))

    count = repredict(workspace, Always(), book_id=int(rows[0]["book_id"]))
    assert count == len(rows) - 1
    assert workspace.diagram(target)["predicted_fen"] == "old"

    count = repredict(workspace, Always(), book_id=int(rows[0]["book_id"]), include_verified=True)
    assert count == len(rows)
    assert workspace.diagram(target)["predicted_fen"] != "old"
    assert workspace.diagram(target)["fen"] == "8/8/8/8/8/8/8/8 w - - 0 1"   # yours is kept


def test_bank_for_book_only_holds_that_book(workspace, demo_pdf):
    ingest(workspace, demo_pdf)
    rows = workspace.diagrams()
    labels = BoardMatrix.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1").flat()
    workspace.save_review(int(rows[0]["id"]), labels, "white", "w", "x")

    bank = bank_for_book(workspace, int(rows[0]["book_id"]))
    assert len(bank) == 64
    assert len(bank_for_book(workspace, 999)) == 0


def test_caption_is_read_from_above_as_well_as_below(workspace, tmp_path):
    """Books caption on both sides of a diagram, and picking one is not safe.

    A real game collection prints the players' names *above* each board;
    reading only the band underneath attached every diagram to its neighbour's
    header, silently and consistently.
    """
    import pymupdf

    from diagramchess import pdfio
    from diagramchess.board import BoardMatrix
    from diagramchess.pieces import available_piece_sets
    from diagramchess.render import DiagramStyle, render_diagram

    rendered = render_diagram(
        BoardMatrix.from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"),
        DiagramStyle(piece_set=available_piece_sets()[0], cell_px=30),
    )
    import cv2

    ok, buf = cv2.imencode(".png", rendered.image)
    assert ok

    doc = pymupdf.open()
    page = doc.new_page(width=400, height=520)
    rect = pymupdf.Rect(80, 90, 320, 330)
    page.insert_text((80, 78), "Hamppe - Steinitz, Vienna 1860", fontsize=9)
    page.insert_image(rect, stream=buf.tobytes())
    path = tmp_path / "above.pdf"
    doc.save(str(path))
    doc.close()

    doc = pdfio.open_pdf(path)
    render = pdfio.render_page(doc, 0, dpi=200)
    box = render.to_pixels(rect)
    caption = pdfio.text_near(doc, 0, box, render)
    doc.close()
    assert "Hamppe" in caption


def test_caption_ignores_a_diagram_typeset_in_a_chess_font(workspace):
    """Books that set diagrams in a chess font put the board in the text layer.

    Those private-use characters are the diagram, not a caption; left in, the
    coordinate strip along the board's edge looks like the nearest line of text.
    """
    from diagramchess.pdfio import _drop_glyph_runs

    glyphs = "".join(chr(c) for c in range(0xF0C8, 0xF0D0))
    assert _drop_glyph_runs(glyphs).strip() == ""
    assert _drop_glyph_runs(f"{glyphs}Black to play").strip() == "Black to play"
    assert _drop_glyph_runs("Diagram 4: White to move") == "Diagram 4: White to move"
