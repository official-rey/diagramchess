"""Wiring: PDF in, stored diagrams and predictions out."""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import pdfio
from .board import BoardMatrix, guess_orientation, guess_side_to_move
from .detect import Detection, crop_board, detect_boards
from .grid import GridFit, extract_squares
from .model import SQUARE_SIZE
from .predict import ExemplarBank, Predictor, bank_for_book
from .store import Workspace


@dataclass
class IngestReport:
    book_id: int
    pages: int
    detected: int = 0
    stored: int = 0
    predicted: int = 0
    skipped_existing: int = 0

    def describe(self) -> str:
        return (
            f"book {self.book_id}: {self.pages} pages, {self.detected} diagrams detected, "
            f"{self.stored} new, {self.skipped_existing} already known, {self.predicted} read"
        )


def ingest(
    workspace: Workspace,
    pdf_path: str | Path,
    dpi: int = pdfio.DEFAULT_DPI,
    pages: range | list[int] | None = None,
    predictor: Predictor | None = None,
    model_id: int | None = None,
    cache_pages: bool = False,
    progress=None,
) -> IngestReport:
    """Find every diagram in a PDF, store it, and read it if a model is available.

    Re-ingesting the same file is safe and cheap: diagrams already stored are
    recognised by position and left alone, so your verified work survives a
    second pass at a different resolution or after a detector improvement.
    """
    pdf_path = Path(pdf_path)
    digest = pdfio.file_digest(pdf_path)
    doc = pdfio.open_pdf(pdf_path)
    book_id = workspace.add_book(pdf_path, digest, len(doc), dpi)
    page_indices = list(pages) if pages is not None else list(range(len(doc)))
    report = IngestReport(book_id=book_id, pages=len(page_indices))

    bank = bank_for_book(workspace, book_id) if predictor else None

    for page_index in page_indices:
        render = pdfio.render_page(doc, page_index, dpi=dpi)
        proposals = pdfio.embedded_image_boxes(doc, page_index, render)
        proposals += pdfio.vector_drawing_boxes(doc, page_index, render)
        detections = detect_boards(render.image, proposals)
        report.detected += len(detections)

        if cache_pages and detections:
            cv2.imwrite(str(workspace.page_path(book_id, page_index)), render.image)

        for index, detection in enumerate(detections):
            stored = _store_detection(
                workspace, doc, render, book_id, page_index, index, detection,
            )
            if stored is None:
                report.skipped_existing += 1
                continue
            report.stored += 1
            if predictor is not None:
                _predict_diagram(workspace, predictor, bank, stored, model_id)
                report.predicted += 1
        if progress:
            progress(page_index, len(detections))

    doc.close()
    return report


@dataclass
class StoredDiagram:
    diagram_id: int
    squares: np.ndarray
    caption: str
    grid: GridFit


def _store_detection(
    workspace: Workspace,
    doc,
    render: pdfio.PageRender,
    book_id: int,
    page_index: int,
    index: int,
    detection: Detection,
) -> StoredDiagram | None:
    crop, grid = crop_board(render.image, detection)
    caption = pdfio.text_near(doc, page_index, detection.box, render)
    crop_path = workspace.crop_path(book_id, page_index, index)

    diagram_id = workspace.add_diagram(
        book_id=book_id,
        page=page_index,
        box=detection.box,
        grid=grid.as_dict(),
        source=detection.source,
        score=detection.score,
        crop_path=str(crop_path.relative_to(workspace.root)),
        caption=caption,
        detect_meta=detection.meta,
    )
    if diagram_id is None:
        return None

    cv2.imwrite(str(crop_path), crop)
    squares = extract_squares(crop, grid, size=SQUARE_SIZE)
    np.save(workspace.squares_path(diagram_id), squares)
    return StoredDiagram(diagram_id, squares, caption, grid)


def _predict_diagram(
    workspace: Workspace,
    predictor: Predictor,
    bank: ExemplarBank | None,
    stored: StoredDiagram,
    model_id: int | None,
) -> BoardMatrix:
    reading = predictor.read_squares(stored.squares, bank)
    board = reading.to_board(caption=stored.caption)
    workspace.set_prediction(
        stored.diagram_id,
        reading.labels,
        reading.confidence,
        board.to_fen(),
        board.orientation,
        board.side_to_move,
        model_id,
    )
    return board


def repredict(
    workspace: Workspace,
    predictor: Predictor,
    book_id: int | None = None,
    model_id: int | None = None,
    include_verified: bool = False,
    progress=None,
) -> int:
    """Re-read stored diagrams with the current model.

    Run this after retraining.  Verified diagrams keep their human reading and
    are only re-read when you ask, so that the numbers you can compare -- what
    the model says against what you said -- stay meaningful.
    """
    rows = workspace.diagrams(book_id=book_id)
    banks: dict[int, ExemplarBank] = {}
    count = 0
    for row in rows:
        if row["status"] == "verified" and not include_verified:
            continue
        path = workspace.squares_path(int(row["id"]))
        if not path.exists():
            continue
        squares = np.load(path)
        book = int(row["book_id"])
        if book not in banks:
            banks[book] = bank_for_book(workspace, book)
        stored = StoredDiagram(int(row["id"]), squares, row["caption"] or "",
                               GridFit.from_dict(json.loads(row["grid"])))
        _predict_diagram(workspace, predictor, banks[book], stored, model_id)
        count += 1
        if progress:
            progress(count, len(rows))
    return count


def load_squares(workspace: Workspace, diagram_id: int) -> np.ndarray | None:
    path = workspace.squares_path(diagram_id)
    return np.load(path) if path.exists() else None
