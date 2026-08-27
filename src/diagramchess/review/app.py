"""The review server.

Correcting a diagram has to be faster than setting the position up by hand, or
nobody will do it twice -- and if nobody does it, the model never improves.  So
the whole page is built around the keyboard: the cursor starts on the square the
model is least sure of, one keystroke sets a piece, and Enter saves and jumps to
the next diagram.  The crops are shown at the size they were classified at, so
you are checking exactly what the model saw.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..board import BLACK_AT_BOTTOM, WHITE_AT_BOTTOM, BoardMatrix
from ..labels import EMPTY, LABELS, LABEL_NAMES
from ..predict import Predictor, bank_for_book
from ..store import Workspace

STATIC = Path(__file__).parent / "static"


def create_app(workspace: Workspace, predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(title="diagramchess review")
    app.state.workspace = workspace
    app.state.predictor = predictor
    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # -- pages ----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/review", response_class=HTMLResponse)
    def review_page() -> str:
        return (STATIC / "review.html").read_text()

    # -- data -----------------------------------------------------------

    @app.get("/api/stats")
    def stats() -> dict:
        model = workspace.active_model()
        return {
            "stats": workspace.stats(),
            "books": [dict(b) for b in workspace.books()],
            "model": dict(model) if model else None,
            "labels": [{"label": l, "name": LABEL_NAMES[l]} for l in LABELS],
        }

    @app.get("/api/queue")
    def queue(book_id: int | None = None, status: str = "pending",
              order: str = "uncertain", limit: int = 200) -> dict:
        rows = workspace.diagrams(book_id=book_id, status=status or None,
                                 order=order, limit=limit)
        return {"diagrams": [_summary(row) for row in rows]}

    @app.get("/api/diagram/{diagram_id}")
    def diagram(diagram_id: int) -> dict:
        row = workspace.diagram(diagram_id)
        if row is None:
            raise HTTPException(404, f"no diagram {diagram_id}")
        squares = _load_squares(workspace, diagram_id)
        stored = {(int(s["row"]), int(s["col"])): s for s in workspace.squares(diagram_id)}

        cells = []
        for index in range(64):
            r, c = index // 8, index % 8
            record = stored.get((r, c))
            predicted = record["predicted"] if record else None
            label = record["label"] if record else None
            cells.append({
                "index": index, "row": r, "col": c,
                "predicted": predicted or EMPTY,
                "confidence": float(record["confidence"]) if record and record["confidence"] is not None else None,
                "label": label or predicted or EMPTY,
                "verified": label is not None,
                "image": _png_data_uri(squares[index]) if squares is not None else None,
            })

        neighbours = workspace.diagrams(book_id=int(row["book_id"]), status="pending", order="uncertain", limit=400)
        order = [int(n["id"]) for n in neighbours]
        position = order.index(diagram_id) if diagram_id in order else -1
        return {
            "diagram": _summary(row),
            "cells": cells,
            "crop": f"/api/crop/{diagram_id}",
            "page_image": f"/api/page/{diagram_id}",
            "next_id": order[position + 1] if 0 <= position < len(order) - 1 else None,
            "prev_id": order[position - 1] if position > 0 else None,
            "remaining": len(order),
        }

    @app.get("/api/crop/{diagram_id}")
    def crop(diagram_id: int) -> Response:
        row = workspace.diagram(diagram_id)
        if row is None:
            raise HTTPException(404, "no such diagram")
        path = workspace.root / row["crop_path"]
        if not path.exists():
            raise HTTPException(404, "the crop for this diagram is missing")
        return Response(path.read_bytes(), media_type="image/png")

    @app.get("/api/page/{diagram_id}")
    def page_image(diagram_id: int) -> Response:
        """The whole page, with the diagram outlined, for checking a detection."""
        row = workspace.diagram(diagram_id)
        if row is None:
            raise HTTPException(404, "no such diagram")
        book = workspace.book(int(row["book_id"]))
        from .. import pdfio

        try:
            doc = pdfio.open_pdf(book["path"])
        except Exception as exc:
            raise HTTPException(404, f"cannot open {book['path']}: {exc}") from exc
        render = pdfio.render_page(doc, int(row["page"]), dpi=int(book["dpi"]))
        doc.close()
        image = cv2.cvtColor(render.image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(image, (int(row["x0"]), int(row["y0"])), (int(row["x1"]), int(row["y1"])),
                      (0, 90, 220), 3)
        scale = 900 / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise HTTPException(500, "could not render the page")
        return Response(buf.tobytes(), media_type="image/png")

    @app.post("/api/diagram/{diagram_id}/save")
    async def save(diagram_id: int, request: Request) -> dict:
        row = workspace.diagram(diagram_id)
        if row is None:
            raise HTTPException(404, "no such diagram")
        body = await request.json()
        labels = body.get("labels")
        if not isinstance(labels, list) or len(labels) != 64:
            raise HTTPException(400, "labels must be a list of 64 piece characters")
        for label in labels:
            if label not in LABELS:
                raise HTTPException(400, f"not a piece label: {label!r}")

        orientation = body.get("orientation", WHITE_AT_BOTTOM)
        if orientation not in (WHITE_AT_BOTTOM, BLACK_AT_BOTTOM):
            raise HTTPException(400, f"unknown orientation: {orientation!r}")
        side = body.get("side_to_move", "w")
        if side not in ("w", "b"):
            raise HTTPException(400, "side_to_move must be 'w' or 'b'")
        castling = body.get("castling")

        board = BoardMatrix.from_labels(labels, orientation=orientation,
                                        side_to_move=side, castling=castling)
        workspace.save_review(diagram_id, labels, orientation, side,
                              board.to_fen(), castling, status=body.get("status", "verified"))
        return {
            "ok": True,
            "fen": board.to_fen(),
            "lichess": board.lichess_url(),
            "problems": board.problems(),
            "corrections": _count_corrections(workspace, diagram_id, labels),
        }

    @app.post("/api/diagram/{diagram_id}/status")
    async def set_status(diagram_id: int, request: Request) -> dict:
        body = await request.json()
        status = body.get("status")
        if status not in ("pending", "verified", "rejected"):
            raise HTTPException(400, f"unknown status: {status!r}")
        workspace.set_status(diagram_id, status)
        return {"ok": True, "status": status}

    @app.post("/api/diagram/{diagram_id}/reread")
    async def reread(diagram_id: int, request: Request) -> dict:
        """Read the diagram again, now that this book has more exemplars on file.

        This is the loop closing in real time: verify two diagrams, press this on
        the third, and the squares that were guesses become matches.
        """
        if app.state.predictor is None:
            raise HTTPException(400, "no model is loaded; start the server with --model")
        row = workspace.diagram(diagram_id)
        if row is None:
            raise HTTPException(404, "no such diagram")
        squares = _load_squares(workspace, diagram_id)
        if squares is None:
            raise HTTPException(404, "the square crops for this diagram are missing")
        body = await request.json() if await request.body() else {}
        bank = bank_for_book(workspace, int(row["book_id"])) if body.get("use_exemplars", True) else None
        reading = app.state.predictor.read_squares(squares, bank)
        board = reading.to_board(caption=row["caption"])
        active = workspace.active_model()
        workspace.set_prediction(diagram_id, reading.labels, reading.confidence,
                                 board.to_fen(), board.orientation, board.side_to_move,
                                 int(active["id"]) if active else None)
        return {
            "ok": True,
            "source": reading.source,
            "exemplars": len(bank) if bank else 0,
            "labels": reading.labels,
            "confidence": reading.confidence,
            "orientation": board.orientation,
            "side_to_move": board.side_to_move,
        }

    @app.post("/api/fen")
    async def fen(request: Request) -> JSONResponse:
        """Turn a grid of labels into a FEN and the links that open it."""
        body = await request.json()
        labels = body.get("labels", [])
        if len(labels) != 64:
            raise HTTPException(400, "labels must be a list of 64 piece characters")
        board = BoardMatrix.from_labels(
            labels,
            orientation=body.get("orientation", WHITE_AT_BOTTOM),
            side_to_move=body.get("side_to_move", "w"),
            castling=body.get("castling"),
        )
        return JSONResponse({
            "fen": board.to_fen(),
            "lichess": board.lichess_url(),
            "lichess_editor": board.lichess_url("editor"),
            "problems": board.problems(),
            "counts": board.counts(),
        })

    return app


def _summary(row) -> dict:
    return {
        "id": int(row["id"]),
        "book_id": int(row["book_id"]),
        "page": int(row["page"]),
        "status": row["status"],
        "score": float(row["detect_score"]),
        "min_confidence": float(row["min_confidence"]) if row["min_confidence"] is not None else None,
        "caption": row["caption"] or "",
        "fen": row["fen"] or row["predicted_fen"] or "",
        "predicted_fen": row["predicted_fen"] or "",
        "orientation": row["orientation"],
        "side_to_move": row["side_to_move"],
        "castling": row["castling"],
        "detect_meta": json.loads(row["detect_meta"] or "{}"),
    }


def _load_squares(workspace: Workspace, diagram_id: int) -> np.ndarray | None:
    path = workspace.squares_path(diagram_id)
    return np.load(path) if path.exists() else None


def _png_data_uri(square: np.ndarray) -> str:
    """Inline a square crop, so a board is one request rather than sixty-five."""
    ok, buf = cv2.imencode(".png", square)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _count_corrections(workspace: Workspace, diagram_id: int, labels: list[str]) -> int:
    """How many squares the human disagreed with, which is the number that matters."""
    stored = {(int(s["row"]), int(s["col"])): s["predicted"] for s in workspace.squares(diagram_id)}
    changed = 0
    for index, label in enumerate(labels):
        predicted = stored.get((index // 8, index % 8))
        if predicted is not None and predicted != label:
            changed += 1
    return changed
