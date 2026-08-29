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
from ..jobs import JobRunner
from ..labels import EMPTY, LABELS, LABEL_NAMES
from ..predict import Predictor, bank_for_book
from ..store import Workspace

STATIC = Path(__file__).parent / "static"

# A book is a few tens of megabytes; a gigabyte arriving on this endpoint is a
# mistake or a nuisance, and either way should not fill the disk first.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
UNSAFE_IN_A_NAME = str.maketrans({c: "_" for c in '/\\:*?"<>|\0'})


def create_app(workspace: Workspace, predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(title="diagramchess review")
    app.state.workspace = workspace
    app.state.predictor = predictor
    app.state.jobs = JobRunner()
    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def only_our_own_pages_may_change_anything(request: Request, call_next):
        """The server listens on localhost, which any page in the browser can
        reach.  Reading diagrams that way is harmless, but this app can now
        also import files and start training, so a request that changes
        something must come from this app's own origin.  Browsers attach
        Origin to every cross-site write; tools that are not browsers send
        none, and are left alone.
        """
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = request.headers.get("origin")
            host = request.headers.get("host", "")
            if origin is not None and origin not in (f"http://{host}", f"https://{host}"):
                return JSONResponse({"detail": "cross-origin write refused"}, status_code=403)
        return await call_next(request)

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

    # -- books, and the long jobs that fill them -------------------------

    def _ingest_job(job, pdf_path: Path, dpi: int, pages, label: str) -> dict:
        from ..pdfio import open_pdf
        from ..pipeline import ingest

        doc = open_pdf(pdf_path)
        page_count = len(doc)
        doc.close()
        wanted = len(list(pages)) if pages is not None else page_count
        job.step(total=wanted, note=f"reading {label}")

        seen = found_so_far = 0

        def progress(page_index: int, found: int) -> None:
            nonlocal seen, found_so_far
            seen += 1
            found_so_far += found
            job.step(done=seen, found=found_so_far,
                     note=f"page {page_index + 1} of {page_count}")

        report = ingest(workspace, pdf_path, dpi=dpi, pages=pages,
                        predictor=app.state.predictor,
                        model_id=_active_model_id(workspace, app.state.predictor),
                        progress=progress)
        return {"book_id": report.book_id, "summary": report.describe(),
                "diagrams": report.stored + report.skipped_existing,
                "read": report.predicted}

    @app.post("/api/books")
    async def add_book(request: Request, name: str = "book.pdf", dpi: int = 200,
                       pages: str | None = None) -> dict:
        """Take a PDF as a raw body and start reading it.

        Raw rather than multipart so the package needs no form parser, and
        streamed rather than read whole so a large book never has to fit in
        memory twice.
        """
        if app.state.jobs.busy():
            raise HTTPException(409, "something is already running; wait for it to finish")
        page_range = _parse_pages(pages)
        books = workspace.root / "books"
        books.mkdir(exist_ok=True)

        safe = Path(name).name.translate(UNSAFE_IN_A_NAME).strip() or "book.pdf"
        if not safe.lower().endswith(".pdf"):
            safe += ".pdf"
        target = books / safe
        stem, suffix, n = target.stem, target.suffix, 2
        while target.exists():
            target = books / f"{stem}-{n}{suffix}"
            n += 1

        size = 0
        try:
            with target.open("wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "that file is larger than 512 MB")
                    handle.write(chunk)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        if size == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(400, "no file was sent")
        if target.read_bytes()[:5] != b"%PDF-":
            target.unlink(missing_ok=True)
            raise HTTPException(400, f"{safe} does not look like a PDF")

        job = app.state.jobs.submit(
            "ingest", safe,
            lambda job: _ingest_job(job, target, dpi, page_range, safe),
            note="opening the book")
        return {"job": job.as_dict()}

    @app.post("/api/books/demo")
    def add_demo_book(pages: int = 8) -> dict:
        """A generated book, so a first run has something to open."""
        if app.state.jobs.busy():
            raise HTTPException(409, "something is already running; wait for it to finish")
        books = workspace.root / "books"
        books.mkdir(exist_ok=True)
        target = books / "sample-book.pdf"

        def work(job) -> dict:
            from ..demo import build_demo_book

            job.step(total=max(1, pages), note="drawing a sample book")
            build_demo_book(target, pages=pages, seed=11)
            return _ingest_job(job, target, 200, None, "sample-book.pdf")

        return {"job": app.state.jobs.submit("ingest", "sample-book.pdf", work,
                                             note="drawing a sample book").as_dict()}

    @app.delete("/api/books/{book_id}")
    def forget_book(book_id: int) -> dict:
        row = workspace.book(book_id)
        if row is None:
            raise HTTPException(404, f"no book {book_id}")
        with workspace.write() as db:
            db.execute("DELETE FROM books WHERE id = ?", (book_id,))
        # Only files this app copied in are removed; a book opened from
        # somewhere else on disk is the reader's, not ours to delete.
        path = Path(row["path"])
        if path.is_file() and path.parent == (workspace.root / "books").resolve():
            path.unlink(missing_ok=True)
        return {"ok": True}

    @app.post("/api/reread")
    def start_reread(book_id: int | None = None, include_verified: bool = False) -> dict:
        if app.state.predictor is None:
            raise HTTPException(400, "no model is loaded")
        if app.state.jobs.busy():
            raise HTTPException(409, "something is already running; wait for it to finish")

        def work(job) -> dict:
            from ..pipeline import repredict

            job.step(note="reading the diagrams again")
            count = repredict(workspace, app.state.predictor, book_id=book_id,
                              model_id=_active_model_id(workspace, app.state.predictor),
                              include_verified=include_verified,
                              progress=lambda done, total: job.step(done=done, total=total))
            return {"summary": f"read {count} diagram(s) again", "book_id": book_id}

        return {"job": app.state.jobs.submit("reread", "every diagram", work,
                                             note="getting ready").as_dict()}

    @app.post("/api/train")
    def start_train(epochs: int = 8, steps: int = 300) -> dict:
        if app.state.jobs.busy():
            raise HTTPException(409, "something is already running; wait for it to finish")

        def work(job) -> dict:
            from datetime import datetime as _dt

            from ..train import TrainConfig, train, training_styles

            verified = workspace.verified_squares()
            styles = training_styles(workspace)
            job.step(total=epochs,
                     note=f"{len(verified)} verified squares, {len(styles)} figurine style(s)")
            stamp = _dt.now().strftime("%Y%m%d-%H%M%S")
            output = workspace.root / "models" / f"piece-net-{stamp}.pt"
            report = train(
                output,
                TrainConfig(epochs=epochs, steps_per_epoch=steps, piece_sets=styles),
                verified=verified if len(verified) else None,
                progress=lambda row: job.step(
                    done=row["epoch"],
                    note=f"epoch {row['epoch']} of {epochs} · "
                         f"held-out accuracy {row['val_accuracy'] * 100:.1f}%"),
            )
            model_id = workspace.register_model(report.checkpoint_path, report.trained_at,
                                                report.metrics, notes=f"epochs={epochs}",
                                                activate=True)
            # The new model is the point of the exercise, so the server starts
            # using it now rather than at the next restart.
            app.state.predictor = Predictor(report.checkpoint_path)
            return {"summary": report.describe(), "model_id": model_id}

        return {"job": app.state.jobs.submit("train", "the piece reader", work,
                                             note="gathering your corrections").as_dict()}

    @app.post("/api/pieces/fetch")
    def start_fetch() -> dict:
        if app.state.jobs.busy():
            raise HTTPException(409, "something is already running; wait for it to finish")

        def work(job) -> dict:
            from ..artwork import fetch
            from ..pieces import piece_sets_in

            job.step(note="downloading figurine styles")
            directory = workspace.root / "pieces"
            fetch(directory)
            found = piece_sets_in(directory)
            return {"summary": f"{len(found)} style(s) available to train on"}

        return {"job": app.state.jobs.submit("pieces", "figurine styles", work,
                                             note="contacting the server").as_dict()}

    @app.get("/api/jobs")
    def jobs() -> dict:
        runner = app.state.jobs
        active = runner.active()
        return {"active": active.as_dict() if active else None,
                "recent": [j.as_dict() for j in runner.recent()]}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.as_dict()

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


def _parse_pages(spec: str | None) -> list[int] | None:
    """'10-40' or '7', one-based and inclusive, as the command line takes it."""
    if not spec or not spec.strip():
        return None
    spec = spec.strip()
    try:
        if "-" in spec:
            start, end = spec.split("-", 1)
            first, last = int(start), int(end)
        else:
            first = last = int(spec)
    except ValueError:
        raise HTTPException(400, f"cannot read {spec!r} as a page range like 10-40") from None
    if first < 1 or last < first:
        raise HTTPException(400, f"{spec!r} is not a page range like 10-40")
    return list(range(first - 1, last))


def _active_model_id(workspace: Workspace, predictor: Predictor | None) -> int | None:
    if predictor is None:
        return None
    active = workspace.active_model()
    return int(active["id"]) if active else None


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
