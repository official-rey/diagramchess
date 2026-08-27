"""The workspace: a SQLite database and a tree of cached images.

Everything the tool learns about your books lives here -- which diagrams were
found, what the model thought, what you corrected it to, and which model version
said what.  Keeping the corrections in one place is the whole point: they are the
training set that makes the next run better, and they are yours, so they survive
retraining, reinstalling, and changing your mind about the model.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dataset import VerifiedSquares
from .labels import LABEL_TO_INDEX

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    digest TEXT NOT NULL UNIQUE,
    title TEXT,
    pages INTEGER NOT NULL,
    dpi INTEGER NOT NULL,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS diagrams (
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    x0 REAL NOT NULL, y0 REAL NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL,
    grid TEXT NOT NULL,
    source TEXT NOT NULL,
    detect_score REAL NOT NULL,
    detect_meta TEXT NOT NULL DEFAULT '{}',
    crop_path TEXT NOT NULL,
    caption TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    orientation TEXT NOT NULL DEFAULT 'white',
    side_to_move TEXT NOT NULL DEFAULT 'w',
    castling TEXT,
    predicted_fen TEXT,
    fen TEXT,
    min_confidence REAL,
    model_id INTEGER REFERENCES models(id),
    reviewed_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (book_id, page, x0, y0)
);

CREATE TABLE IF NOT EXISTS squares (
    id INTEGER PRIMARY KEY,
    diagram_id INTEGER NOT NULL REFERENCES diagrams(id) ON DELETE CASCADE,
    row INTEGER NOT NULL,
    col INTEGER NOT NULL,
    predicted TEXT,
    confidence REAL,
    label TEXT,
    UNIQUE (diagram_id, row, col)
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    metrics TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS diagrams_book ON diagrams(book_id, page);
CREATE INDEX IF NOT EXISTS diagrams_status ON diagrams(status, min_confidence);
CREATE INDEX IF NOT EXISTS squares_diagram ON squares(diagram_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Book:
    id: int
    path: str
    digest: str
    title: str
    pages: int
    dpi: int


class Workspace:
    """A directory holding the database, the cached page images and the models."""

    def __init__(self, root: str | Path = ".diagramchess"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "crops").mkdir(exist_ok=True)
        (self.root / "squares").mkdir(exist_ok=True)
        (self.root / "pages").mkdir(exist_ok=True)
        (self.root / "models").mkdir(exist_ok=True)
        self.db_path = self.root / "diagramchess.db"
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        with self._connection:
            self._connection.executescript(SCHEMA)
            self._connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # -- plumbing --------------------------------------------------------

    @contextmanager
    def write(self):
        with self._connection:
            yield self._connection

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return list(self._connection.execute(sql, params))

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self._connection.execute(sql, params).fetchone()

    def close(self) -> None:
        self._connection.close()

    # -- books -----------------------------------------------------------

    def add_book(self, path: str | Path, digest: str, pages: int, dpi: int, title: str = "") -> int:
        existing = self.one("SELECT id FROM books WHERE digest = ?", (digest,))
        if existing:
            # Same file, possibly moved: keep the diagrams, refresh the path.
            with self.write() as db:
                db.execute("UPDATE books SET path = ? WHERE id = ?", (str(path), existing["id"]))
            return int(existing["id"])
        with self.write() as db:
            cursor = db.execute(
                "INSERT INTO books (path, digest, title, pages, dpi, added_at) VALUES (?,?,?,?,?,?)",
                (str(path), digest, title or Path(path).stem, pages, dpi, _now()),
            )
        return int(cursor.lastrowid)

    def books(self) -> list[sqlite3.Row]:
        return self.query("""
            SELECT b.*,
                   (SELECT COUNT(*) FROM diagrams d WHERE d.book_id = b.id) AS diagram_count,
                   (SELECT COUNT(*) FROM diagrams d WHERE d.book_id = b.id AND d.status = 'verified') AS verified_count
            FROM books b ORDER BY b.added_at DESC
        """)

    def book(self, book_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM books WHERE id = ?", (book_id,))

    # -- diagrams --------------------------------------------------------

    def add_diagram(
        self,
        book_id: int,
        page: int,
        box: tuple[float, float, float, float],
        grid: dict,
        source: str,
        score: float,
        crop_path: str,
        caption: str = "",
        detect_meta: dict | None = None,
    ) -> int | None:
        """Insert a detection, or return ``None`` if this one is already stored."""
        with self.write() as db:
            cursor = db.execute(
                """INSERT OR IGNORE INTO diagrams
                   (book_id, page, x0, y0, x1, y1, grid, source, detect_score, detect_meta,
                    crop_path, caption, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (book_id, page, box[0], box[1], box[2], box[3], json.dumps(grid), source,
                 score, json.dumps(detect_meta or {}), crop_path, caption, _now()),
            )
        return int(cursor.lastrowid) if cursor.rowcount else None

    def diagram(self, diagram_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM diagrams WHERE id = ?", (diagram_id,))

    def diagrams(
        self,
        book_id: int | None = None,
        status: str | None = None,
        order: str = "page",
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        """List diagrams, optionally ordered by how much they need a human.

        ``order='uncertain'`` puts the least confident first.  That ordering is
        the active learning loop: correcting the squares the model is worst at
        teaches it more per keystroke than correcting the ones it already knows.
        """
        clauses, params = [], []
        if book_id is not None:
            clauses.append("book_id = ?")
            params.append(book_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        ordering = {
            "page": "page ASC, y0 ASC, x0 ASC",
            "uncertain": "COALESCE(min_confidence, -1) ASC, page ASC",
            "recent": "COALESCE(reviewed_at, created_at) DESC",
        }.get(order, "page ASC, y0 ASC, x0 ASC")
        sql = f"SELECT * FROM diagrams {where} ORDER BY {ordering}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.query(sql, tuple(params))

    def set_prediction(
        self,
        diagram_id: int,
        labels: list[str],
        confidences: list[float],
        fen: str,
        orientation: str,
        side_to_move: str,
        model_id: int | None,
    ) -> None:
        """Record what the model read, leaving any human corrections alone."""
        with self.write() as db:
            db.execute(
                """UPDATE diagrams
                   SET predicted_fen = ?, min_confidence = ?, model_id = ?,
                       orientation = CASE WHEN status = 'verified' THEN orientation ELSE ? END,
                       side_to_move = CASE WHEN status = 'verified' THEN side_to_move ELSE ? END,
                       fen = CASE WHEN status = 'verified' THEN fen ELSE ? END
                   WHERE id = ?""",
                (fen, min(confidences) if confidences else None, model_id,
                 orientation, side_to_move, fen, diagram_id),
            )
            db.executemany(
                """INSERT INTO squares (diagram_id, row, col, predicted, confidence)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT (diagram_id, row, col)
                   DO UPDATE SET predicted = excluded.predicted, confidence = excluded.confidence""",
                [(diagram_id, i // 8, i % 8, labels[i], confidences[i]) for i in range(64)],
            )

    def save_review(
        self,
        diagram_id: int,
        labels: list[str],
        orientation: str,
        side_to_move: str,
        fen: str,
        castling: str | None = None,
        status: str = "verified",
    ) -> None:
        """Record a human's reading of a diagram.  This is training data."""
        with self.write() as db:
            db.execute(
                """UPDATE diagrams
                   SET status = ?, orientation = ?, side_to_move = ?, fen = ?,
                       castling = ?, reviewed_at = ?
                   WHERE id = ?""",
                (status, orientation, side_to_move, fen, castling, _now(), diagram_id),
            )
            db.executemany(
                """INSERT INTO squares (diagram_id, row, col, label)
                   VALUES (?,?,?,?)
                   ON CONFLICT (diagram_id, row, col) DO UPDATE SET label = excluded.label""",
                [(diagram_id, i // 8, i % 8, labels[i]) for i in range(64)],
            )

    def set_status(self, diagram_id: int, status: str) -> None:
        with self.write() as db:
            db.execute("UPDATE diagrams SET status = ?, reviewed_at = ? WHERE id = ?",
                       (status, _now(), diagram_id))

    def squares(self, diagram_id: int) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM squares WHERE diagram_id = ? ORDER BY row, col", (diagram_id,)
        )

    # -- paths -----------------------------------------------------------

    def crop_path(self, book_id: int, page: int, index: int) -> Path:
        directory = self.root / "crops" / f"book{book_id:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"p{page:04d}_{index:02d}.png"

    def squares_path(self, diagram_id: int) -> Path:
        return self.root / "squares" / f"d{diagram_id:06d}.npy"

    def page_path(self, book_id: int, page: int) -> Path:
        directory = self.root / "pages" / f"book{book_id:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"p{page:04d}.png"

    # -- models ----------------------------------------------------------

    def register_model(self, path: str | Path, trained_at: str, metrics: dict,
                       notes: str = "", activate: bool = True) -> int:
        with self.write() as db:
            cursor = db.execute(
                "INSERT INTO models (path, trained_at, metrics, notes, active) VALUES (?,?,?,?,0)",
                (str(path), trained_at, json.dumps(metrics), notes),
            )
            model_id = int(cursor.lastrowid)
            if activate:
                db.execute("UPDATE models SET active = 0")
                db.execute("UPDATE models SET active = 1 WHERE id = ?", (model_id,))
        return model_id

    def active_model(self) -> sqlite3.Row | None:
        return self.one("SELECT * FROM models WHERE active = 1 ORDER BY id DESC LIMIT 1")

    def models(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM models ORDER BY id DESC")

    # -- training data ---------------------------------------------------

    def verified_squares(self, square_size: int = 48) -> VerifiedSquares:
        """Every hand-corrected square, ready to train on.

        Squares whose crop is missing are skipped rather than guessed at; a
        stale cache should cost you those examples, not corrupt the rest.
        """
        rows = self.query("""
            SELECT s.diagram_id, s.row, s.col, s.label, d.book_id
            FROM squares s JOIN diagrams d ON d.id = s.diagram_id
            WHERE d.status = 'verified' AND s.label IS NOT NULL
            ORDER BY s.diagram_id, s.row, s.col
        """)
        images: list[np.ndarray] = []
        labels: list[int] = []
        books: list[int] = []
        cache: dict[int, np.ndarray | None] = {}
        for row in rows:
            diagram_id = int(row["diagram_id"])
            if diagram_id not in cache:
                path = self.squares_path(diagram_id)
                cache[diagram_id] = np.load(path) if path.exists() else None
            stack = cache[diagram_id]
            if stack is None or stack.shape[0] != 64:
                continue
            square = stack[int(row["row"]) * 8 + int(row["col"])]
            if square.shape[0] != square_size:
                import cv2
                square = cv2.resize(square, (square_size, square_size), interpolation=cv2.INTER_AREA)
            images.append(square)
            labels.append(LABEL_TO_INDEX[row["label"]])
            books.append(int(row["book_id"]))
        if not images:
            return VerifiedSquares(
                np.zeros((0, square_size, square_size), np.uint8),
                np.zeros(0, np.int64), np.zeros(0, np.int64),
            )
        return VerifiedSquares(np.stack(images), np.array(labels, np.int64), np.array(books, np.int64))

    def stats(self) -> dict:
        row = self.one("""
            SELECT
              (SELECT COUNT(*) FROM books) AS books,
              (SELECT COUNT(*) FROM diagrams) AS diagrams,
              (SELECT COUNT(*) FROM diagrams WHERE status = 'verified') AS verified,
              (SELECT COUNT(*) FROM diagrams WHERE status = 'pending') AS pending,
              (SELECT COUNT(*) FROM diagrams WHERE status = 'rejected') AS rejected,
              (SELECT COUNT(*) FROM squares WHERE label IS NOT NULL) AS labelled_squares
        """)
        return dict(row) if row else {}
