"""Piece artwork, used to draw synthetic diagrams for the cold start.

The classifier has to learn what a rook looks like before it has ever seen your
book, so we draw our own diagrams from whatever piece artwork we can find and
train on those.  Three sources are built in -- the vector set that ships with
python-chess, and the chess glyphs in two system fonts -- and you can drop more
in as PNG files, which is worth doing if you have a book whose figurine style
none of these resemble.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SYMBOLS = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")

#: Unicode chess characters, indexed the same way as the FEN symbols.
UNICODE = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

_FONT_CANDIDATES = (
    ("dejavu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ("freeserif", "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"),
)


@dataclass(frozen=True)
class PieceSet:
    """One visual style of the twelve pieces, rendered on demand."""

    name: str
    kind: str  # "svg", "font" or "png"
    source: str = ""

    def render(self, symbol: str, size: int) -> np.ndarray:
        """The piece as an RGBA array of the given size, transparent around it."""
        if symbol not in SYMBOLS:
            raise ValueError(f"not a piece symbol: {symbol!r}")
        return _render_cached(self.name, self.kind, self.source, symbol, int(size)).copy()


@functools.lru_cache(maxsize=4096)
def _render_cached(name: str, kind: str, source: str, symbol: str, size: int) -> np.ndarray:
    if kind == "svg":
        return _render_svg(symbol, size)
    if kind == "font":
        return _render_font(source, symbol, size)
    if kind == "png":
        return _render_png(source, symbol, size)
    raise ValueError(f"unknown piece set kind: {kind!r}")


def _render_svg(symbol: str, size: int) -> np.ndarray:
    import chess
    import chess.svg
    import pymupdf

    svg = chess.svg.piece(chess.Piece.from_symbol(symbol))
    doc = pymupdf.open(stream=svg.encode("utf-8"), filetype="svg")
    page = doc[0]
    zoom = size / max(page.rect.width, page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=True)
    raw = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:
        raw = np.dstack([raw, np.full(raw.shape[:2], 255, np.uint8)])
    return _fit_canvas(raw, size)


def _render_font(font_path: str, symbol: str, size: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    # Render large and shrink, so hinting at small sizes does not eat the detail.
    work = max(size * 4, 128)
    font = ImageFont.truetype(font_path, int(work * 0.86))
    image = Image.new("RGBA", (work * 2, work * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    glyph = UNICODE[symbol]
    draw.text((work // 2, work // 2), glyph, font=font, fill=(0, 0, 0, 255))
    bbox = image.getbbox()
    if bbox is None:
        raise ValueError(f"font {font_path} has no glyph for {symbol!r}")
    return _fit_canvas(np.array(image.crop(bbox)), size)


def _render_png(directory: str, symbol: str, size: int) -> np.ndarray:
    from PIL import Image

    colour = "w" if symbol.isupper() else "b"
    stem = f"{colour}{symbol.upper()}"
    for name in (stem, stem.lower(), f"{colour}_{symbol.upper()}"):
        path = Path(directory) / f"{name}.png"
        if path.exists():
            image = Image.open(path).convert("RGBA")
            return _fit_canvas(np.array(image), size)
    raise FileNotFoundError(f"no image for {symbol!r} in {directory}")


def _fit_canvas(rgba: np.ndarray, size: int) -> np.ndarray:
    """Trim to the ink, scale to fit, and centre on a transparent square."""
    import cv2

    alpha = rgba[:, :, 3]
    ys, xs = np.nonzero(alpha > 8)
    if len(ys) == 0:
        return np.zeros((size, size, 4), np.uint8)
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = rgba.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 4), np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def available_piece_sets(extra_dir: str | Path | None = None) -> list[PieceSet]:
    """Every piece style we can draw right now.

    ``extra_dir`` may hold subdirectories of PNG files named ``wK.png``,
    ``bQ.png`` and so on; each subdirectory becomes another style.
    """
    sets: list[PieceSet] = [PieceSet("cburnett", "svg")]
    for name, path in _FONT_CANDIDATES:
        if Path(path).exists() and _font_has_pieces(path):
            sets.append(PieceSet(name, "font", path))
    if extra_dir:
        root = Path(extra_dir)
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir() and any(child.glob("*.png")):
                    sets.append(PieceSet(f"custom-{child.name}", "png", str(child)))
    return sets


def _font_has_pieces(path: str) -> bool:
    """Does this font actually draw the chess characters, or just tofu boxes?"""
    try:
        from fontTools.ttLib import TTFont  # optional; fall back to a render test
    except ImportError:
        try:
            for symbol in ("K", "p"):
                if _render_font(path, symbol, 32)[:, :, 3].max() == 0:
                    return False
            return True
        except Exception:
            return False
    try:
        font = TTFont(path, fontNumber=0, lazy=True)
        cmap = font.getBestCmap()
        return all(ord(UNICODE[s]) in cmap for s in SYMBOLS)
    except Exception:
        return False
