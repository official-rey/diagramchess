"""Piece artwork, used to draw synthetic diagrams for the cold start.

The classifier has to learn what a rook looks like before it has ever seen your
book, so we draw our own diagrams from whatever piece artwork we can find and
train on those.  Three sources are built in -- the vector set that ships with
python-chess, and the chess glyphs in two system fonts.

Three is not many, and figurine styles differ far more between real books than
these differ from each other, so `dgc pieces --fetch` pulls down several dozen
more.  Point ``available_piece_sets(extra_dir=...)`` at a directory of styles,
each a subdirectory of twelve files named ``wK`` / ``bQ`` and so on, as either
SVG or PNG.
"""

from __future__ import annotations

import functools
import io
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
        size = int(size)
        master = _render_cached(self.name, self.kind, self.source, symbol, MASTER_SIZE)
        if size == MASTER_SIZE:
            return master.copy()
        import cv2

        mode = cv2.INTER_AREA if size < MASTER_SIZE else cv2.INTER_CUBIC
        return cv2.resize(master, (size, size), interpolation=mode)


#: Every piece is rasterised once at this size and scaled from there.  The
#: synthesiser asks for a different cell size on nearly every board, so caching
#: on the requested size caches nothing -- and a real SVG renderer is expensive
#: enough that it became the bottleneck in training rather than the network.
MASTER_SIZE = 128


@functools.lru_cache(maxsize=2048)
def _render_cached(name: str, kind: str, source: str, symbol: str, size: int) -> np.ndarray:
    if kind == "svg":
        return _render_builtin_svg(symbol, size)
    if kind == "font":
        return _render_font(source, symbol, size)
    if kind == "files":
        return _render_file(source, symbol, size)
    raise ValueError(f"unknown piece set kind: {kind!r}")


def _render_builtin_svg(symbol: str, size: int) -> np.ndarray:
    import chess
    import chess.svg

    return _rasterise_svg(chess.svg.piece(chess.Piece.from_symbol(symbol)).encode("utf-8"), size)


def _rasterise_svg(svg: bytes, size: int) -> np.ndarray:
    """Rasterise piece artwork, preferring the renderer that draws it correctly.

    PyMuPDF's SVG support silently drops gradient fills, and several figurine
    sets -- merida among them, which is one of the fonts printed books actually
    use -- paint the white pieces' bodies with a gradient.  Rendered through
    PyMuPDF those pieces come out as bare dark outlines, indistinguishable from
    the black ones, and a classifier trained on that learns to confuse the two
    colours.  Cairo draws them properly, so we use it when it is installed and
    fall back only when it is not.
    """
    try:
        import cairosvg
    except ImportError:
        return _rasterise_svg_pymupdf(svg, size)

    from PIL import Image

    scale = 2 if size >= MASTER_SIZE else 4
    png = cairosvg.svg2png(bytestring=svg, output_width=size * scale, output_height=size * scale)
    return _fit_canvas(np.array(Image.open(io.BytesIO(png)).convert("RGBA")), size)


def _rasterise_svg_pymupdf(svg: bytes, size: int) -> np.ndarray:
    import pymupdf

    doc = pymupdf.open(stream=svg, filetype="svg")
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


def _render_file(directory: str, symbol: str, size: int) -> np.ndarray:
    path = piece_file(Path(directory), symbol)
    if path is None:
        raise FileNotFoundError(f"no artwork for {symbol!r} in {directory}")
    if path.suffix.lower() == ".svg":
        return _rasterise_svg(path.read_bytes(), size)
    from PIL import Image

    return _fit_canvas(np.array(Image.open(path).convert("RGBA")), size)


def piece_file(directory: Path, symbol: str) -> Path | None:
    """The artwork file for one piece, under any of the usual naming habits."""
    colour = "w" if symbol.isupper() else "b"
    stems = (f"{colour}{symbol.upper()}", f"{colour}{symbol.upper()}".lower(),
             f"{colour}_{symbol.upper()}", f"{colour}{symbol.lower()}")
    for stem in stems:
        for suffix in (".svg", ".png"):
            path = directory / f"{stem}{suffix}"
            if path.exists():
                return path
    return None


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


def rejected_styles_in(directory: str | Path) -> dict[str, str]:
    """Style directories that were skipped, and why.

    Surfaced by ``dgc pieces`` so that a style quietly vanishing from training
    is something you can see rather than something you have to notice.
    """
    root = Path(directory)
    if not root.is_dir():
        return {}
    rejected: dict[str, str] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        missing = [s for s in SYMBOLS if piece_file(child, s) is None]
        if missing:
            if len(missing) < len(SYMBOLS):
                rejected[child.name] = f"missing {len(missing)} of 12 pieces"
            continue
        candidate = PieceSet(child.name, "files", str(child))
        if not fill_distinguishes_colours(candidate):
            rejected[child.name] = "white and black artwork render alike"
    return rejected


def piece_sets_in(directory: str | Path, prefix: str = "") -> list[PieceSet]:
    """Every complete style found in a directory of style subdirectories.

    A style counts only if all twelve pieces are there; a half-copied set would
    otherwise poison training with boards missing their bishops.
    """
    root = Path(directory)
    if not root.is_dir():
        return []
    sets: list[PieceSet] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not all(piece_file(child, symbol) is not None for symbol in SYMBOLS):
            continue
        candidate = PieceSet(f"{prefix}{child.name}", "files", str(child))
        if fill_distinguishes_colours(candidate):
            sets.append(candidate)
    return sets


def fill_distinguishes_colours(piece_set: PieceSet, tolerance: float = 25.0) -> bool:
    """Do this style's white pieces actually come out lighter than its black ones?

    This is a check on the *rasteriser*, not on the artwork's taste.  Several
    styles paint the white pieces' bodies with a gradient, and a renderer that
    quietly drops gradients hands back white pieces with nothing inside them.
    The result still looks like a chess set, so nothing errors; the classifier
    simply learns to confuse the two colours on every book afterwards, and the
    confusion matrix blames the model.

    It applies only to styles drawn from artwork files.  The font-based styles
    carry the distinction in the glyph itself -- an outlined king against a
    filled one -- and are drawn in one ink on purpose, so measuring their fill
    would reject them for doing the right thing.
    """
    if piece_set.kind == "font":
        return True
    for symbol in ("K", "Q", "P"):
        white = piece_set.render(symbol, 48).astype(np.float32)
        black = piece_set.render(symbol.lower(), 48).astype(np.float32)
        weights = np.array([0.299, 0.587, 0.114], np.float32)
        pair = []
        for art in (white, black):
            solid = art[:, :, 3] > 200
            if not solid.any():
                return False
            pair.append(float((art[:, :, :3] @ weights)[solid].mean()))
        if abs(pair[0] - pair[1]) >= tolerance:
            return True
    return False


def available_piece_sets(extra_dir: str | Path | None = None) -> list[PieceSet]:
    """Every piece style we can draw right now.

    ``extra_dir`` may hold a subdirectory per style, each with twelve SVG or PNG
    files named ``wK`` / ``bQ`` and so on.
    """
    sets: list[PieceSet] = [PieceSet("cburnett", "svg")]
    for name, path in _FONT_CANDIDATES:
        if Path(path).exists() and _font_has_pieces(path):
            sets.append(PieceSet(name, "font", path))
    sets.extend(piece_sets_in(extra_dir) if extra_dir else [])
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
