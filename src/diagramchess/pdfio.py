"""Everything that talks to the PDF itself: page images, embedded pictures, text."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
import numpy as np

DEFAULT_DPI = 200
_PDF_DPI = 72.0  # PDF user space is 72 units to the inch


@dataclass(frozen=True)
class PageRender:
    """A rasterised page plus the scale that maps PDF points to its pixels."""

    page_index: int
    image: np.ndarray  # grayscale uint8, (height, width)
    dpi: int

    @property
    def zoom(self) -> float:
        return self.dpi / _PDF_DPI

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def to_pixels(self, rect: "fitz.Rect") -> tuple[float, float, float, float]:
        z = self.zoom
        return (rect.x0 * z, rect.y0 * z, rect.x1 * z, rect.y1 * z)

    def to_points(self, box: tuple[float, float, float, float]) -> "fitz.Rect":
        z = self.zoom
        return fitz.Rect(box[0] / z, box[1] / z, box[2] / z, box[3] / z)


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file, used to recognise a book we have already ingested."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_pdf(path: str | Path) -> "fitz.Document":
    doc = fitz.open(str(path))
    if doc.is_encrypted and not doc.authenticate(""):
        raise ValueError(f"{path} is password protected")
    return doc


def render_page(doc: "fitz.Document", page_index: int, dpi: int = DEFAULT_DPI) -> PageRender:
    """Rasterise one page to a grayscale array.

    Grayscale is what both the detector and the classifier want, and it keeps
    a 600-page book's worth of page images at a size we can cache on disk.
    """
    page = doc[page_index]
    zoom = dpi / _PDF_DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return PageRender(page_index=page_index, image=image.copy(), dpi=dpi)


def embedded_image_boxes(doc: "fitz.Document", page_index: int, render: PageRender) -> list[tuple[float, float, float, float]]:
    """Pixel boxes of the raster images a page draws.

    Most chess books paste each diagram in as one picture, so these boxes are
    the cheapest and most exact board proposals we can get.  Books that draw
    their diagrams as vectors return nothing here and fall through to the
    image-based detector.
    """
    page = doc[page_index]
    boxes: list[tuple[float, float, float, float]] = []
    for info in page.get_images(full=True):
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            if rect.is_empty or rect.is_infinite:
                continue
            boxes.append(render.to_pixels(rect))
    return boxes


def vector_drawing_boxes(doc: "fitz.Document", page_index: int, render: PageRender) -> list[tuple[float, float, float, float]]:
    """Pixel boxes of filled rectangles a page draws with vector graphics.

    Diagram fonts and vector diagrams paint the dark squares as filled
    rectangles, so clusters of these give us proposals on books where
    :func:`embedded_image_boxes` finds nothing.  We return the bounding box of
    each cluster of similarly sized, aligned rectangles.
    """
    page = doc[page_index]
    squares: list[tuple[float, float, float, float]] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None or rect.is_empty or rect.is_infinite:
            continue
        w, h = rect.width, rect.height
        if w < 4 or h < 4 or w > 200 or h > 200:
            continue
        if abs(w - h) > 0.25 * max(w, h):
            continue
        squares.append(render.to_pixels(rect))
    return _cluster_boxes(squares)


def _cluster_boxes(boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Group boxes that touch or nearly touch into their bounding boxes."""
    if not boxes:
        return []
    remaining = list(boxes)
    clusters: list[list[tuple[float, float, float, float]]] = []
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            bx0 = min(b[0] for b in cluster) - 8
            by0 = min(b[1] for b in cluster) - 8
            bx1 = max(b[2] for b in cluster) + 8
            by1 = max(b[3] for b in cluster) + 8
            still: list[tuple[float, float, float, float]] = []
            for box in remaining:
                if box[0] < bx1 and box[2] > bx0 and box[1] < by1 and box[3] > by0:
                    cluster.append(box)
                    changed = True
                else:
                    still.append(box)
            remaining = still
        clusters.append(cluster)
    out = []
    for cluster in clusters:
        if len(cluster) < 8:  # a board shows far more filled squares than this
            continue
        out.append((
            min(b[0] for b in cluster), min(b[1] for b in cluster),
            max(b[2] for b in cluster), max(b[3] for b in cluster),
        ))
    return out


def text_near(
    doc: "fitz.Document",
    page_index: int,
    box: tuple[float, float, float, float],
    render: PageRender,
    margin_px: float = 60.0,
) -> str:
    """Text printed just around a diagram, where books put 'Black to play'."""
    page = doc[page_index]
    x0, y0, x1, y1 = box
    rect = render.to_points((x0 - margin_px, y0 - margin_px, x1 + margin_px, y1 + margin_px))
    rect = rect & page.rect
    if rect.is_empty:
        return ""
    return page.get_text("text", clip=rect).strip()
