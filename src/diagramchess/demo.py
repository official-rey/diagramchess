"""Building a sample chess book, so the tool can be tried without one to hand.

The pages are laid out the way a real book lays them out -- running head, body
text, two diagrams to a page with captions -- and the true position of every
diagram is written alongside as JSON, which makes this the fixture the accuracy
tests measure against.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import pymupdf

from .board import BoardMatrix
from .pieces import available_piece_sets
from .render import random_style, render_diagram
from .synth import degrade, random_position

_BODY = (
    "White's plan is straightforward enough: complete development, castle short, "
    "and only then look for a break in the centre. Black's counterplay depends "
    "entirely on whether the queenside pawns can get moving before that happens."
)


@dataclass
class DemoDiagram:
    page: int
    fen: str
    caption: str
    box_pt: tuple[float, float, float, float]


def _draw_crosstable(page: "pymupdf.Page", rng: random.Random, top: float) -> None:
    """A tournament crosstable: the page furniture most likely to fool a board detector.

    It is a ruled grid of roughly square cells with marks in them, which is the
    detector's own description of a chessboard.  Every demo book gets one so the
    false-positive rate is measured against the hard case rather than the easy one.
    """
    rows, cols = rng.randint(7, 10), rng.randint(7, 10)
    cell = 26.0
    left = 60.0
    for r in range(rows + 1):
        y = top + r * cell
        page.draw_line(pymupdf.Point(left, y), pymupdf.Point(left + cols * cell, y),
                       color=(0.25, 0.25, 0.25), width=0.6)
    for c in range(cols + 1):
        x = left + c * cell
        page.draw_line(pymupdf.Point(x, top), pymupdf.Point(x, top + rows * cell),
                       color=(0.25, 0.25, 0.25), width=0.6)
    for r in range(rows):
        for c in range(cols):
            mark = rng.choice(["1", "0", "½", "-", "1", "0"])
            page.insert_text((left + c * cell + 9, top + r * cell + 17), mark, fontsize=8)


def build_demo_book(
    path: str | Path,
    pages: int = 6,
    seed: int = 11,
    style_seed: int | None = None,
    degrade_pages: bool = True,
    crosstables: bool = True,
    piece_set=None,
) -> list[DemoDiagram]:
    """Write a sample chess book to ``path`` and return its ground truth.

    One piece set and one diagram style are chosen for the whole book, because
    that is how books work -- and it is what makes per-book learning pay off.
    Pass ``piece_set`` to pin the figurine style, which is how the cross-style
    benchmark isolates one font at a time.
    """
    rng = random.Random(seed)
    style_rng = random.Random(seed if style_seed is None else style_seed)
    piece_set = piece_set or style_rng.choice(available_piece_sets())
    style = random_style(style_rng, piece_set, cell_px=44)
    style.coordinates = style_rng.random() < 0.5

    doc = pymupdf.open()
    truth: list[DemoDiagram] = []
    for page_index in range(pages):
        page = doc.new_page(width=396, height=612)  # a small-format chess book
        page.insert_text((54, 46), f"Chapter {page_index // 2 + 1}", fontsize=9, color=(0.35, 0.35, 0.35))
        page.insert_textbox(pymupdf.Rect(54, 60, 342, 120), _BODY, fontsize=9.5, align=pymupdf.TEXT_ALIGN_JUSTIFY)

        if crosstables and page_index % 3 == 2:
            page.insert_text((54, 150), "Table 1: final standings", fontsize=8.5)
            _draw_crosstable(page, rng, 160)
            page.insert_text((190, 590), str(page_index + 1), fontsize=8)
            continue

        for slot in range(2):
            board = random_position(rng)
            rendered = render_diagram(board, style)
            image = degrade(rendered.image, rng) if degrade_pages else rendered.image
            ok, buf = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError("could not encode a diagram")

            top = 140 + slot * 230
            size = 180.0
            rect = pymupdf.Rect(108, top, 108 + size, top + size)
            page.insert_image(rect, stream=buf.tobytes())

            side = "White" if rng.random() < 0.5 else "Black"
            caption = f"Diagram {page_index * 2 + slot + 1}: {side} to play"
            page.insert_text((108, top + size + 14), caption, fontsize=8.5)

            matrix = BoardMatrix(
                [list(r) for r in board.rows],
                side_to_move="w" if side == "White" else "b",
            )
            truth.append(DemoDiagram(
                page=page_index,
                fen=matrix.to_fen(),
                caption=caption,
                box_pt=(rect.x0, rect.y0, rect.x1, rect.y1),
            ))
        page.insert_text((190, 590), str(page_index + 1), fontsize=8)

    path = Path(path)
    doc.save(str(path))
    doc.close()
    meta = {
        "piece_set": piece_set.name,
        "style": {k: v for k, v in vars(style).items() if k != "piece_set"},
        "diagrams": [vars(d) for d in truth],
    }
    path.with_suffix(".truth.json").write_text(json.dumps(meta, indent=2, default=str))
    return truth
