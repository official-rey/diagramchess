import numpy as np
import pytest

from diagramchess.artwork import (
    BOOK_FONTS, LICENCES, SHIPPABLE_LICENCES, describe,
)
from diagramchess.pieces import (
    SYMBOLS, available_piece_sets, fill_distinguishes_colours, piece_file, piece_sets_in,
)

WEIGHTS = np.array([0.299, 0.587, 0.114], np.float32)


def _luminance(art):
    solid = art[:, :, 3] > 200
    if not solid.any():
        return None
    return float((art[:, :, :3].astype(np.float32) @ WEIGHTS)[solid].mean())


def test_built_in_styles_render_the_two_colours_differently():
    """However a style carries the distinction, the two colours must not render alike."""
    for piece_set in available_piece_sets():
        white = piece_set.render("K", 48)
        black = piece_set.render("k", 48)
        assert not np.array_equal(white, black), f"{piece_set.name} draws both kings alike"


def test_the_fill_check_exempts_the_font_styles():
    """They draw both colours in one ink on purpose, distinguishing by glyph."""
    for piece_set in available_piece_sets():
        assert fill_distinguishes_colours(piece_set), piece_set.name
        if piece_set.kind == "font":
            white = _luminance(piece_set.render("K", 48))
            black = _luminance(piece_set.render("k", 48))
            # Measuring their fill would reject them for doing the right thing.
            assert abs(white - black) < 25


def test_gradient_filled_artwork_survives_rasterising(tmp_path):
    """Reproduces the merida bug: a white body painted with a linear gradient.

    PyMuPDF renders this as an empty outline; a real SVG renderer fills it.
    """
    style = tmp_path / "gradientish"
    style.mkdir()
    outline = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50">'
               '<linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#fff"/>'
               '<stop offset="1" stop-color="#fff"/></linearGradient>'
               '<circle cx="25" cy="25" r="20" fill="url(#g)" stroke="#111" stroke-width="3"/></svg>')
    solid = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50">'
             '<circle cx="25" cy="25" r="20" fill="#111"/></svg>')
    for symbol in SYMBOLS:
        (style / f"{'w' if symbol.isupper() else 'b'}{symbol.upper()}.svg").write_text(
            outline if symbol.isupper() else solid
        )

    sets = piece_sets_in(tmp_path)
    assert [s.name for s in sets] == ["gradientish"], "the style was rejected outright"
    white = _luminance(sets[0].render("K", 48))
    black = _luminance(sets[0].render("k", 48))
    assert white is not None and black is not None
    assert white - black > 25, f"gradient fill was dropped: white {white}, black {black}"


def test_a_style_that_cannot_tell_the_colours_apart_is_left_out(tmp_path):
    style = tmp_path / "monochrome"
    style.mkdir()
    solid = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50">'
             '<circle cx="25" cy="25" r="20" fill="#111"/></svg>')
    for symbol in SYMBOLS:
        (style / f"{'w' if symbol.isupper() else 'b'}{symbol.upper()}.svg").write_text(solid)
    assert piece_sets_in(tmp_path) == []


def test_an_incomplete_style_is_left_out(tmp_path):
    """A half-copied set would train the model on boards missing their bishops."""
    style = tmp_path / "partial"
    style.mkdir()
    solid = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50"><circle cx="25" cy="25" r="20"/></svg>'
    for symbol in ("K", "Q", "k"):
        (style / f"{'w' if symbol.isupper() else 'b'}{symbol.upper()}.svg").write_text(solid)
    assert piece_sets_in(tmp_path) == []


def test_piece_files_are_found_under_the_usual_naming_habits(tmp_path):
    (tmp_path / "wK.svg").write_text("<svg/>")
    (tmp_path / "bq.png").write_bytes(b"")
    assert piece_file(tmp_path, "K").name == "wK.svg"
    assert piece_file(tmp_path, "q").name == "bq.png"
    assert piece_file(tmp_path, "N") is None


def test_nothing_is_reported_for_a_directory_that_is_not_there(tmp_path):
    assert piece_sets_in(tmp_path / "nope") == []


def test_licence_table_is_self_consistent():
    for name, licence in LICENCES.items():
        info = describe(name)
        assert info.licence == licence
        assert info.shippable == (licence in SHIPPABLE_LICENCES)
    assert describe("something-new").licence == "unstated"
    assert not describe("something-new").shippable


def test_the_book_fonts_are_kept_out_of_anything_shipped():
    """They are the honest held-out test set, so they must not leak into training."""
    for name in BOOK_FONTS:
        assert name in LICENCES
        assert not describe(name).shippable, f"{name} would end up in the shipped model"
