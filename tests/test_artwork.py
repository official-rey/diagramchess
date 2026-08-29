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


def test_rejected_styles_say_why(tmp_path):
    from diagramchess.pieces import rejected_styles_in

    solid = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50">'
             '<circle cx="25" cy="25" r="20" fill="#111"/></svg>')

    partial = tmp_path / "partial"
    partial.mkdir()
    for symbol in ("K", "Q", "k"):
        (partial / f"{'w' if symbol.isupper() else 'b'}{symbol.upper()}.svg").write_text(solid)

    monochrome = tmp_path / "monochrome"
    monochrome.mkdir()
    for symbol in SYMBOLS:
        (monochrome / f"{'w' if symbol.isupper() else 'b'}{symbol.upper()}.svg").write_text(solid)

    rejected = rejected_styles_in(tmp_path)
    assert "missing" in rejected["partial"]
    assert "alike" in rejected["monochrome"]
    assert rejected_styles_in(tmp_path / "nowhere") == {}


def test_artwork_still_draws_when_native_cairo_is_missing(monkeypatch):
    """cairosvg binds to a native library, and on a machine without it the
    import succeeds and then raises OSError from inside cffi.  Catching only
    ImportError would leave such a machine unable to draw a piece at all --
    worse than falling back to the renderer that merely drops gradients.
    """
    import builtins

    import diagramchess.pieces as pieces

    real_import = builtins.__import__

    def without_cairo(name, *args, **kwargs):
        if name == "cairosvg":
            raise OSError('no library called "cairo-2" was found')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cairo)
    monkeypatch.setattr(pieces, "_CAIRO_WORKS", None)
    pieces._render_cached.cache_clear()

    assert pieces.cairo_available() is False
    sets = pieces.available_piece_sets()
    assert sets, "no styles usable at all without cairo"
    art = sets[0].render("N", 48)
    assert art.shape == (48, 48, 4)
    assert (art[:, :, 3] > 8).sum() > 100, "the fallback renderer drew nothing"


def test_cairo_is_working_in_this_environment():
    """Not a guarantee anywhere else -- a note for whoever reads a failure here."""
    from diagramchess.pieces import cairo_available

    assert cairo_available(), (
        "native Cairo is missing, so gradient-filled piece artwork will render "
        "as bare outlines; install libcairo2 (Linux) or cairo (macOS)"
    )
