"""Guards on the review screen's stylesheet.

The screen exists so a human can tell what the model read from what the page
printed.  Two of its properties are therefore not decoration, and both were
got wrong once:

  - the two piece colours must be legible as *different*, in either theme;
  - the whole board must fit on screen without scrolling.

Neither is checked by rendering here -- that would need a browser -- so these
read the stylesheet for the specific mistakes that produced the bugs.
"""

import re

import pytest

from diagramchess.review import app as review_app

STATIC = review_app.STATIC
CSS = (STATIC / "style.css").read_text()


def _declarations(selector: str) -> str:
    """The body of the first rule with exactly this selector."""
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", CSS)
    assert match, f"no rule for {selector}"
    return match.group(1)


def _root_variable(name: str) -> str:
    match = re.search(rf"{re.escape(name)}\s*:\s*([^;]+);", CSS)
    assert match, f"{name} is not defined"
    return match.group(1).strip()


def _luminance(hex_colour: str) -> float:
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


@pytest.mark.parametrize("name", ["--piece-white", "--piece-black",
                                  "--piece-white-edge", "--piece-black-edge"])
def test_the_piece_colours_are_literal_and_not_borrowed_from_the_theme(name):
    """The bug this replaces: black pieces were filled with var(--ink).

    In dark mode --ink is near-white, so both colours rendered as the same
    pale blob and the one thing a reviewer is here to check -- which side a
    piece belongs to -- became invisible.  A piece's colour is data, so it
    must not be derived from anything the reader's theme can move.
    """
    value = _root_variable(name)
    assert re.fullmatch(r"#[0-9a-fA-F]{6}", value), f"{name} is {value!r}, not a literal colour"


def test_the_piece_colours_are_not_redefined_under_a_colour_scheme():
    dark_block = CSS.split("@media (prefers-color-scheme: dark)", 1)[1].split("\n}", 1)[0]
    for name in ("--piece-white", "--piece-black", "--sq-light", "--sq-dark"):
        assert name not in dark_block, f"{name} changes with the theme"


def test_white_and_black_pieces_stay_far_apart_in_both_fill_and_outline():
    white, black = _luminance(_root_variable("--piece-white")), _luminance(_root_variable("--piece-black"))
    assert white - black > 150, f"fills are only {white - black:.0f} apart"
    # Each is outlined against the other's fill, so a piece stays readable
    # wherever it lands: on its own square, on the crop, on the other colour.
    assert _luminance(_root_variable("--piece-white-edge")) < 100
    assert _luminance(_root_variable("--piece-black-edge")) > 180


def test_the_glyphs_use_those_variables():
    assert "var(--piece-white)" in _declarations(".cell .glyph.white")
    assert "var(--piece-black)" in _declarations(".cell .glyph.black")


def test_the_board_is_sized_to_fit_the_window():
    """It used to be width:100% of its column, so on a laptop the board ran
    off the bottom of the screen and the reviewer scrolled to see rank 1."""
    assert "overflow: hidden" in _declarations("body.app")
    stack = _declarations(".board-stack")
    assert "100cqh" in stack and "100cqw" in stack, (
        "the board must be capped by the column's height as well as its width"
    )


def test_the_review_screen_does_not_inherit_the_index_page_width_cap():
    """The shared `main` rule centres the index page in 1400px.  Left to apply
    here it took a third of the board's width away."""
    assert "max-width: none" in _declarations("main.review")


def test_every_static_file_the_pages_ask_for_exists():
    for page in ("index.html", "review.html"):
        for href in re.findall(r'(?:src|href)="/static/([^"]+)"', (STATIC / page).read_text()):
            assert (STATIC / href).exists(), f"{page} references a missing /static/{href}"


def test_the_review_page_has_the_elements_its_script_wires_up():
    html = (STATIC / "review.html").read_text()
    script = (STATIC / "review.js").read_text()
    for element_id in sorted(set(re.findall(r"\$\('#([a-z-]+)'\)", script))):
        assert f'id="{element_id}"' in html, f"review.js wires #{element_id}, which the page lacks"
