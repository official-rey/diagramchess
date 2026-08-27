import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def demo_pdf(tmp_path_factory):
    """A generated chess book with known contents, shared across the tests."""
    from diagramchess.demo import build_demo_book

    path = tmp_path_factory.mktemp("book") / "demo.pdf"
    build_demo_book(path, pages=6, seed=100, style_seed=200)
    return path


@pytest.fixture
def workspace(tmp_path):
    from diagramchess.store import Workspace

    return Workspace(tmp_path / "ws")
