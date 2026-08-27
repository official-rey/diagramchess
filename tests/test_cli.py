import json

import pytest

from diagramchess.cli import main


@pytest.fixture
def run(tmp_path, capsys):
    def invoke(*args, workspace=None):
        argv = ["-w", str(workspace or tmp_path / "ws"), *args]
        code = main(argv)
        return code, capsys.readouterr().out
    return invoke


def test_status_on_an_empty_workspace(run):
    code, out = run("status")
    assert code == 0
    assert "books: 0" in out
    assert "active model: none" in out


def test_demo_book_then_ingest_then_export(run, tmp_path):
    book = tmp_path / "sample.pdf"
    code, out = run("demo-book", str(book), "--pages", "4", "--seed", "100")
    assert code == 0 and book.exists()
    assert json.loads(book.with_suffix(".truth.json").read_text())["diagrams"]

    code, out = run("ingest", str(book))
    assert code == 0
    assert "diagrams detected" in out

    code, out = run("status")
    assert "diagrams: " in out and "verified: 0" in out

    code, out = run("export", "--status", "all", "--format", "fen")
    assert code == 0
    # Nothing has been read yet -- no model -- so there are no positions to print.
    assert out.strip() == ""


def test_export_formats_after_a_review(run, tmp_path):
    from diagramchess.board import BoardMatrix
    from diagramchess.store import Workspace

    workspace_dir = tmp_path / "ws"
    book = tmp_path / "sample.pdf"
    run("demo-book", str(book), "--pages", "2", "--seed", "100")
    run("ingest", str(book))

    workspace = Workspace(workspace_dir)
    row = workspace.diagrams()[0]
    fen = "8/8/4k3/8/2K5/8/6R1/8 w - - 0 1"
    labels = BoardMatrix.from_fen(fen).flat()
    workspace.save_review(int(row["id"]), labels, "white", "w", fen)
    workspace.close()

    code, out = run("export", "--format", "fen")
    assert code == 0 and fen in out

    code, out = run("export", "--format", "csv")
    assert "id,book,page,status,fen,lichess" in out
    assert "lichess.org/analysis" in out

    code, out = run("export", "--format", "pgn")
    assert '[SetUp "1"]' in out and f'[FEN "{fen}"]' in out

    code, out = run("export", "--format", "json")
    assert json.loads(out)

    code, out = run("accuracy")
    assert "nothing verified yet" in out or "measured against" in out


def test_reread_without_a_model_explains_itself(run, capsys):
    code = main(["-w", "/tmp/does-not-matter-ws", "reread"])
    assert code == 1
    assert "no model" in capsys.readouterr().err


def test_unknown_model_id_is_reported(run):
    code, out = run("status")
    with pytest.raises(SystemExit):
        main(["--help"])
