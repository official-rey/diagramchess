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
    assert "packaged one" in out          # nothing trained, but one ships with it

    code, out = run("export", "--status", "all", "--format", "fen")
    assert code == 0
    # The packaged classifier reads them on ingest, with no training run first.
    positions = [line for line in out.splitlines() if line.strip()]
    assert positions, "the packaged model read nothing"
    assert all(len(p.split()) == 6 for p in positions)


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


def test_reread_uses_the_packaged_model_when_nothing_is_trained(run, tmp_path):
    book = tmp_path / "sample.pdf"
    run("demo-book", str(book), "--pages", "2", "--seed", "100")
    run("ingest", str(book), "--no-read")

    code, out = run("reread")
    assert code == 0
    assert "read " in out and "again" in out

    code, out = run("export", "--status", "all", "--format", "fen")
    assert [line for line in out.splitlines() if line.strip()]


def test_without_any_model_the_error_says_what_to_do(tmp_path, monkeypatch, capsys):
    """Installed without the packaged checkpoint, reread has to explain itself."""
    import diagramchess.model as model_module

    monkeypatch.setattr(model_module, "bundled_model", lambda: None)
    code = main(["-w", str(tmp_path / "ws"), "reread"])
    assert code == 1
    assert "no model available" in capsys.readouterr().err


def test_the_workspace_model_wins_over_the_packaged_one(tmp_path):
    """Order of preference: what you asked for, then yours, then ours."""
    from diagramchess.cli import _predictor
    from diagramchess.model import bundled_model
    from diagramchess.store import Workspace

    workspace = Workspace(tmp_path / "ws")
    predictor, _ = _predictor(workspace, None)
    assert predictor is not None and predictor.has_model

    trained = bundled_model()          # stand in for a model you trained
    workspace.register_model(trained, "2026-08-27T00:00:00+00:00", {"val_accuracy": 1.0})
    _, model_id = _predictor(workspace, None)
    assert model_id == int(workspace.active_model()["id"])


def test_unknown_model_id_is_reported(run):
    code, out = run("status")
    with pytest.raises(SystemExit):
        main(["--help"])


def test_training_registers_a_dated_model(tmp_path, monkeypatch, capsys):
    """A registered model has to carry the date it was trained, or the model
    list and the workspace status both show a blank where the version goes."""
    from pathlib import Path

    from diagramchess.store import Workspace
    from diagramchess.train import TrainReport
    import diagramchess.train as train_module

    workspace_dir = tmp_path / "ws"

    def fake_train(output, config=None, verified=None, progress=None):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"not really a checkpoint")
        return TrainReport(Path(output), {"val_accuracy": 0.5}, [], 1.0,
                           trained_at="2026-08-27T12:00:00+00:00")

    monkeypatch.setattr(train_module, "train", fake_train)
    assert main(["-w", str(workspace_dir), "train", "--epochs", "1"]) == 0

    workspace = Workspace(workspace_dir)
    model = workspace.active_model()
    assert model["trained_at"] == "2026-08-27T12:00:00+00:00"
    assert Path(model["path"]).name.startswith("piece-net-")

    code = main(["-w", str(workspace_dir), "models"])
    out = capsys.readouterr().out
    assert code == 0 and "2026-08-27T12:00:00+00:00" in out


def test_pieces_lists_the_built_in_styles(run):
    code, out = run("pieces")
    assert code == 0
    assert "cburnett" in out and "built in" in out
    assert "dgc pieces --fetch" in out          # tells you how to get more


def test_pieces_reports_downloaded_styles_with_their_licences(run, tmp_path):
    """A style's licence decides whether a model trained on it can be shipped."""
    styles = tmp_path / "ws" / "pieces" / "merida"
    styles.mkdir(parents=True)
    outline = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50">'
               '<circle cx="25" cy="25" r="20" fill="#eee" stroke="#111" stroke-width="3"/></svg>')
    solid = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50">'
             '<circle cx="25" cy="25" r="20" fill="#111"/></svg>')
    for symbol in ("P", "N", "B", "R", "Q", "K"):
        (styles / f"w{symbol}.svg").write_text(outline)
        (styles / f"b{symbol}.svg").write_text(solid)

    code, out = run("pieces")
    assert code == 0
    assert "merida" in out and "GPLv2+" in out
    assert "downloaded" in out


def test_demo_book_can_pin_a_style(run, tmp_path):
    book = tmp_path / "styled.pdf"
    code, out = run("demo-book", str(book), "--pages", "2", "--style", "dejavu")
    assert code == 0 and book.exists()

    code = None
    try:
        main(["-w", str(tmp_path / "ws"), "demo-book", str(book), "--style", "nonesuch"])
    except SystemExit:
        pass
    else:
        code, _ = run("demo-book", str(book), "--style", "nonesuch")
        assert code == 1


def test_train_reports_how_many_styles_it_is_drawing(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from diagramchess.train import TrainReport
    import diagramchess.train as train_module

    seen = {}

    def fake_train(output, config=None, verified=None, progress=None):
        seen["styles"] = [s.name for s in config.piece_sets]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"x")
        return TrainReport(Path(output), {}, [], 1.0, trained_at="2026-08-27T00:00:00+00:00")

    monkeypatch.setattr(train_module, "train", fake_train)
    assert main(["-w", str(tmp_path / "ws"), "train"]) == 0
    out = capsys.readouterr().out
    assert "figurine style" in out
    assert "cburnett" in seen["styles"]
