"""The ``dgc`` command line.

The intended run of play is: ``ingest`` a book, ``review`` the diagrams the tool
was least sure of, ``train`` on what you corrected, and ``reread`` the rest of
the book with the better model.  Each of those is one subcommand, and every one
of them works on a workspace directory that you can move, back up or delete.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

DEFAULT_WORKSPACE = ".diagramchess"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dgc",
        description="Read chess diagrams out of PDF books and open them on Lichess.",
    )
    parser.add_argument("--version", action="version", version=f"diagramchess {__version__}")
    parser.add_argument("-w", "--workspace", default=DEFAULT_WORKSPACE,
                        help="where diagrams, corrections and models are kept (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="find the diagrams in a PDF and read them")
    p.add_argument("pdf", nargs="+", type=Path)
    p.add_argument("--dpi", type=int, default=200, help="rendering resolution (default: %(default)s)")
    p.add_argument("--pages", help="page range to limit to, e.g. 10-40 (1-based, inclusive)")
    p.add_argument("--model", type=Path, help="checkpoint to read with; defaults to the active model")
    p.add_argument("--no-read", action="store_true", help="detect only, do not classify")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("review", help="serve the review UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", type=Path, help="checkpoint to re-read with; defaults to the active model")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("train", help="train the piece classifier")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--steps", type=int, default=300, help="batches per epoch")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=3, help="processes drawing synthetic diagrams")
    p.add_argument("--holdout-style", help="a piece style to keep out of training, to measure transfer")
    p.add_argument("--no-verified", action="store_true", help="train on synthetic data alone")
    p.add_argument("--output", type=Path, help="where to write the checkpoint")
    p.add_argument("--no-activate", action="store_true", help="register the model but do not make it active")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("reread", help="read stored diagrams again with the current model")
    p.add_argument("--book", type=int, help="limit to one book id")
    p.add_argument("--model", type=Path)
    p.add_argument("--include-verified", action="store_true",
                   help="also re-read diagrams you have already verified (their corrections are kept)")
    p.set_defaults(func=cmd_reread)

    p = sub.add_parser("export", help="print the positions found so far")
    p.add_argument("--book", type=int)
    p.add_argument("--status", default="verified",
                   choices=["verified", "pending", "rejected", "all"])
    p.add_argument("--format", default="text",
                   choices=["text", "board", "fen", "csv", "json", "pgn"])
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("accuracy", help="how often the model agrees with your corrections")
    p.add_argument("--book", type=int)
    p.set_defaults(func=cmd_accuracy)

    p = sub.add_parser("status", help="what is in the workspace")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("models", help="list trained models")
    p.add_argument("--activate", type=int, help="make this model id the active one")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("demo-book", help="write a sample chess book to try the tool on")
    p.add_argument("output", type=Path)
    p.add_argument("--pages", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_demo_book)

    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # a CLI should explain itself, not traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


# -- helpers -------------------------------------------------------------

def _workspace(args):
    from .store import Workspace
    return Workspace(args.workspace)


def _predictor(workspace, explicit: Path | None, required: bool = False):
    """Load the classifier the user asked for, or the workspace's active one."""
    from .predict import Predictor

    path = explicit
    if path is None:
        active = workspace.active_model()
        path = Path(active["path"]) if active else None
    if path is None:
        if required:
            raise RuntimeError("no model available; run 'dgc train' first")
        return None, None
    if not Path(path).exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    active = workspace.active_model()
    model_id = int(active["id"]) if active and str(active["path"]) == str(path) else None
    return Predictor(path), model_id


def _page_range(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start) - 1, int(end)))
    return [int(spec) - 1]


# -- commands ------------------------------------------------------------

def cmd_ingest(args) -> int:
    from .pipeline import ingest

    workspace = _workspace(args)
    predictor, model_id = (None, None)
    if not args.no_read:
        predictor, model_id = _predictor(workspace, args.model)
        if predictor is None:
            print("no model yet: detecting diagrams without reading them.\n"
                  "  run 'dgc train' to get one, then 'dgc reread'.", file=sys.stderr)

    for pdf in args.pdf:
        found = 0

        def progress(page_index: int, count: int) -> None:
            nonlocal found
            found += count
            print(f"\r  {pdf.name}: page {page_index + 1}, {found} diagram(s) so far",
                  end="\033[K", flush=True)

        report = ingest(workspace, pdf, dpi=args.dpi, pages=_page_range(args.pages),
                        predictor=predictor, model_id=model_id, progress=progress)
        print(f"\r{pdf.name}: {report.describe()}\033[K")
    return 0


def cmd_review(args) -> int:
    import uvicorn

    from .review.app import create_app

    workspace = _workspace(args)
    predictor, _ = _predictor(workspace, args.model)
    if predictor is None:
        print("no model loaded: you can still check and correct diagrams by hand,\n"
              "and 'read again' will work once you have verified one diagram in a book.",
              file=sys.stderr)
    app = create_app(workspace, predictor)
    print(f"review UI on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_train(args) -> int:
    from .train import TrainConfig, train

    workspace = _workspace(args)
    verified = None if args.no_verified else workspace.verified_squares()
    if verified is not None and len(verified):
        print(f"training with {len(verified)} verified squares from your books")
    else:
        print("training on synthetic diagrams only (nothing verified yet)")

    output = args.output or (Path(workspace.root) / "models" / f"piece-net-{_next_index(workspace)}.pt")
    config = TrainConfig(
        epochs=args.epochs, steps_per_epoch=args.steps, batch_size=args.batch_size,
        workers=args.workers, holdout_style=args.holdout_style,
    )

    def progress(row: dict) -> None:
        print(f"  epoch {row['epoch']:>2}  loss {row['loss']:.4f}  "
              f"train {row['train_accuracy']:.4f}  val {row['val_accuracy']:.4f}", flush=True)

    report = train(output, config, verified=verified, progress=progress)
    print(report.describe())
    model_id = workspace.register_model(
        report.checkpoint_path, report.metrics.get("trained_at", ""), report.metrics,
        notes=f"epochs={args.epochs} steps={args.steps}", activate=not args.no_activate,
    )
    print(f"registered as model #{model_id}" + ("" if args.no_activate else " and made active"))
    return 0


def _next_index(workspace) -> int:
    return len(workspace.models()) + 1


def cmd_reread(args) -> int:
    from .pipeline import repredict

    workspace = _workspace(args)
    predictor, model_id = _predictor(workspace, args.model, required=True)
    count = repredict(workspace, predictor, book_id=args.book, model_id=model_id,
                      include_verified=args.include_verified)
    print(f"read {count} diagram(s) again")
    return 0


def cmd_export(args) -> int:
    from .board import BoardMatrix

    workspace = _workspace(args)
    status = None if args.status == "all" else args.status
    rows = workspace.diagrams(book_id=args.book, status=status, order="page")
    if args.format == "json":
        print(json.dumps([dict(r) for r in rows], indent=2, default=str))
        return 0
    if args.format == "csv":
        print("id,book,page,status,fen,lichess")
        for row in rows:
            fen = row["fen"] or row["predicted_fen"] or ""
            url = BoardMatrix.from_fen(fen).lichess_url() if fen else ""
            print(f'{row["id"]},{row["book_id"]},{row["page"] + 1},{row["status"]},"{fen}",{url}')
        return 0
    if args.format == "fen":
        for row in rows:
            fen = row["fen"] or row["predicted_fen"]
            if fen:
                print(fen)
        return 0
    if args.format == "pgn":
        for row in rows:
            fen = row["fen"] or row["predicted_fen"]
            if not fen:
                continue
            caption = (row["caption"] or "").split("\n")[0].replace('"', "'")
            event = caption or f"diagram {row['id']}"
            print(f'[Event "{event}"]')
            print('[SetUp "1"]')
            print(f'[FEN "{fen}"]')
            print(f'[Annotator "diagramchess, page {row["page"] + 1}"]')
            print("*\n")
        return 0

    if args.format == "board":
        for row in rows:
            fen = row["fen"] or row["predicted_fen"]
            if not fen:
                continue
            board = BoardMatrix.from_fen(fen)
            caption = (row["caption"] or "").splitlines()
            print(f'#{row["id"]} page {row["page"] + 1}  {caption[0] if caption else ""}')
            print(board.ascii())
            print(f"  {board.lichess_url()}\n")
        return 0

    for row in rows:
        fen = row["fen"] or row["predicted_fen"] or "(not read)"
        confidence = row["min_confidence"]
        marker = "✓" if row["status"] == "verified" else " "
        print(f'{marker} #{row["id"]:<5} p{row["page"] + 1:<4} {fen}')
        if fen != "(not read)":
            print(f'      {BoardMatrix.from_fen(fen).lichess_url()}')
        if confidence is not None and row["status"] != "verified":
            first_line = (row["caption"] or "").splitlines()
            note = first_line[0] if first_line else ""
            print(f"      worst square {confidence * 100:.0f}% · {note}")
    return 0


def cmd_accuracy(args) -> int:
    from .accuracy import measure

    workspace = _workspace(args)
    report = measure(workspace, book_id=args.book)
    print(report.describe())
    for book_id, (perfect, total) in sorted(report.per_book.items()):
        book = workspace.book(book_id)
        title = book["title"] if book else f"book {book_id}"
        print(f"  {title}: {perfect}/{total} diagrams read perfectly")
    return 0


def cmd_status(args) -> int:
    workspace = _workspace(args)
    stats = workspace.stats()
    print(f"workspace: {workspace.root}")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ')}: {value}")
    active = workspace.active_model()
    if active:
        metrics = json.loads(active["metrics"])
        print(f"  active model: #{active['id']} {active['path']}")
        for key in sorted(metrics):
            print(f"      {key}: {metrics[key]}")
    else:
        print("  active model: none (run 'dgc train')")
    for book in workspace.books():
        print(f"  book {book['id']}: {book['title']} "
              f"({book['verified_count']}/{book['diagram_count']} verified)")
    return 0


def cmd_models(args) -> int:
    workspace = _workspace(args)
    if args.activate is not None:
        row = workspace.one("SELECT * FROM models WHERE id = ?", (args.activate,))
        if row is None:
            raise ValueError(f"no model #{args.activate}")
        with workspace.write() as db:
            db.execute("UPDATE models SET active = 0")
            db.execute("UPDATE models SET active = 1 WHERE id = ?", (args.activate,))
        print(f"model #{args.activate} is now active")
        return 0
    for row in workspace.models():
        metrics = json.loads(row["metrics"])
        flag = "*" if row["active"] else " "
        summary = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in sorted(metrics.items()))
        print(f"{flag} #{row['id']:<3} {row['trained_at']:<22} {summary}")
        print(f"    {row['path']}")
    return 0


def cmd_demo_book(args) -> int:
    from .demo import build_demo_book

    truth = build_demo_book(args.output, pages=args.pages, seed=args.seed)
    print(f"wrote {args.output} with {len(truth)} diagrams "
          f"(ground truth in {Path(args.output).with_suffix('.truth.json').name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
