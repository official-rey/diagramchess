"""Choose how much say the exemplar bank gets, measured on realistic books.

The first version of this measurement ran against books drawn in the three
piece styles the classifier was trained on, where the classifier was already
right about everything -- so the only thing it could measure was how much
damage the bank did, and the answer was "least when it says least".

That was the wrong question.  This runs the same comparison on books set in
figurine fonts the classifier has never seen, where it is wrong about several
squares a diagram and, worse, confidently wrong.  There the bank has something
to contribute, and the weighting has to let it.
"""
import argparse, json, sys, tempfile
sys.path.insert(0, "src")
sys.path.insert(0, "tools")
from pathlib import Path

import numpy as np

import diagramchess.predict as P
from diagramchess.demo import build_demo_book
from diagramchess.labels import LABEL_TO_INDEX, LABELS, NUM_CLASSES
from diagramchess.pieces import piece_sets_in
from diagramchess.pipeline import ingest, load_squares
from diagramchess.predict import ExemplarBank, Predictor
from diagramchess.store import Workspace
from eval_loop import truth_for_diagrams


def weights_for(scheme, net, bank):
    coverage = len(bank.classes) / NUM_CLASSES
    depth = min(1.0, len(bank) / 640.0)          # ten verified diagrams
    doubt = 1.0 - net.max(axis=1)
    if scheme == "none":
        return np.zeros(len(net), np.float32)
    if scheme == "doubt-only":                    # what the code does today
        return (0.8 * coverage * doubt).astype(np.float32)
    if scheme == "coverage":
        return np.full(len(net), 0.8 * coverage, np.float32)
    if scheme == "depth":
        return np.full(len(net), 0.85 * coverage * depth, np.float32)
    if scheme == "depth+doubt":
        authority = 0.85 * coverage * depth
        return (authority + (1 - authority) * doubt).astype(np.float32)
    if scheme == "bank-only":
        return np.ones(len(net), np.float32) * (1.0 if len(bank.classes) >= 12 else 0.0)
    raise ValueError(scheme)


def run(model, styles, pieces_dir, pages, verified_counts):
    sets = {s.name: s for s in piece_sets_in(pieces_dir)}
    predictor = Predictor(model)
    schemes = ("none", "doubt-only", "coverage", "depth", "depth+doubt", "bank-only")
    totals = {s: [] for s in schemes}

    for style in styles:
        tmp = Path(tempfile.mkdtemp())
        pdf = tmp / "book.pdf"
        build_demo_book(pdf, pages=pages, seed=900, style_seed=3, piece_set=sets[style])
        meta = json.loads(pdf.with_suffix(".truth.json").read_text())
        workspace = Workspace(tmp / "ws")
        report = ingest(workspace, pdf, predictor=predictor)
        truth = truth_for_diagrams(workspace, report.book_id, pdf, meta)
        ids = sorted(truth)

        print(f"\n{style}: {len(ids)} diagrams")
        header = "  verified " + "".join(f"{s:>13}" for s in schemes)
        print(header)
        for n in verified_counts:
            if n >= len(ids):
                break
            bank = ExemplarBank()
            for did in ids[:n]:
                bank.add(load_squares(workspace, did),
                         np.array([LABEL_TO_INDEX[c] for c in truth[did]]))
            held = ids[n:]
            row = {}
            for scheme in schemes:
                errors = 0
                for did in held:
                    squares = load_squares(workspace, did)
                    net = predictor._net_probabilities(squares)
                    if scheme == "none":
                        probs = net
                    else:
                        w = weights_for(scheme, net, bank)[:, None]
                        probs = (1 - w) * net + w * bank.probabilities(squares)
                    labels = [LABELS[i] for i in probs.argmax(axis=1)]
                    errors += sum(1 for a, b in zip(labels, truth[did]) if a != b)
                row[scheme] = errors / len(held)
                totals[scheme].append(row[scheme])
            print(f"  {n:>8} " + "".join(f"{row[s]:>13.2f}" for s in schemes))
        workspace.close()

    print("\nmean errors per diagram, all styles and all bank sizes:")
    for scheme in schemes:
        print(f"  {scheme:<14}{np.mean(totals[scheme]):8.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--pieces", default="/home/user/lichess-org/lila/public/piece")
    parser.add_argument("--styles", default="letter,companion,chess7,staunty")
    parser.add_argument("--pages", type=int, default=14)
    args = parser.parse_args()
    run(args.model, args.styles.split(","), args.pieces, args.pages, (1, 2, 4, 8, 12))
