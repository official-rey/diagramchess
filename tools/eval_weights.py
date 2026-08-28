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


def weights_for(scheme, net, bank, exemplar=None):
    coverage = len(bank.classes) / NUM_CLASSES
    depth = min(1.0, len(bank) / 640.0)          # ten verified diagrams
    doubt = 1.0 - net.max(axis=1)
    if scheme == "none":
        return np.zeros(len(net), np.float32)
    if scheme == "doubt-only":
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
    if scheme == "adaptive":
        # How far out of its depth is the model on *this* book?  When the two
        # readers disagree about a lot of the board, the model may be reading a
        # figurine style it does not know.  But disagreement is symmetric -- it
        # cannot say *which* of the two is wrong -- so this is only a guess.
        disagreement = float((net.argmax(axis=1) != exemplar.argmax(axis=1)).mean())
        lost = float(np.clip((disagreement - 0.06) / 0.16, 0.0, 1.0))
        base = 0.8 * coverage
        return (base * np.maximum(lost, doubt)).astype(np.float32)
    if scheme == "trackrecord":
        # The asymmetric version: how each reader has actually done on the
        # diagrams you already verified in this book, held out one at a time.
        # ``bank.trust`` is set by the caller from that comparison.
        base = 0.8 * coverage * getattr(bank, "trust", 0.0)
        return np.maximum(base, 0.8 * coverage * doubt).astype(np.float32)
    raise ValueError(scheme)


#: What to assume about the bank before there is enough evidence to measure it.
#: Not zero: on a book whose figurine style the model does not know, the bank is
#: better from the second verified diagram onwards, and starting at zero throws
#: away the diagrams where it would have helped most.
TRUST_PRIOR = 0.6
#: How much the prior is worth, in diagrams.  Measurement outweighs it quickly.
TRUST_PRIOR_WEIGHT = 2.0


def measure_trust(predictor, bank_squares, bank_labels, truth_labels):
    """How much better the bank has been than the model, on this book, in 0..1.

    Leave-one-out over the verified diagrams: read each one with a bank built
    from the others, and compare both readers against what the human said.
    The answer is pulled towards a prior while there is little to go on, so a
    single verified diagram does not decide it either way.
    """
    if len(bank_squares) < 2:
        return TRUST_PRIOR
    net_errors = bank_errors = total = 0
    for held in range(len(bank_squares)):
        rest = ExemplarBank()
        for other in range(len(bank_squares)):
            if other != held:
                rest.add(bank_squares[other], bank_labels[other])
        if not len(rest):
            continue
        squares, actual = bank_squares[held], truth_labels[held]
        net = predictor._net_probabilities(squares).argmax(axis=1)
        exemplar = rest.probabilities(squares).argmax(axis=1)
        net_errors += int((net != actual).sum())
        bank_errors += int((exemplar != actual).sum())
        total += len(actual)
    if not total:
        return TRUST_PRIOR
    gap = float(np.clip((net_errors - bank_errors) / max(net_errors, bank_errors, 1), 0.0, 1.0))
    seen = len(bank_squares)
    return (TRUST_PRIOR * TRUST_PRIOR_WEIGHT + gap * seen) / (TRUST_PRIOR_WEIGHT + seen)


def run(model, styles, pieces_dir, pages, verified_counts):
    sets = {s.name: s for s in piece_sets_in(pieces_dir)}
    predictor = Predictor(model)
    schemes = ("none", "doubt-only", "coverage", "adaptive", "trackrecord")
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
            per_diagram_squares, per_diagram_labels = [], []
            for did in ids[:n]:
                squares_i = load_squares(workspace, did)
                labels_i = np.array([LABEL_TO_INDEX[c] for c in truth[did]])
                bank.add(squares_i, labels_i)
                per_diagram_squares.append(squares_i)
                per_diagram_labels.append(labels_i)
            bank.trust = measure_trust(predictor, per_diagram_squares,
                                       per_diagram_labels, per_diagram_labels)
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
                        exemplar = bank.probabilities(squares)
                        w = weights_for(scheme, net, bank, exemplar)[:, None]
                        probs = (1 - w) * net + w * exemplar
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
