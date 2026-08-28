"""Train the shipped classifier on real figurine artwork.

Only styles under a licence compatible with this project are used, so the
weights can ship.  The classic freeware book fonts -- alpha, leipzig, companion,
chess7 -- are deliberately kept out, because they are the closest thing we have
to the fonts a real book is set in, and a held-out score against them is the
only cold-start number here worth quoting.
"""
import argparse, sys
sys.path.insert(0, "src")

from diagramchess.pieces import available_piece_sets, piece_sets_in
from diagramchess.train import TrainConfig, train

#: Licences from lichess-org/lila COPYING.md.  GPL/AGPL are compatible with this
#: project's GPL-3.0-or-later; MIT, Apache-2.0, CC0 and CC BY are freer still.
#: Everything non-commercial, no-derivatives or unstated is left out.
SHIPPABLE = {
    "cburnett": "GPLv2+", "merida": "GPLv2+",
    "letter": "AGPLv3+", "pirouetti": "AGPLv3+", "pixel": "AGPLv3+",
    "mpchess": "GPLv3+",
    "chessnut": "Apache-2.0",
    "fantasy": "MIT", "spatial": "MIT", "celtic": "MIT",
    "rhosgfx": "CC0-1.0",
    "kiwen-suwi": "CC BY 4.0", "firi": "CC BY 4.0",
    "totoy": "CC BY 4.0", "papercut": "CC BY 4.0",
    "shapes": "CC BY-SA 4.0",
}

#: Real book fonts, freeware rather than free, kept out of training on purpose.
HELD_OUT = ("alpha", "leipzig", "companion", "chess7")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--pieces", default="/home/user/lichess-org/lila/public/piece")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    found = {s.name: s for s in piece_sets_in(args.pieces)}
    sets = [found[name] for name in sorted(SHIPPABLE) if name in found]
    missing = sorted(set(SHIPPABLE) - set(found))
    sets += available_piece_sets()          # the three that ship with the tool

    print(f"training on {len(sets)} styles: {', '.join(s.name for s in sets)}")
    if missing:
        print(f"not found under {args.pieces}: {', '.join(missing)}")
    print(f"held out of training on purpose: {', '.join(HELD_OUT)}")

    config = TrainConfig(epochs=args.epochs, steps_per_epoch=args.steps,
                         workers=args.workers, piece_sets=sets, seed=7)
    report = train(args.output, config,
                   progress=lambda row: print(row, flush=True))
    print(report.describe())


if __name__ == "__main__":
    main()
