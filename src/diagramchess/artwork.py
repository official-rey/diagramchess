"""Getting hold of more figurine styles.

Three piece styles ship with the tool, and three is not enough.  Measured on
books set in figurine fonts the classifier had never seen, a model trained on
those three read 92% of squares -- perfectly respectable, and still about five
corrections a diagram.  The single cheapest way to close that is more styles in
training, and there is a large collection of them, openly licensed, in Lichess's
source tree.

Nothing is downloaded unless you ask for it, and nothing is redistributed with
this project: the licences differ per style and several forbid commercial use,
so the choice of what to keep stays yours.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PIECE_REPO = "https://github.com/lichess-org/lila"
PIECE_PATH = "public/piece"

#: Licences as stated in lichess-org/lila's COPYING.md.  Styles absent from this
#: table are still usable for reading your own books; they are only excluded
#: from anything this project would redistribute.
LICENCES: dict[str, str] = {
    "cburnett": "GPLv2+", "merida": "GPLv2+", "mono": "GPLv2+",
    "letter": "AGPLv3+", "pirouetti": "AGPLv3+", "pixel": "AGPLv3+",
    "mpchess": "GPLv3+",
    "chessnut": "Apache-2.0",
    "fantasy": "MIT", "spatial": "MIT", "celtic": "MIT",
    "rhosgfx": "CC0-1.0",
    "kiwen-suwi": "CC BY 4.0", "firi": "CC BY 4.0",
    "totoy": "CC BY 4.0", "papercut": "CC BY 4.0",
    "shapes": "CC BY-SA 4.0",
    "alpha": "freeware, non-commercial", "leipzig": "freeware",
    "companion": "freeware", "chess7": "freeware",
    "horsey": "CC BY-NC-SA 4.0", "california": "CC BY-NC-SA 4.0",
    "caliente": "CC BY-NC-SA 4.0", "maestro": "CC BY-NC-SA 4.0",
    "fresca": "CC BY-NC-SA 4.0", "cardinal": "CC BY-NC-SA 4.0",
    "icpieces": "CC BY-NC-SA 4.0", "gioco": "CC BY-NC-SA 4.0",
    "tatiana": "CC BY-NC-SA 4.0", "staunty": "CC BY-NC-SA 4.0",
    "dubrovny": "CC BY-NC-SA 4.0", "anarcandy": "CC BY-NC-SA 4.0",
    "disguised": "CC BY-NC-SA 4.0", "cooke": "CC BY-NC-SA 4.0",
    "monarchy": "CC BY-NC-SA 4.0", "xkcd": "CC BY-NC-SA 2.5",
    "shahi-ivory-brown": "no derivatives",
}

#: Licences under which a model trained on the artwork can be redistributed
#: under this project's own terms.
SHIPPABLE_LICENCES = {
    "GPLv2+", "GPLv3+", "AGPLv3+", "Apache-2.0", "MIT",
    "CC0-1.0", "CC BY 4.0", "CC BY-SA 4.0",
}

#: The classic printed-book fonts.  Freeware rather than free, so they stay out
#: of anything shipped -- which makes them the honest held-out test set.
BOOK_FONTS = ("alpha", "leipzig", "companion", "chess7")


@dataclass
class StyleInfo:
    name: str
    licence: str
    shippable: bool
    source: str


def describe(name: str, source: str = "") -> StyleInfo:
    licence = LICENCES.get(name, "unstated")
    return StyleInfo(name, licence, licence in SHIPPABLE_LICENCES, source)


def fetch(destination: str | Path, quiet: bool = False) -> Path:
    """Download the Lichess piece styles into ``destination``.

    Only the artwork directory is checked out, not the whole repository, which
    keeps this to a few megabytes.
    """
    destination = Path(destination)
    if shutil.which("git") is None:
        raise RuntimeError("git is needed to fetch piece styles")

    work = destination.parent / (destination.name + ".checkout")
    if work.exists():
        shutil.rmtree(work)
    commands = [
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         PIECE_REPO, str(work)],
        ["git", "-C", str(work), "sparse-checkout", "set", PIECE_PATH],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError(f"{' '.join(command[:3])} failed: {result.stderr.strip()[:300]}")

    source = work / PIECE_PATH
    if not source.is_dir():
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"{PIECE_PATH} is not in the checkout")

    destination.mkdir(parents=True, exist_ok=True)
    for style in sorted(source.iterdir()):
        if style.is_dir():
            target = destination / style.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(style, target)
    shutil.rmtree(work, ignore_errors=True)
    return destination
