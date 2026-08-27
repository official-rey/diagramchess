"""The piece classifier: a small convolutional net over single-square crops.

Reading a diagram is 64 independent 13-way classifications, and each one is an
easy problem -- a centred glyph on a plain ground, no clutter, no occlusion.
So the net is deliberately small.  What it has to be good at is *saying when it
is unsure*, because that is what drives the review queue, so the checkpoint
carries a fitted temperature and the probabilities it reports are calibrated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .labels import LABELS, NUM_CLASSES

SQUARE_SIZE = 48


def _torch():
    """Import torch on demand, with a message that says what to install.

    Detection, review and export all work without torch; only prediction and
    training need it, and a user who just wants to trace a diagram by hand
    should not have to install a deep learning stack first.
    """
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "this needs PyTorch: pip install 'diagramchess[ml]' "
            "(or pip install torch --index-url https://download.pytorch.org/whl/cpu)"
        ) from exc


def build_net(num_classes: int = NUM_CLASSES):
    """A small convolutional net over 48x48 grayscale squares.

    The first layer strides by two rather than working at full resolution.  A
    piece glyph is a large, high-contrast shape filling most of its crop, so
    nothing that matters lives at 48x48 detail -- and paying for it made
    training three and a half times slower on the CPU this is meant to run on.
    """
    torch = _torch()
    nn = torch.nn

    def block(in_ch: int, out_ch: int, layers: int = 2) -> list:
        mods: list = []
        for i in range(layers):
            mods += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        mods.append(nn.MaxPool2d(2))
        return mods

    return nn.Sequential(
        nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),  # 48 -> 24
        nn.BatchNorm2d(24),
        nn.ReLU(inplace=True),
        *block(24, 48),         # 24 -> 12
        *block(48, 96),         # 12 -> 6
        *block(96, 128, 1),     # 6 -> 3
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.15),
        nn.Linear(128, num_classes),
    )


def normalise(squares: np.ndarray) -> np.ndarray:
    """Standardise each crop on its own statistics.

    Books print at wildly different densities and scanners expose them
    differently, and none of that carries information about which piece is on
    the square.  Normalising per crop rather than per dataset throws it away
    before the net can learn to depend on it.
    """
    x = squares.astype(np.float32)
    if x.ndim == 2:
        x = x[None, ...]
    flat = x.reshape(len(x), -1)
    mean = flat.mean(axis=1)[:, None, None]
    std = flat.std(axis=1)[:, None, None]
    return (x - mean) / np.maximum(std, 1e-3)


@dataclass
class Checkpoint:
    """A trained model plus everything needed to interpret and reproduce it."""

    state_dict: dict
    square_size: int = SQUARE_SIZE
    classes: tuple[str, ...] = LABELS
    temperature: float = 1.0
    metrics: dict = field(default_factory=dict)
    trained_at: str = ""
    notes: str = ""


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> Path:
    torch = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": checkpoint.state_dict,
            "square_size": checkpoint.square_size,
            "classes": list(checkpoint.classes),
            "temperature": checkpoint.temperature,
            "metrics": checkpoint.metrics,
            "trained_at": checkpoint.trained_at or datetime.now(timezone.utc).isoformat(),
            "notes": checkpoint.notes,
        },
        path,
    )
    path.with_suffix(".json").write_text(json.dumps({
        "square_size": checkpoint.square_size,
        "classes": list(checkpoint.classes),
        "temperature": checkpoint.temperature,
        "metrics": checkpoint.metrics,
        "trained_at": checkpoint.trained_at,
        "notes": checkpoint.notes,
    }, indent=2))
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    torch = _torch()
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    return Checkpoint(
        state_dict=blob["state_dict"],
        square_size=int(blob.get("square_size", SQUARE_SIZE)),
        classes=tuple(blob.get("classes", LABELS)),
        temperature=float(blob.get("temperature", 1.0)),
        metrics=dict(blob.get("metrics", {})),
        trained_at=str(blob.get("trained_at", "")),
        notes=str(blob.get("notes", "")),
    )


def load_net(path: str | Path):
    """Load a checkpoint into an evaluation-ready net."""
    torch = _torch()
    checkpoint = load_checkpoint(path)
    net = build_net(len(checkpoint.classes))
    net.load_state_dict(checkpoint.state_dict)
    net.eval()
    torch.set_grad_enabled(False)
    return net, checkpoint
