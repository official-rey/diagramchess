"""Training the piece classifier.

There are two situations this has to serve.  The first is the cold start, with
no verified data at all, where everything comes from synthetic diagrams and the
honest question is how well that transfers to a style the net has never seen.
The second is retraining after you have reviewed some diagrams, where a few
hundred real squares from your own books are mixed in and matter far more than
their number suggests.  Both run through :func:`train`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dataset import VerifiedSquares, fixed_set, make_batches
from .labels import LABELS, NUM_CLASSES
from .model import SQUARE_SIZE, Checkpoint, build_net, normalise, save_checkpoint
from .pieces import PieceSet, available_piece_sets, piece_sets_in


def training_styles(workspace) -> list[PieceSet]:
    """Everything a workspace can draw training diagrams in: the styles built
    into the package, plus any downloaded into it by ``dgc pieces --fetch`` or
    its button in the web interface.  Shared so the command line and the server
    train on the same material."""
    return available_piece_sets() + piece_sets_in(Path(workspace.root) / "pieces")


@dataclass
class TrainConfig:
    epochs: int = 8
    steps_per_epoch: int = 300
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    square_size: int = SQUARE_SIZE
    seed: int = 0
    workers: int = 3
    verified_fraction: float = 0.35
    holdout_style: str | None = None
    eval_squares: int = 4000
    piece_sets: list[PieceSet] = field(default_factory=list)


@dataclass
class TrainReport:
    checkpoint_path: Path
    metrics: dict
    history: list[dict]
    seconds: float
    trained_at: str = ""

    def describe(self) -> str:
        lines = [f"trained in {self.seconds:.0f}s -> {self.checkpoint_path}"]
        for key in sorted(self.metrics):
            value = self.metrics[key]
            lines.append(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        return "\n".join(lines)


def _batch_dataset(torch, config: TrainConfig, piece_sets, verified):
    class _Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            seed = config.seed + (info.id + 1) * 7919 if info else config.seed
            for x, y in make_batches(
                seed, config.batch_size, piece_sets, config.square_size,
                verified, config.verified_fraction,
            ):
                yield torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)

    return _Stream()


def _evaluate(torch, net, images: np.ndarray, labels: np.ndarray, batch_size: int = 512):
    """Accuracy and raw logits over a fixed evaluation set."""
    net.eval()
    logits: list = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = normalise(images[start:start + batch_size])
            logits.append(net(torch.from_numpy(chunk).unsqueeze(1)))
    all_logits = torch.cat(logits) if logits else torch.zeros(0, NUM_CLASSES)
    if not len(all_logits):
        return 0.0, all_logits
    predicted = all_logits.argmax(dim=1).numpy()
    return float((predicted == labels).mean()), all_logits


def _fit_temperature(torch, logits, labels) -> float:
    """Fit one scalar that makes the reported probabilities honest.

    The review queue is ordered by how unsure the model says it is, so an
    overconfident model does not just report the wrong number -- it puts the
    squares that need a human at the bottom of the list.
    """
    if not len(logits):
        return 1.0
    log_t = torch.zeros(1, requires_grad=True)
    targets = torch.from_numpy(labels)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        optimiser.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(np.clip(log_t.exp().item(), 0.25, 8.0))


def train(
    output: str | Path,
    config: TrainConfig | None = None,
    verified: VerifiedSquares | None = None,
    progress=None,
) -> TrainReport:
    """Train a classifier and write it to ``output``."""
    from .model import _torch

    torch = _torch()
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    started = time.time()

    all_sets = config.piece_sets or available_piece_sets()
    if not all_sets:
        raise RuntimeError("no piece artwork available to draw training diagrams with")
    train_sets = [s for s in all_sets if s.name != config.holdout_style]
    held_sets = [s for s in all_sets if s.name == config.holdout_style]
    if config.holdout_style and not held_sets:
        raise ValueError(f"no piece set named {config.holdout_style!r}")
    if not train_sets:
        raise ValueError("holding out that style leaves nothing to train on")

    verified_train, verified_val = (None, None)
    if verified is not None and len(verified):
        verified_train, verified_val = verified.split(0.25, seed=config.seed)

    net = build_net()
    optimiser = torch.optim.AdamW(net.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    total_steps = config.epochs * config.steps_per_epoch
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=config.learning_rate, total_steps=total_steps, pct_start=0.25,
    )

    loader = torch.utils.data.DataLoader(
        _batch_dataset(torch, config, train_sets, verified_train),
        batch_size=None, num_workers=config.workers,
        persistent_workers=config.workers > 0,
    )

    val_images, val_labels = fixed_set(config.seed + 101, config.eval_squares,
                                       train_sets, config.square_size)
    held_images = held_labels = None
    if held_sets:
        held_images, held_labels = fixed_set(config.seed + 202, config.eval_squares // 2,
                                             held_sets, config.square_size)

    history: list[dict] = []
    batches = iter(loader)
    for epoch in range(config.epochs):
        net.train()
        running, seen, correct = 0.0, 0, 0
        for _ in range(config.steps_per_epoch):
            x, y = next(batches)
            logits = net(x)
            loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=0.03)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            running += float(loss.detach()) * len(y)
            correct += int((logits.argmax(dim=1) == y).sum())
            seen += len(y)

        accuracy, _ = _evaluate(torch, net, val_images, val_labels)
        row = {
            "epoch": epoch + 1,
            "loss": running / max(1, seen),
            "train_accuracy": correct / max(1, seen),
            "val_accuracy": accuracy,
        }
        history.append(row)
        if progress:
            progress(row)

    val_accuracy, val_logits = _evaluate(torch, net, val_images, val_labels)
    temperature = _fit_temperature(torch, val_logits, val_labels)

    metrics: dict = {
        "val_accuracy": val_accuracy,
        "train_styles": ",".join(s.name for s in train_sets),
        "verified_squares": 0 if verified_train is None else len(verified_train),
    }
    if held_images is not None:
        metrics["heldout_style"] = config.holdout_style
        metrics["heldout_style_accuracy"], _ = _evaluate(torch, net, held_images, held_labels)
    if verified_val is not None and len(verified_val):
        metrics["verified_accuracy"], _ = _evaluate(torch, net, verified_val.images, verified_val.labels)
        metrics["verified_val_squares"] = len(verified_val)

    checkpoint = Checkpoint(
        state_dict={k: v.clone() for k, v in net.state_dict().items()},
        square_size=config.square_size,
        classes=LABELS,
        temperature=temperature,
        metrics=metrics,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=f"{config.epochs}x{config.steps_per_epoch} steps, batch {config.batch_size}",
    )
    path = save_checkpoint(checkpoint, output)
    return TrainReport(path, metrics, history, time.time() - started, checkpoint.trained_at)
