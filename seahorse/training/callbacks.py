"""Training callbacks used by higher-level orchestration flows."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from seahorse.evaluation.likelihood import compute_next_event_test_nll
from seahorse.runner.runner import STPPRunner


CURVE_FIELDNAMES = [
    "suite",
    "config_id",
    "preset",
    "seed",
    "epoch",
    "train_progress_fraction",
    "train_progress_percent",
    "test_nll",
    "nll_kind",
    "nll_report_space",
    "test_nll_method",
    "test_nll_contexts",
    "test_nll_scored_contexts",
    "test_nll_missing_contexts",
    "wall_time_sec",
]


def _progress_milestones(n_epochs: int, curve_step: float) -> list[int]:
    if n_epochs <= 0:
        return []
    if curve_step <= 0.0 or curve_step > 1.0:
        raise ValueError(f"curve_step must lie in (0, 1], got {curve_step!r}")
    milestones: set[int] = set()
    n_steps = max(1, int(math.floor((1.0 / curve_step) + 1e-9)))
    for idx in range(1, n_steps + 1):
        frac = min(float(idx) * float(curve_step), 1.0)
        target_epoch = round(frac * n_epochs, 12)
        milestones.add(max(1, int(math.ceil(target_epoch))))
    milestones.add(int(n_epochs))
    return sorted(m for m in milestones if 1 <= m <= int(n_epochs))


class PeriodicTestNLLCallback(pl.Callback):
    """Evaluate benchmark-facing test NLL at fixed training-progress milestones."""

    def __init__(
        self,
        *,
        suite: str,
        config_id: str,
        preset: str,
        seed: int,
        curve_step: float,
        config,
    ):
        super().__init__()
        self.suite = str(suite)
        self.config_id = str(config_id)
        self.preset = str(preset)
        self.seed = int(seed)
        self.curve_step = float(curve_step)
        self.config = config
        self._run_dir: Path | None = None
        self._jsonl_path: Path | None = None
        self._csv_path: Path | None = None
        self._milestones: list[int] = []
        self._start_time: float | None = None

    def bind_run_context(self, *, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._jsonl_path = self._run_dir / "test_nll_curve.jsonl"
        self._csv_path = self._run_dir / "test_nll_curve.csv"

    def on_fit_start(self, trainer, pl_module) -> None:
        del trainer, pl_module
        self._milestones = _progress_milestones(
            int(self.config.training.n_epochs),
            self.curve_step,
        )
        self._start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        if epoch not in self._milestones:
            return

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None or getattr(datamodule._bundle, "test_dataset", None) is None:
            return
        test_dataset = datamodule._bundle.test_dataset
        seqs = list(getattr(test_dataset, "sequences", []) or [])
        if not seqs:
            return

        if self._run_dir is None or self._jsonl_path is None or self._csv_path is None:
            raise RuntimeError(
                "PeriodicTestNLLCallback requires bind_run_context(run_dir=...) before training."
            )

        eval_runner = STPPRunner(self.config)
        eval_runner._lightning_module = pl_module
        eval_runner._data_module = datamodule
        eval_runner._norm_stats = datamodule.get_norm_stats(self.config.data.normalize)

        was_training = bool(getattr(pl_module.model, "training", False))
        try:
            summary = compute_next_event_test_nll(
                eval_runner,
                seqs,
                device=pl_module.device,
                predictive_samples=self.config.training.predictive_test_nll_samples,
                seed=self.seed,
            )
        finally:
            if was_training:
                pl_module.model.train()
            else:
                pl_module.model.eval()

        row = {
            "suite": self.suite,
            "config_id": self.config_id,
            "preset": self.preset,
            "seed": self.seed,
            "epoch": epoch,
            "train_progress_fraction": float(epoch / max(int(self.config.training.n_epochs), 1)),
            "train_progress_percent": float(100.0 * epoch / max(int(self.config.training.n_epochs), 1)),
            "test_nll": float(summary["mean_nll"]),
            "nll_kind": str(summary["kind"]),
            "nll_report_space": str(summary["report_space"]),
            "test_nll_method": str(summary["method"]),
            "test_nll_contexts": int(summary["n_contexts"]),
            "test_nll_scored_contexts": int(summary["n_scored_contexts"]),
            "test_nll_missing_contexts": int(summary["n_missing_contexts"]),
            "wall_time_sec": float(
                0.0 if self._start_time is None else time.perf_counter() - self._start_time
            ),
        }
        self._append_row(row)

    def _append_row(self, row: dict[str, Any]) -> None:
        assert self._jsonl_path is not None
        assert self._csv_path is not None
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

        write_header = not self._csv_path.exists()
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CURVE_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


__all__ = ["CURVE_FIELDNAMES", "PeriodicTestNLLCallback"]
