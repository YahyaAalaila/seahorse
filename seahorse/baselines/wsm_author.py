"""Optional author-code adapter for WSM/AWSM plus its same-backbone MLE control.

The pinned repository trains WSM in ``model/SM_model.py`` (``wsm_loss`` plus
``get_surviving_loss``) and MLE via the same score model's ``get_lls`` path.
We deliberately invoke that implementation rather than reimplement its loss.
"""

from __future__ import annotations

import copy
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import numpy as np
import torch

from .provenance import WSM_PROVENANCE

Estimator = Literal["wsm", "wsm_survival", "mle"]


@dataclass(frozen=True)
class WSMContract:
    horizon: float
    spatial_lower: tuple[float, float]
    spatial_upper: tuple[float, float]
    backbone: Literal["SMASH", "SMASH_decouple", "SMASH_GMM"] = "SMASH_decouple"
    cond_dim: int = 48
    with_survival: bool = True
    grid_t: int = 5
    grid_s: int = 5
    eval_grid_t: int = 25
    eval_grid_s: int = 10
    num_units: int = 16
    n_layers: int = 1

    def validate(self):
        if self.horizon <= 0 or not np.isfinite(self.horizon):
            raise ValueError("WSM needs finite horizon")
        if any(a >= b for a, b in zip(self.spatial_lower, self.spatial_upper)):
            raise ValueError("WSM needs bounded spatial support")
        if not self.with_survival:
            raise ValueError("launch gate: WSM integration requires survival classification")


class WSMAuthorAdapter:
    """Materialize train-only normalized author input and launch matched estimators.

    The upstream ``run_copy.py`` recomputes normalization over train/val/test,
    so direct use is prohibited.  This adapter directly invokes the pinned
    author model/loss classes with routes and statistics built from training
    data only.
    """

    def __init__(self, contract: WSMContract, author_root: str | Path):
        contract.validate()
        self.contract = contract
        self.author_root = Path(author_root)

    def prepare(self, splits: dict[str, list[dict[str, Any]]], out: str | Path) -> Path:
        """Write author-format routes using the predeclared observation rectangle.

        ``splits`` may contain only train/validation during fitting.  Test data
        is intentionally absent from that route: it is read only by the
        explicit evaluate stage.  Empty windows are written as empty lists;
        the author model cannot currently batch them, so ``fit``/``score``
        raises a launch gate instead of discarding them.
        """
        if not {"train", "val"}.issubset(splits) or set(splits) - {"train", "val", "test"}:
            raise ValueError("require train and val; test is optional")
        train_nonempty = [
            np.asarray(s["locations"], float) for s in splits["train"] if len(s["locations"])
        ]
        if not train_nonempty:
            raise ValueError("WSM needs at least one nonempty training window")
        # The author's finite [0,1]^2 support must cover the declared source
        # rectangle. Train extrema would give held-out points positive density
        # outside the normalization domain used by its compensator.
        lo = np.asarray(self.contract.spatial_lower, float)
        hi = np.asarray(self.contract.spatial_upper, float)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for name, seqs in splits.items():
            rows = []
            for s in seqs:
                t = np.asarray(s["times"], float)
                x = np.asarray(s["locations"], float)
                if not len(t) and not x.size:
                    x = x.reshape(0, 2)
                if len(t) != len(x) or x.ndim != 2 or (len(x) and x.shape[1] != 2):
                    raise ValueError("WSM requires paired two-dimensional locations")
                if (
                    np.any(~np.isfinite(t))
                    or np.any(~np.isfinite(x))
                    or np.any(np.diff(t) <= 0)
                    or np.any(t <= 0)
                    or np.any(t >= self.contract.horizon)
                ):
                    raise ValueError("WSM requires strict finite events inside finite horizon")
                if np.any(x < lo) or np.any(x > hi):
                    raise ValueError("WSM event outside declared spatial support")
                z = (x - lo) / (hi - lo)
                rows.append([[float(a), *map(float, b)] for a, b in zip(t, z)])
            with (out / f"{name}.pkl").open("wb") as f:
                pickle.dump(rows, f)
        (out / "train_only_transform.npz").write_bytes(b"")
        np.savez(
            out / "train_only_transform.npz", loc_min=lo, loc_max=hi, horizon=self.contract.horizon
        )
        WSM_PROVENANCE.write(out)
        return out

    def command(
        self, prepared: str | Path, *, estimator: Estimator, seed: int, epochs: int
    ) -> list[str]:
        if estimator not in ("wsm", "wsm_survival", "mle"):
            raise ValueError("estimator must be wsm_survival or mle")
        author_estimator = "wsm" if estimator == "wsm_survival" else estimator
        return [
            "python",
            "run_copy.py",
            "--mode",
            "train",
            "--estimator",
            author_estimator,
            "--model",
            self.contract.backbone,
            "--seed",
            str(seed),
            "--total_epochs",
            str(epochs),
            "--with_survival",
            "1",
            "--cond_dim",
            str(self.contract.cond_dim),
            "--grid_t",
            str(self.contract.grid_t),
            "--grid_s",
            str(self.contract.grid_s),
            "--eval_grid_t",
            str(self.contract.eval_grid_t),
            "--eval_grid_s",
            str(self.contract.eval_grid_s),
        ]

    def _imports(self):
        root = str(self.author_root.resolve())
        if not (self.author_root / "model" / "SM_model.py").exists():
            raise FileNotFoundError("pinned WSM author checkout required")
        if root not in sys.path:
            sys.path.insert(0, root)
        from model.Models import Transformer_ST
        from model.SM_model import SMSTPP
        from model.intensity_model import smash_intensity_score

        return Transformer_ST, SMSTPP, smash_intensity_score

    def _batch(self, seqs):
        # Match author EventData/collate convention: origin, interarrival, lng, lat.
        if any(len(s["times"]) == 0 for s in seqs):
            raise ValueError(
                "launch gate: pinned WSM author model cannot batch empty observation windows; preserve them in prepared routes and use a model-specific finite-window treatment before inclusion"
            )
        m = max(len(s["times"]) for s in seqs)
        import torch

        o = torch.zeros((len(seqs), m))
        gap = torch.zeros_like(o)
        x = torch.zeros_like(o)
        y = torch.zeros_like(o)
        for i, s in enumerate(seqs):
            t = np.asarray(s["times"], float)
            z = np.asarray(s["locations"], float)
            o[i, : len(t)] = torch.tensor(np.r_[0.0, t[:-1]])
            gap[i, : len(t)] = torch.tensor(np.diff(np.r_[0.0, t]))
            x[i, : len(t)] = torch.tensor(z[:, 0])
            y[i, : len(t)] = torch.tensor(z[:, 1])
        return o, gap, x, y

    def _model(self, estimator):
        if estimator == "wsm_survival":
            estimator = "wsm"
        if estimator not in ("wsm", "mle"):
            raise ValueError("estimator must be wsm_survival or mle")
        Transformer_ST, SMSTPP, score_cls = self._imports()
        c = self.contract
        kind = {"SMASH": "vanilla", "SMASH_decouple": "decouple", "SMASH_GMM": "GMM"}[c.backbone]
        score = score_cls(
            c.num_units,
            c.cond_dim,
            T=c.horizon,
            S=[[0.0, 1.0], [0.0, 1.0]],
            K_trig=2,
            kernel_type=kind,
        )
        transformer = Transformer_ST(
            d_model=c.cond_dim,
            d_rnn=4 * c.cond_dim,
            d_inner=2 * c.cond_dim,
            n_layers=c.n_layers,
            n_head=1,
            d_k=c.cond_dim,
            d_v=c.cond_dim,
            dropout=0.0,
            device="cpu",
            loc_dim=2,
            CosSin=True,
            num_types=1,
        )
        return SMSTPP(
            score,
            c.horizon,
            [[0.0, 1.0], [0.0, 1.0]],
            estimator,
            transformer,
            "cpu",
            grid_t=c.grid_t,
            grid_s=c.grid_s,
            with_survival=True,
            eval_grid_t=c.eval_grid_t,
            eval_grid_s=c.eval_grid_s,
            intensity_grid_t=c.eval_grid_t,
            intensity_grid_s=c.eval_grid_s,
        )

    def fit(
        self,
        splits: dict[str, list[dict[str, Any]]],
        *,
        estimator: Estimator,
        epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 64,
        work_dir: str | Path | None = None,
        max_steps: int | None = None,
        max_time_seconds: float | None = None,
    ):
        """One or more author-objective steps with a predeclared spatial support transform.

        This bypasses upstream ``data_loader`` entirely; both estimators receive
        the same freshly built backbone and finite horizon.
        """
        import torch

        if set(splits) != {"train", "val"}:
            raise ValueError("fit accepts train and val only; use score for an explicit test stage")
        work = self.prepare(
            splits,
            work_dir if work_dir is not None else self.author_root / ".seahorse_wsm_prepared",
        )
        stats = np.load(work / "train_only_transform.npz")
        lo, hi = stats["loc_min"], stats["loc_max"]
        normalized = {}
        for name, seqs in splits.items():
            normalized[name] = [
                {**s, "locations": (np.asarray(s["locations"], float) - lo) / (hi - lo)}
                for s in seqs
            ]
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model = self._model(estimator)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        losses = []
        steps = 0
        started = time.monotonic()
        budget_limited = False
        best = None
        best_val = -float("inf")
        best_report = None
        for _ in range(int(epochs)):
            model.train()
            for start in range(0, len(normalized["train"]), batch_size):
                if (max_steps is not None and steps >= max_steps) or (
                    max_time_seconds is not None and time.monotonic() - started >= max_time_seconds
                ):
                    budget_limited = True
                    break
                loss, _ = model(
                    self._batch(normalized["train"][start : start + batch_size]), "train"
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
                steps += 1
            # A partial final epoch is still a valid candidate.  The held-out
            # route is fixed before fitting and never includes test windows.
            model.eval()
            report = self._score_with_transform(model, splits["val"], lo, hi)
            value = report["mean_loglik_per_window_normalized"]
            if not np.isfinite(value):
                raise ValueError(
                    "author WSM validation likelihood is nonfinite; no checkpoint accepted"
                )
            if value > best_val:
                best_val = value
                best = copy.deepcopy(model.state_dict())
                best_report = report
            if budget_limited:
                break
        if not steps:
            raise ValueError(
                "external budget expired before an optimizer step and finite validation checkpoint"
            )
        if best is None:
            raise ValueError("external budget expired before one finite validation checkpoint")
        model.load_state_dict(best)
        return model, {
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_loglik": best_val,
            "val_reporting": best_report,
            "transform": str(work / "train_only_transform.npz"),
            "optimizer_steps": steps,
            "elapsed_seconds": time.monotonic() - started,
            "budget_limited": budget_limited,
            "validation_checkpoint": "best_finite_validation",
        }

    def save(
        self, model, metrics: dict[str, Any], directory: str | Path, *, estimator: Estimator
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        transform = np.load(metrics["transform"])
        path = directory / "wsm_author.pt"
        torch.save(
            {
                "contract": self.contract.__dict__,
                "estimator": estimator,
                "state_dict": model.state_dict(),
                "loc_min": torch.as_tensor(transform["loc_min"]),
                "loc_max": torch.as_tensor(transform["loc_max"]),
                "author_root": str(self.author_root),
            },
            path,
        )
        WSM_PROVENANCE.write(directory)
        return path

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        adapter = cls(WSMContract(**payload["contract"]), payload["author_root"])
        model = adapter._model(payload["estimator"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return adapter, model, payload

    def _score_with_transform(
        self, model, sequences: list[dict[str, Any]], lo, hi
    ) -> dict[str, float | int | str]:
        """Return author normalized and source-coordinate log likelihoods.

        The pinned author's third return value is batch size, so its reported
        log likelihood is a *mean per fixed observation window*, including
        each window's survival term.  It is not a mean per event.
        """
        total = 0.0
        windows = events = 0
        time_log_jacobian = 0.0
        area = float(np.prod(np.asarray(hi, float) - np.asarray(lo, float)))
        if not area > 0:
            raise ValueError("WSM train-only spatial transform has nonpositive area")
        for start in range(0, len(sequences), 64):
            part = sequences[start : start + 64]
            if any(
                np.any(np.asarray(row["locations"], float) < lo)
                or np.any(np.asarray(row["locations"], float) > hi)
                for row in part
            ):
                raise ValueError("WSM scored event outside declared spatial support")
            normalized = [
                {**s, "locations": (np.asarray(s["locations"], float) - lo) / (hi - lo)}
                for s in part
            ]
            with torch.no_grad():
                t, s, b = model(self._batch(normalized), "test")
            total += float(t + s)
            windows += int(b)
            events += sum(len(row["times"]) for row in part)
            time_log_jacobian += sum(
                float(row.get("time_log_jacobian", 0.0)) * len(row["times"]) for row in part
            )
        if not windows:
            raise ValueError("WSM cannot score zero observation windows")
        normalized = total / windows
        raw = (total - events * math.log(area) - time_log_jacobian) / windows
        return {
            "mean_loglik_per_window_normalized": normalized,
            "mean_loglik_per_window_source": raw,
            "denominator": "observation_windows_including_survival",
            "windows": windows,
            "events": events,
            "spatial_log_jacobian_per_event": math.log(area),
            "time_log_jacobian_sum": time_log_jacobian,
        }

    def score(
        self, model, sequences: list[dict[str, Any]], payload: dict[str, Any]
    ) -> dict[str, float | int | str]:
        def _array(value):
            return (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value, float)
            )

        lo, hi = _array(payload["loc_min"]), _array(payload["loc_max"])
        return self._score_with_transform(model, sequences, lo, hi)

    def sample(self, *args, **kwargs):
        raise NotImplementedError(
            "author WSM_STPP exposes intensity and finite-window likelihood, but no author continuation sampler"
        )
