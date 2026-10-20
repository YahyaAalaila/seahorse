"""Pinned-author TriTPP temporal-flow adapter.

TriTPP is a *temporal* TPP.  This module intentionally does not call the
combined model an author STPP: a campaign must supply and report its spatial
head separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy, time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from .provenance import TRITPP_PROVENANCE
from seahorse.models.spatial_models.gaussian_mixture import GaussianMixtureSpatialModel


def _author_ttpp():
    try:
        import ttpp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TriTPP requires the pinned author package: "
            "git+https://github.com/shchur/triangular-tpp@a384a50caf6f489335c6829707e2d7abddb44492"
        ) from exc
    return ttpp


@dataclass(frozen=True)
class TriTPPContract:
    """Fixed-horizon author-flow contract in the input time unit."""

    t_max: float
    spatial_head_name: str
    n_knots: int = 20
    spline_order: int = 2
    n_blocks: int = 4
    block_size: int = 16

    def validate(self) -> None:
        if not np.isfinite(self.t_max) or self.t_max <= 0:
            raise ValueError("TriTPP requires finite t_max > 0")
        if not self.spatial_head_name:
            raise ValueError("declare the separately fitted Seahorse spatial head")


class TriTPPAdapter:
    """Author-compatible temporal fit/score/sample wrapper.

    ``sequences`` contain only a per-window ``times`` array. All windows must
    share the same complete observation horizon [0, contract.t_max]. This is a
    launch gate, rather than an attempt to reinterpret variable horizons.
    """

    def __init__(self, contract: TriTPPContract, *, device: str | torch.device = "cpu"):
        contract.validate()
        self.contract = contract
        self.device = torch.device(device)
        self.model: Any | None = None
        self.spatial_model: GaussianMixtureSpatialModel | None = None
        self.loc_mean: torch.Tensor | None = None
        self.loc_std: torch.Tensor | None = None

    def _pack(self, sequences: Iterable[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [np.asarray(s["times"], dtype=np.float64).reshape(-1) for s in sequences]
        if not rows:
            raise ValueError("TriTPP needs at least one fixed-horizon sequence")
        for times in rows:
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("TriTPP author adapter requires finite strictly increasing times")
            # ttpp.data.SequenceDataset masks t=0 and pads with t_max.
            if np.any(times <= 0.0) or np.any(times >= self.contract.t_max):
                raise ValueError("author TriTPP contract requires 0 < event time < t_max")
        width = max(len(x) for x in rows) + 1
        values = np.full((len(rows), width, 1), self.contract.t_max, dtype=np.float32)
        mask = np.zeros_like(values)
        for i, times in enumerate(rows):
            values[i, : len(times), 0] = times
            mask[i, : len(times), 0] = 1.0
        return torch.as_tensor(values, device=self.device), torch.as_tensor(mask, device=self.device)

    def _pack_locations(self, sequences: Iterable[dict[str, Any]], mask: torch.Tensor) -> torch.Tensor:
        rows = [np.asarray(s["locations"], dtype=np.float32) for s in sequences]
        if len(rows) != mask.shape[0] or any(x.ndim != 2 or x.shape[0] != int(mask[i].sum()) for i, x in enumerate(rows)):
            raise ValueError("TriTPP spatial augmentation requires one finite location per event")
        if any(not np.isfinite(x).all() for x in rows):
            raise ValueError("TriTPP spatial augmentation requires finite locations")
        d = rows[0].shape[1]
        if any(x.shape[1] != d for x in rows): raise ValueError("inconsistent spatial dimensions")
        raw = np.concatenate(rows, axis=0)
        self.loc_mean = torch.as_tensor(raw.mean(axis=0), device=self.device)
        self.loc_std = torch.as_tensor(np.maximum(raw.std(axis=0), 1e-8), device=self.device)
        out = torch.zeros((len(rows), mask.shape[1], d), device=self.device)
        for i, row in enumerate(rows): out[i, :len(row)] = (torch.as_tensor(row, device=self.device) - self.loc_mean) / self.loc_std
        return out

    def fit(self, sequences: Iterable[dict[str, Any]], *, validation_sequences: Iterable[dict[str, Any]] | None = None, epochs: int = 1, lr: float = 1e-2, batch_size: int = 64, max_steps: int | None = None, max_time_seconds: float | None = None) -> dict[str, float]:
        """Fit the author temporal objective; spatial fitting is external."""
        sequences = list(sequences)
        validation_sequences = None if validation_sequences is None else list(validation_sequences)
        x, mask = self._pack(sequences)
        locations = self._pack_locations(sequences, mask)
        ttpp = _author_ttpp()
        mean_count = float(mask.sum(dim=(1, 2)).mean().item())
        self.model = ttpp.models.TriTPP(
            t_max=self.contract.t_max, lambda_init=mean_count,
            n_knots=self.contract.n_knots, spline_order=self.contract.spline_order,
            n_blocks=self.contract.n_blocks, block_size=self.contract.block_size,
        ).to(self.device)
        self.spatial_model = GaussianMixtureSpatialModel().to(self.device)
        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.spatial_model.parameters()), lr=lr)
        best=None; best_val=float("inf"); steps=0; started=time.monotonic(); budget_limited=False
        for _ in range(int(epochs)):
            for start in range(0,x.shape[0],max(1,int(batch_size))):
                if (max_steps is not None and steps >= max_steps) or (max_time_seconds is not None and time.monotonic()-started >= max_time_seconds):
                    budget_limited=True; break
                xb,mb,lb=x[start:start+batch_size],mask[start:start+batch_size],locations[start:start+batch_size]
                optimizer.zero_grad(); temporal=self.model.log_prob(xb,mb); spatial=(self.spatial_model.logprob(xb.squeeze(-1),lb,mb.squeeze(-1))*mb.squeeze(-1)).sum(-1)
                loss=-((temporal+spatial)/max(mean_count,1.)).mean(); loss.backward(); optimizer.step(); steps += 1
            if validation_sequences is not None:
                value=float(-self.log_prob(validation_sequences).mean())
                if np.isfinite(value) and value < best_val: best_val=value; best=(copy.deepcopy(self.model.state_dict()),copy.deepcopy(self.spatial_model.state_dict()))
            if budget_limited: break
        if best is None: raise ValueError("external budget expired before one finite validation checkpoint")
        if not steps: raise ValueError("external budget expired before an optimizer step and finite validation checkpoint")
        self.model.load_state_dict(best[0]); self.spatial_model.load_state_dict(best[1])
        return {"factorized_joint_nll_per_train_event": float(loss.detach().cpu()), "mean_train_events": mean_count, "best_validation_joint_nll": best_val, "optimizer_steps":steps, "elapsed_seconds":time.monotonic()-started, "budget_limited":budget_limited, "validation_checkpoint":"best_finite_validation"}

    def temporal_log_prob(self, sequences: Iterable[dict[str, Any]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit or load TriTPP before scoring")
        x, mask = self._pack(sequences)
        with torch.no_grad():
            return self.model.log_prob(x, mask).detach().cpu().numpy()

    def log_prob(self, sequences: Iterable[dict[str, Any]], *, raw_space: bool = True) -> np.ndarray:
        """Joint augmented log probability; applies the spatial z-score Jacobian."""
        if self.model is None or self.spatial_model is None or self.loc_mean is None or self.loc_std is None:
            raise RuntimeError("fit or load TriTPP spatial augmentation before scoring")
        sequences = list(sequences); x, mask = self._pack(sequences)
        # Do not refit transform at evaluation.
        rows=[np.asarray(s["locations"],dtype=np.float32) for s in sequences]; d=rows[0].shape[1]
        loc=torch.zeros((len(rows),mask.shape[1],d),device=self.device)
        for i,row in enumerate(rows): loc[i,:len(row)]=(torch.as_tensor(row,device=self.device)-self.loc_mean)/self.loc_std
        with torch.no_grad():
            answer=self.model.log_prob(x,mask)+(self.spatial_model.logprob(x.squeeze(-1),loc,mask.squeeze(-1))*mask.squeeze(-1)).sum(-1)
            if raw_space: answer=answer-mask.squeeze(-1).sum(-1)*torch.log(self.loc_std).sum()
            if raw_space:
                time_correction = torch.tensor([len(s["times"])*float(s.get("time_log_jacobian",0.)) for s in sequences], device=answer.device, dtype=answer.dtype)
                answer=answer-time_correction
        return answer.cpu().numpy()

    def sample_fixed_horizon(self, n: int, *, init_seq_len: int = 100) -> list[dict[str, np.ndarray]]:
        if self.model is None or self.spatial_model is None or self.loc_mean is None or self.loc_std is None:
            raise RuntimeError("fit or load TriTPP before sampling")
        with torch.no_grad():
            raw = self.model.rsample(int(n), device=str(self.device), init_seq_len=init_seq_len)
            times, mask = self.model.clip_times(raw)
        sigma0=torch.nn.functional.softplus(self.spatial_model.log_sigma_prior).detach()
        sigmak=torch.nn.functional.softplus(self.spatial_model.log_sigma_kernel).detach()
        tau=torch.nn.functional.softplus(self.spatial_model.log_tau).detach()
        result=[]
        for i in range(int(n)):
            ts=times[i, mask[i,:,0].bool(),0]; loc=[]
            for j,t in enumerate(ts):
                if j == 0: z=torch.randn_like(self.loc_mean)*sigma0
                else:
                    hist=torch.stack(loc); w=torch.softmax(-tau*(t-ts[:j]),dim=0); z=hist[torch.multinomial(w,1).item()]+torch.randn_like(self.loc_mean)*sigmak
                loc.append(z)
            z=torch.stack(loc) if loc else torch.empty((0,len(self.loc_mean)),device=self.device)
            result.append({"times":ts.cpu().numpy(),"locations":(z*self.loc_std+self.loc_mean).cpu().numpy()})
        return result

    def save(self, directory: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("cannot save an unfitted TriTPP")
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tritpp_temporal.pt"
        if self.spatial_model is None or self.loc_mean is None or self.loc_std is None: raise RuntimeError("spatial head missing")
        torch.save({"contract": self.contract.__dict__, "state_dict": self.model.state_dict(), "spatial_state_dict":self.spatial_model.state_dict(),"loc_mean":self.loc_mean,"loc_std":self.loc_std}, path)
        TRITPP_PROVENANCE.write(directory)
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "TriTPPAdapter":
        """Load a temporal-flow checkpoint produced by :meth:`save`."""
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        contract = TriTPPContract(**payload["contract"])
        adapter = cls(contract, device=device)
        ttpp = _author_ttpp()
        adapter.model = ttpp.models.TriTPP(
            t_max=contract.t_max, lambda_init=1.0,
            n_knots=contract.n_knots, spline_order=contract.spline_order,
            n_blocks=contract.n_blocks, block_size=contract.block_size,
        ).to(adapter.device)
        adapter.model.load_state_dict(payload["state_dict"])
        adapter.spatial_model = GaussianMixtureSpatialModel().to(adapter.device)
        adapter.spatial_model.load_state_dict(payload["spatial_state_dict"])
        adapter.loc_mean=payload["loc_mean"].to(adapter.device); adapter.loc_std=payload["loc_std"].to(adapter.device)
        adapter.model.eval()
        return adapter
