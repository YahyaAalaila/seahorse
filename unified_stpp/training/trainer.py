"""
Training loop for unified STPP models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import time
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float = 5.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Newer PyTorch versions removed/changed some scheduler kwargs (e.g. verbose).
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_nll = 0.0
        total_events = 0
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()
            output = self.model(
                times=batch["times"],
                locations=batch["locations"],
                lengths=batch["lengths"],
                marks=batch.get("marks"),
                x_event=batch.get("event_covariates"),
                x_field_at_events=batch.get("field_covariates"),
            )

            loss = output["nll"]
            if torch.isfinite(loss):
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_nll += loss.item() * output["total_events"].item()
                total_events += output["total_events"].item()
            n_batches += 1

        avg_nll = total_nll / max(total_events, 1)
        return {"nll": avg_nll, "n_events": total_events}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_nll = 0.0
        total_events = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            output = self.model(
                times=batch["times"],
                locations=batch["locations"],
                lengths=batch["lengths"],
                marks=batch.get("marks"),
                x_event=batch.get("event_covariates"),
                x_field_at_events=batch.get("field_covariates"),
            )

            if torch.isfinite(output["nll"]):
                total_nll += output["nll"].item() * output["total_events"].item()
                total_events += output["total_events"].item()

        avg_nll = total_nll / max(total_events, 1)
        return {"nll": avg_nll, "n_events": total_events}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_epochs: int = 100,
        log_every: int = 1,
    ) -> Dict[str, list]:
        history = {"train_nll": [], "val_nll": [], "epoch_time_sec": []}

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            elapsed = time.time() - t0
            history["train_nll"].append(train_metrics["nll"])
            history["epoch_time_sec"].append(elapsed)

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_nll"].append(val_metrics["nll"])
                self.scheduler.step(val_metrics["nll"])

            if epoch % log_every == 0 or epoch == 1:
                msg = f"[Epoch {epoch:3d}] train NLL: {train_metrics['nll']:.4f}"
                if val_metrics:
                    msg += f"  val NLL: {val_metrics['nll']:.4f}"
                msg += f"  ({elapsed:.1f}s)"
                print(msg)

        return history
