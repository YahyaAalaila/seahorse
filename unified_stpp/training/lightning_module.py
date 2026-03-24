"""
PyTorch Lightning wrapper for UnifiedSTPP models.
"""
import math

import torch
import pytorch_lightning as pl

from unified_stpp.config.schema import TrainingConfig


class STPPLightningModule(pl.LightningModule):
    def __init__(self, model, tc: TrainingConfig):
        super().__init__()
        self.model = model  # UnifiedSTPP instance
        self.tc = tc
        self.save_hyperparameters({"tc": tc.model_dump()}, ignore=["model"])

    def forward(self, batch):
        return self.model(
            times=batch["times"],
            locations=batch["locations"],
            lengths=batch["lengths"],
            marks=batch.get("marks"),
            x_event=batch.get("event_covariates"),
            x_field_at_events=batch.get("field_covariates"),
        )

    @staticmethod
    def _batch_size_from_batch(batch) -> int:
        if isinstance(batch, dict) and "lengths" in batch and batch["lengths"] is not None:
            return int(batch["lengths"].shape[0])
        if isinstance(batch, dict) and "times" in batch and batch["times"] is not None:
            return int(batch["times"].shape[0])
        return 1

    def _log_state_regularization_terms(self, stage: str, output: dict, n_ev: int):
        terms = output.get("state_regularization_terms")
        if not isinstance(terms, dict):
            return
        for name, value in terms.items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            self.log(
                f"{stage}/state_reg/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=n_ev,
            )

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = output.get("loss", output["nll"])
        # Weight epoch average by event count so epoch NLL =
        #   Σ(nll_batch × n_events_batch) / Σ(n_events_batch)
        # i.e. the true per-event NLL over the full epoch, not a
        # sequence-count-weighted average of per-batch NLLs.
        n_ev  = max(1, int(output["total_events"].item()))
        n_seq = self._batch_size_from_batch(batch)

        if self.tc.vae_beta > 0 and "kl_loss" in output:
            kl = output["kl_loss"]
            loss = loss + self.tc.vae_beta * kl
            self.log("train/kl", kl, on_step=False, on_epoch=True, batch_size=n_ev)

        self.log("train/nll", output["nll"], on_step=False, on_epoch=True, prog_bar=True, batch_size=n_ev)
        self.log("train/n_events", output["total_events"], on_step=False, on_epoch=True, batch_size=n_seq)
        self._log_state_regularization_terms("train", output, n_ev)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        n_ev  = max(1, int(output["total_events"].item()))
        n_seq = self._batch_size_from_batch(batch)
        self.log("val/nll", output["nll"], on_step=False, on_epoch=True, prog_bar=True, batch_size=n_ev)
        self.log("val/n_events", output["total_events"], on_step=False, on_epoch=True, batch_size=n_seq)
        self._log_state_regularization_terms("val", output, n_ev)

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        n_ev = max(1, int(output["total_events"].item()))
        self.log("test/nll", output["nll"], on_step=False, on_epoch=True, batch_size=n_ev)
        self._log_state_regularization_terms("test", output, n_ev)

    def configure_optimizers(self):
        tc = self.tc
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
            betas=(tc.adam_beta1, tc.adam_beta2),
        )

        if tc.lr_schedule == "cosine":
            n_epochs = self.trainer.max_epochs
            warmup = tc.lr_warmup_epochs

            def _schedule(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / max(warmup, 1)
                progress = (epoch - warmup) / max(n_epochs - warmup, 1)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_schedule)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        if tc.lr_schedule == "step" or tc.lr_step_size is not None:
            if tc.lr_step_size is None:
                raise ValueError(
                    "lr_schedule='step' requires lr_step_size to be set in TrainingConfig."
                )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=tc.lr_step_size, gamma=tc.lr_step_gamma
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        if tc.lr_schedule == "constant":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda _: 1.0
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        if tc.lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/nll",
                    "interval": "epoch",
                },
            }

        raise ValueError(
            f"Unknown lr_schedule={tc.lr_schedule!r}. "
            "Valid options: 'constant', 'cosine', 'step', 'reduce_on_plateau'."
        )

    def on_before_optimizer_step(self, optimizer):
        if self.tc.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.tc.grad_clip
            )
