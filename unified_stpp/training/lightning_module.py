"""
PyTorch Lightning wrapper for UnifiedSTPP models.
"""
import torch
import pytorch_lightning as pl


class STPPLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=1e-3,
        weight_decay=1e-5,
        grad_clip=5.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_schedule="constant",
        lr_warmup_epochs=0,
        lr_step_size=None,
        lr_step_gamma=0.5,
        vae_beta=0.0,
    ):
        super().__init__()
        self.model = model  # UnifiedSTPP instance
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.lr_schedule = lr_schedule
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_step_size = lr_step_size
        self.lr_step_gamma = lr_step_gamma
        self.vae_beta = vae_beta

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

        if self.vae_beta > 0 and "kl_loss" in output:
            kl = output["kl_loss"]
            loss = loss + self.vae_beta * kl
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
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
        )

        if self.lr_schedule == "cosine":
            import math
            n_epochs = self.trainer.max_epochs
            warmup = self.lr_warmup_epochs

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

        if self.lr_step_size is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_step_size, gamma=self.lr_step_gamma
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        # Default: ReduceLROnPlateau (monitor val/nll)
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

    def on_before_optimizer_step(self, optimizer):
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
