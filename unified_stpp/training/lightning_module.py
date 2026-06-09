"""
PyTorch Lightning wrapper for UnifiedSTPP models — Lightning glue only.

Responsibilities:
  - forward(): pass batch to model
  - *_step(): call model.compute_loss(), log metrics
  - configure_optimizers(): delegate entirely to TrainingConfig
  - on_before_optimizer_step(): gradient clipping

Everything else (optimizer construction, scheduler selection, loss extraction)
is owned by TrainingConfig or UnifiedSTPP respectively.
"""
import torch
import pytorch_lightning as pl

from unified_stpp.config.schema import TrainingConfig
from unified_stpp.runner.results import resolve_loss_result_reporting


class STPPLightningModule(pl.LightningModule):
    def __init__(self, model, tc: TrainingConfig):
        super().__init__()
        self.model = model  # UnifiedSTPP instance
        self.tc = tc
        self.save_hyperparameters({"tc": tc.model_dump()}, ignore=["model"])
        _caps = getattr(getattr(model, "event_model", None), "capabilities", None)
        self._train_key = getattr(_caps, "metric_key", "nll")

    @property
    def val_monitor_key(self) -> str:
        return f"val/{self._train_key}"

    def forward(self, batch):
        return self.model(
            times=batch["times"],
            locations=batch["locations"],
            lengths=batch["lengths"],
            marks=batch.get("marks"),
            x_event=batch.get("event_covariates"),
            x_field_at_events=batch.get("field_covariates"),
        )

    def eval_forward(self, batch):
        """Route through model.eval_forward() for test-time evaluation.

        For exact models, eval_forward delegates to training_loss (no-op).
        For SMASH and Diffusion, this runs the separate approximate NLL path.
        Validation and training steps are unaffected.
        """
        return self.model.eval_forward(
            times=batch["times"],
            locations=batch["locations"],
            lengths=batch["lengths"],
            marks=batch.get("marks"),
            x_event=batch.get("event_covariates"),
            x_field_at_events=batch.get("field_covariates"),
        )

    def _log_state_regularization_terms(self, stage: str, aux_terms: dict, n_ev: int):
        for name, value in aux_terms.items():
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

    def _log_extra_metrics(self, stage: str, extra_metrics: dict, n_ev: int):
        for name, value in (extra_metrics or {}).items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            if value.ndim != 0:
                continue
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=n_ev,
            )

    def _resolve_test_reporting(self, result) -> tuple[torch.Tensor, float | None, float | None, dict]:
        reported_nll, reported_temporal, reported_spatial, extra, _ = (
            resolve_loss_result_reporting(
                result,
                requested_space=self.tc.test_nll_space,
            )
        )
        return (
            torch.as_tensor(reported_nll, device=self.device, dtype=torch.float32),
            reported_temporal,
            reported_spatial,
            extra,
        )

    def training_step(self, batch, _batch_idx):
        result = self.model.compute_loss(self.forward(batch))
        n_ev = max(1, int(result.total_events.item()))
        loss = result.loss
        if result.kl is not None and self.tc.vae_beta > 0:
            loss = loss + self.tc.vae_beta * result.kl
            self.log("train/kl", result.kl, on_step=False, on_epoch=True, batch_size=n_ev)
        self.log(f"train/{self._train_key}", result.nll, on_step=False, on_epoch=True, prog_bar=True, batch_size=n_ev)
        self._log_state_regularization_terms("train", result.aux_terms, n_ev)
        return loss

    def validation_step(self, batch, _batch_idx):
        result = self.model.compute_loss(self.forward(batch))
        n_ev = max(1, int(result.total_events.item()))
        self.log(f"val/{self._train_key}", result.nll, on_step=False, on_epoch=True, prog_bar=True, batch_size=n_ev)
        if result.temporal_nll is not None:
            self.log("val/temporal_nll", result.temporal_nll, on_step=False, on_epoch=True, batch_size=n_ev)
        if result.spatial_nll is not None:
            self.log("val/spatial_nll", result.spatial_nll, on_step=False, on_epoch=True, batch_size=n_ev)
        self._log_extra_metrics("val", result.extra_metrics, n_ev)
        self._log_state_regularization_terms("val", result.aux_terms, n_ev)

    def test_step(self, batch, _batch_idx):
        result = self.model.compute_loss(self.eval_forward(batch))
        n_ev = max(1, int(result.total_events.item()))
        test_nll, temporal_nll, spatial_nll, extra_metrics = self._resolve_test_reporting(result)
        self.log("test/nll", test_nll, on_step=False, on_epoch=True, batch_size=n_ev)
        if temporal_nll is not None:
            self.log("test/temporal_nll", temporal_nll, on_step=False, on_epoch=True, batch_size=n_ev)
        if spatial_nll is not None:
            self.log("test/spatial_nll", spatial_nll, on_step=False, on_epoch=True, batch_size=n_ev)
        self._log_extra_metrics("test", extra_metrics, n_ev)
        self._log_state_regularization_terms("test", result.aux_terms, n_ev)

    def configure_optimizers(self):
        opt = self.tc.build_optimizer(self.model.parameters())
        sched = self.tc.build_lr_scheduler(opt, self._trainer, monitor_key=self.val_monitor_key)
        return {"optimizer": opt, "lr_scheduler": sched}

    def on_before_optimizer_step(self, _optimizer):
        if self.tc.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        projector = getattr(self.model, "project_parameters", None)
        if callable(projector):
            projector()
