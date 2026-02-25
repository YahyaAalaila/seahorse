"""
PyTorch Lightning wrapper for UnifiedSTPP models.
"""
import torch
import pytorch_lightning as pl


class STPPLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-5, grad_clip=5.0):
        super().__init__()
        self.model = model  # UnifiedSTPP instance
        self.save_hyperparameters(ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def forward(self, batch):
        return self.model(
            times=batch["times"],
            locations=batch["locations"],
            lengths=batch["lengths"],
            marks=batch.get("marks"),
            x_event=batch.get("event_covariates"),
            x_field_at_events=batch.get("field_covariates"),
        )

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = output["nll"]
        self.log("train/nll", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/n_events", output["total_events"], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.log("val/nll", output["nll"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/n_events", output["total_events"], on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.log("test/nll", output["nll"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
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
