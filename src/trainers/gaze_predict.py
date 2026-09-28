import pytorch_lightning as pl
import torch
from torch import nn


class GazeTraining(pl.LightningModule):
    def __init__(self, hparams, net):
        super().__init__()
        self.model = net
        self.criterion = nn.KLDivLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y, _ = batch

        output = self.forward(x)
        loss = self.criterion(output, y)

        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=2.5 * 1e-4,
        )
        return {"optimizer": optimizer}
