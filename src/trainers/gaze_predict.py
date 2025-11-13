import pytorch_lightning as pl
import torch
from torch import nn


class GazeTraining(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super().__init__()
        self.model = net
        self.data_loader = data_loader
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        loss = self.criterion(output, y)
        self.log("test_data", loss, on_epoch=True, on_step=False)
        return loss

    def train_dataloader(self):
        return self.data_loader["train"]

    def val_dataloader(self):
        return self.data_loader["test"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
        )
        return {"optimizer": optimizer}
