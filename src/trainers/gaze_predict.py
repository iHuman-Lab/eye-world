import pytorch_lightning as pl
import torch
import torchvision
from torch import nn


class GazeTraining(pl.LightningModule):
    def __init__(self, hparams, net):
        super().__init__()
        self.model = net
        self.criterion = nn.KLDivLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        # grid = torchvision.utils.make_grid(output[0:10], normalize=True, nrow=5)
        # self.logger.experiment.add_image("predicted_train", grid, self.current_epoch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        output = self.forward(x)
        loss = self.criterion(output, y)

        self.log("test_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        # Log to tensorboard
        grid = torchvision.utils.make_grid(output[0:10], normalize=True, nrow=5)
        self.logger.experiment.add_image("predicted_test", grid, self.current_epoch)

        grid = torchvision.utils.make_grid(y[0:10], nrow=5)
        self.logger.experiment.add_image("ground_truth", grid, self.current_epoch)

        grid = torchvision.utils.make_grid(x[0:10], nrow=5)
        self.logger.experiment.add_image("input", grid, self.current_epoch)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=2.5 * 1e-4,
        )
        return {"optimizer": optimizer}
