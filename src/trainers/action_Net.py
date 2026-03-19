import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision


class ActionTraining(pl.LightningModule):
    def __init__(self, hparams, net, data_loader):
        super().__init__()

        self.model = net
        self.data_loader = data_loader

        # classification loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        stacked_imgs, stacked_gaze, stacked_actions = batch

        # reshape batch like in format_batch_for_vjepa
        B, CT, H, W = stacked_imgs.shape
        T = stacked_actions.shape[1]  # sequence length
        C = CT // T  # channels per frame

        # reshape: [B, C*T, H, W] → [B, T, C, H, W]
        x = stacked_imgs.view(B, T, C, H, W)

        # optionally convert grayscale
        if C == 1:
            x = x.squeeze(2)  # → [B, T, H, W]
        else:
            x = x.mean(dim=2)  # average over channels → [B, T, H, W]

        # last action as target
        y = stacked_actions[:, -1]

        logits = self(x)
        loss = self.criterion(logits, y)

        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        stacked_imgs, stacked_gaze, stacked_actions = batch

        # reshape batch
        B, CT, H, W = stacked_imgs.shape
        T = stacked_actions.shape[1]
        C = CT // T

        x = stacked_imgs.view(B, T, C, H, W)
        if C == 1:
            x = x.squeeze(2)
        else:
            x = x.mean(dim=2)

        y = stacked_actions[:, -1]

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, on_step=False, sync_dist=True)

        # log some inputs
        grid = torchvision.utils.make_grid(
            x[:, -1, :, :].unsqueeze(1)[0:10], normalize=True, nrow=5
        )
        self.logger.experiment.add_image("input_frames", grid, self.current_epoch)

        return loss

    def format_batch_for_vjepa(batch, config):
        stacked_imgs, stacked_gaze, stacked_actions = batch

        # -----------------------------
        # Get shapes
        # -----------------------------
        B, CT, H, W = stacked_imgs.shape
        T = stacked_actions.shape[1]  # sequence length
        C = CT // T  # channels per frame

        # -----------------------------
        # Reshape images: [B, C*T, H, W] → [B, T, C, H, W]
        # -----------------------------
        imgs = stacked_imgs.view(B, T, C, H, W)

        # If grayscale (C=1), squeeze channel dim
        if C == 1:
            imgs = imgs.squeeze(2)  # → [B, T, H, W]
        else:
            # If RGB, you may want to convert or keep as is
            # Option: average channels → grayscale
            imgs = imgs.mean(dim=2)  # → [B, T, H, W]

        # -----------------------------
        # Actions (already correct shape)
        # -----------------------------
        actions = stacked_actions  # [B, T]

        return imgs, actions

    def train_dataloader(self):
        return self.data_loader["train"]

    def val_dataloader(self):
        return self.data_loader["test"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=2.5e-4,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    num_actions = 18  # typical Atari action space


def training_step(self, batch, batch_idx):
    stacked_imgs, stacked_gaze, stacked_actions = batch

    # reshape batch like in format_batch_for_vjepa
    B, CT, H, W = stacked_imgs.shape
    T = stacked_actions.shape[1]  # sequence length
    C = CT // T  # channels per frame

    # reshape: [B, C*T, H, W] → [B, T, C, H, W]
    x = stacked_imgs.view(B, T, C, H, W)

    # optionally convert grayscale
    if C == 1:
        x = x.squeeze(2)  # → [B, T, H, W]
    else:
        x = x.mean(dim=2)  # average over channels → [B, T, H, W]

    # last action as target
    y = stacked_actions[:, -1]

    logits = self(x)
    loss = self.criterion(logits, y)

    self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
    return loss


def validation_step(self, batch, batch_idx):
    stacked_imgs, stacked_gaze, stacked_actions = batch

    # reshape batch
    B, CT, H, W = stacked_imgs.shape
    T = stacked_actions.shape[1]
    C = CT // T

    x = stacked_imgs.view(B, T, C, H, W)
    if C == 1:
        x = x.squeeze(2)
    else:
        x = x.mean(dim=2)

    y = stacked_actions[:, -1]

    logits = self(x)
    loss = self.criterion(logits, y)

    preds = torch.argmax(logits, dim=1)
    acc = (preds == y).float().mean()

    self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
    self.log("val_acc", acc, on_epoch=True, on_step=False, sync_dist=True)

    # log some inputs
    grid = torchvision.utils.make_grid(
        x[:, -1, :, :].unsqueeze(1)[0:10], normalize=True, nrow=5
    )
    self.logger.experiment.add_image("input_frames", grid, self.current_epoch)

    return loss


"""    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, on_step=False, sync_dist=True)

        # log some inputs
        grid = torchvision.utils.make_grid(x[0:10], normalize=True, nrow=5)
        self.logger.experiment.add_image("input_frames", grid, self.current_epoch)

        return loss"""
