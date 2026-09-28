import pytorch_lightning as pl
import torch
from torch import nn

# ================================================================
# Lightning training module
# ================================================================


class ActionTraining(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        net: nn.Module,
        data_loader: dict,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        self.net = net
        self.data_loaders = data_loader
        # Optional [num_actions] inverse-frequency weights over the full
        # 18-way ALE action space (see compute_action_class_weights).
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.learning_rate = float(hparams.get("learning_rate", 1e-3))
        self.weight_decay = float(hparams.get("weight_decay", 0.0))

        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_actions": hparams["num_actions"],
            }
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, _, actions = batch

        images = images.float()

        if images.ndim == 5:
            images = images.squeeze(2)

        # actions: [batch, sequence_length, 1]
        actions = torch.as_tensor(actions)

        # Remove trailing singleton dimension if present:
        # [B, T, 1] -> [B, T]
        if actions.ndim >= 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        # Only use the LAST action in each packet.
        # [B, T] -> [B]
        if actions.ndim > 1:
            actions = actions[:, -1]

        # Targets are the raw ALE action IDs (0-17); no per-game remapping.
        targets = actions.long()

        if self.global_step == 0:
            print("ACTION SHAPE:", actions.shape)
            print("UNIQUE ACTIONS:", torch.unique(targets, return_counts=True))

        # net output: [B, 1, num_actions]
        logits = self.net(images)

        # We only make one prediction per packet:
        # [B, 1, num_actions] -> [B, num_actions]
        logits = logits[:, -1, :]

        loss = self.criterion(logits, targets)

        predicted_actions = logits.argmax(dim=-1)
        accuracy = (predicted_actions == targets).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    """def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, _, actions = batch

        images = images.float()

        if images.ndim == 5:
            images = images.squeeze(2)
        targets = atari_to_gym(torch.as_tensor(actions).squeeze(-1)).long()

        logits = self.net(images)

        loss = self.criterion(
            logits.reshape(-1, 9),
            targets.reshape(-1),
        )

        predicted_actions = logits.argmax(dim=2)
        accuracy = (predicted_actions == targets).float().mean()

        self.log(f"{stage}_loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=True)
        self.log(f"{stage}_accuracy", accuracy, on_epoch=True, prog_bar=True)

        return loss"""

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def train_dataloader(self):
        return self.data_loaders["train"]

    def val_dataloader(self):
        loader = self.data_loaders.get("val")

        if loader is None:
            # No val split available; fall back to the test split so
            # per-epoch checkpointing has a metric to monitor.
            loader = self.data_loaders.get("test")

        if loader is None:
            return []

        return loader

    def test_dataloader(self):
        loader = self.data_loaders.get("test")

        if loader is None:
            return []

        return loader
