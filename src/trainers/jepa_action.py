import pytorch_lightning as pl
import torch
from torch import nn

# ================================================================
# Lightning training module
# ================================================================


class ActionHistoryTraining(pl.LightningModule):
    """
    Trains the action-conditioned JEPA classifier.

    Expected action packet:

        actions:
            [B, T]
        or:
            [B, T, 1]

    For a four-action sequence:

        actions[:, -4] = oldest conditioning action
        actions[:, -3] = conditioning action
        actions[:, -2] = conditioning action
        actions[:, -1] = action to predict

    Therefore:

        previous_actions = actions[:, -4:-1]

        target = actions[:, -1]
    """

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

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.learning_rate = float(
            hparams.get(
                "learning_rate",
                1e-3,
            )
        )

        self.weight_decay = float(
            hparams.get(
                "weight_decay",
                0.0,
            )
        )

        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_actions": hparams["num_actions"],
            }
        )

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        images: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(
            images,
            previous_actions,
        )

    # =========================================================
    # Shared train / validation / test logic
    # =========================================================

    def _shared_step(
        self,
        batch,
        stage: str,
    ) -> torch.Tensor:

        # -----------------------------------------------------
        # Existing dataset format
        #
        # images, gaze, actions
        # -----------------------------------------------------

        images, _, actions = batch

        images = images.float()

        # -----------------------------------------------------
        # Preserve compatibility with current dataset shape
        #
        # Example:
        #
        # [B,12,1,84,84]
        #
        # ->
        #
        # [B,12,84,84]
        # -----------------------------------------------------

        if images.ndim == 5:
            if images.shape[2] == 1:
                images = images.squeeze(2)

        if images.ndim != 4:
            raise ValueError(
                f"Expected classifier images [B,12,H,W], got {tuple(images.shape)}"
            )

        # -----------------------------------------------------
        # Actions
        # -----------------------------------------------------

        actions = torch.as_tensor(
            actions,
            device=images.device,
        )

        # Example:
        #
        # [B,T,1]
        #
        # ->
        #
        # [B,T]

        if actions.ndim >= 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        if actions.ndim != 2:
            raise ValueError(f"Expected actions [B,T], got {tuple(actions.shape)}")

        # -----------------------------------------------------
        # We require at least four actions.
        #
        # Three previous actions condition the model.
        #
        # Final action is the classification target.
        # -----------------------------------------------------

        if actions.shape[1] < 4:
            raise ValueError(
                "ActionHistoryTraining requires at least "
                "4 actions per packet: "
                "3 conditioning actions + 1 target action. "
                f"Got {tuple(actions.shape)}"
            )

        # -----------------------------------------------------
        # Previous three actions
        #
        # [B,T]
        #
        # ->
        #
        # [B,3]
        # -----------------------------------------------------

        previous_actions = actions[
            :,
            -4:-1,
        ].long()

        # -----------------------------------------------------
        # Current action target
        #
        # [B,T]
        #
        # ->
        #
        # [B]
        # -----------------------------------------------------

        targets = actions[
            :,
            -1,
        ].long()

        # -----------------------------------------------------
        # Debug first step
        # -----------------------------------------------------

        if self.global_step == 0 and stage == "train":
            print(
                "IMAGE SHAPE:",
                images.shape,
            )

            print(
                "FULL ACTION SHAPE:",
                actions.shape,
            )

            print(
                "PREVIOUS ACTION SHAPE:",
                previous_actions.shape,
            )

            print(
                "TARGET SHAPE:",
                targets.shape,
            )

            print(
                "PREVIOUS ACTION EXAMPLE:",
                previous_actions[0],
            )

            print(
                "TARGET EXAMPLE:",
                targets[0],
            )

            print(
                "UNIQUE TARGET ACTIONS:",
                torch.unique(
                    targets,
                    return_counts=True,
                ),
            )

        # -----------------------------------------------------
        # Network
        #
        # images:
        # [B,12,84,84]
        #
        # previous_actions:
        # [B,3]
        #
        # ->
        #
        # logits:
        # [B,1,18]
        # -----------------------------------------------------

        logits = self.net(
            images,
            previous_actions,
        )

        if logits.ndim != 3:
            raise RuntimeError(
                "Expected ActionNet output "
                "[B,1,num_actions], "
                f"got {tuple(logits.shape)}"
            )

        # -----------------------------------------------------
        # One prediction per packet
        #
        # [B,1,18]
        #
        # ->
        #
        # [B,18]
        # -----------------------------------------------------

        logits = logits[
            :,
            -1,
            :,
        ]

        # -----------------------------------------------------
        # Classification loss
        # -----------------------------------------------------

        loss = self.criterion(
            logits,
            targets,
        )

        # -----------------------------------------------------
        # Accuracy
        # -----------------------------------------------------

        predicted_actions = logits.argmax(dim=-1)

        accuracy = (predicted_actions == targets).float().mean()

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

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
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    # =========================================================
    # Lightning steps
    # =========================================================

    def training_step(
        self,
        batch,
        batch_idx,
    ):

        return self._shared_step(
            batch,
            "train",
        )

    def validation_step(
        self,
        batch,
        batch_idx,
    ):

        return self._shared_step(
            batch,
            "val",
        )

    def test_step(
        self,
        batch,
        batch_idx,
    ):

        return self._shared_step(
            batch,
            "test",
        )

    # =========================================================
    # Optimizer
    # =========================================================

    def configure_optimizers(
        self,
    ):

        # Only include trainable parameters.
        #
        # If tubelet_embed and student are frozen,
        # this trains:
        #
        #   action_history_embed
        #   action_head

        trainable_parameters = [
            parameter for parameter in self.net.parameters() if parameter.requires_grad
        ]

        return torch.optim.Adam(
            trainable_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    # =========================================================
    # Data loaders
    # =========================================================

    def train_dataloader(
        self,
    ):

        return self.data_loaders["train"]

    def val_dataloader(
        self,
    ):

        loader = self.data_loaders.get("val")

        if loader is None:
            loader = self.data_loaders.get("test")

        if loader is None:
            return []

        return loader

    def test_dataloader(
        self,
    ):

        loader = self.data_loaders.get("test")

        if loader is None:
            return []

        return loader
