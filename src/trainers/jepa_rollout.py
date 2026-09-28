import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from trainers.utils import atari_to_gym

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


# -----------------------------
# TRAINING LOOP (ROLLOUT VERSION)
# -----------------------------
class RolloutActionJEPA(pl.LightningModule):
    def __init__(self, base_model, rollout_steps):
        super().__init__()

        self.model = base_model.model
        self.action_embed = base_model.action_embed
        self.latent_predictor = base_model.latent_predictor

        self.rollout_steps = rollout_steps

        # Freeze everything except predictor
        self.model.tubelet_embed.requires_grad_(False)
        self.model.student.requires_grad_(False)
        self.action_embed.requires_grad_(False)

    # -----------------------------
    # One-step transition
    # -----------------------------
    def predict_next(self, latent, action):
        with torch.no_grad():
            a = self.action_embed(atari_to_gym(action))

            if a.ndim > 2:
                a = a.mean(dim=tuple(range(1, a.ndim - 1)))

            a = a.unsqueeze(1)
            seq = torch.cat([latent, a], dim=1)

        out = self.latent_predictor(seq, seq)
        return out[:, :-1, :]

    # -----------------------------
    # Shared rollout/loss
    # -----------------------------
    def _shared_step(self, batch):
        img, actions, *_ = batch
        context_frames = config["context_frames"]

        # Encode initial context
        with torch.no_grad():
            context = img[:, :context_frames].unsqueeze(2)
            z = self.model.tubelet_embed(context)
            z = self.model.student(z)

        # Teacher rollout targets
        future = img[:, context_frames : context_frames + self.rollout_steps]

        targets = []

        with torch.no_grad():
            context_window = img[:, :context_frames]

            for t in range(self.rollout_steps):
                next_frame = future[:, t].unsqueeze(1)

                context_window = torch.cat(
                    [context_window[:, 1:], next_frame],
                    dim=1,
                )

                z_t = self.model.tubelet_embed(context_window.unsqueeze(2))
                z_t = self.model.student(z_t)

                targets.append(z_t)

        # Predictor rollout
        preds = []
        latent = z

        for t in range(self.rollout_steps):
            a_t = actions[:, context_frames + t]
            latent = self.predict_next(latent, a_t)
            preds.append(latent)

        loss = sum(F.smooth_l1_loss(p, g) for p, g in zip(preds, targets))
        loss /= self.rollout_steps

        return loss

    # -----------------------------
    # Training
    # -----------------------------
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)

        self.log(
            "train/rollout_dynamics",
            loss,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    # -----------------------------
    # Validation
    # -----------------------------
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)

        self.log(
            "val/rollout_dynamics",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    # -----------------------------
    # Optimizer
    # -----------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.latent_predictor.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )


# ============================================================
# Latent Rollout
# ============================================================


class LatentRollout(nn.Module):
    """
    Rolls the JEPA latent forward using the learned
    latent dynamics model.

    Produces the future latent after cycle_steps.
    """

    def __init__(
        self,
        latent_predictor,
        action_embed,
    ):
        super().__init__()

        self.latent_predictor = latent_predictor
        self.action_embed = action_embed

    def step(self, latent, action):

        with torch.no_grad():
            action = self.action_embed(action)

            if action.ndim > 2:
                action = action.mean(dim=tuple(range(1, action.ndim - 1)))

            action = action.unsqueeze(1)

            sequence = torch.cat(
                [
                    latent,
                    action,
                ],
                dim=1,
            )

        prediction = self.latent_predictor(sequence, sequence)

        return prediction[:, :-1]

    def forward(self, latent, actions):

        for action in actions.unbind(dim=1):
            latent = self.step(latent, action)

        return latent
