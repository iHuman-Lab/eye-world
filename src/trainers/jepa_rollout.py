from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from trainers.utils import format_batch_for_vjepa


def load_encoder_weights(encoder, checkpoint_path):
    """
    Load the pretrained V-JEPA tubelet embedding and student into a
    VJEPAEncoder from a Lightning checkpoint of trainers.jepa.VJEPA.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint was not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)

    for name in ("tubelet_embed", "student"):
        prefix = f"model.{name}."
        weights = {
            k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
        }

        if not weights:
            raise KeyError(f"No '{prefix}*' weights in {checkpoint_path}")

        getattr(encoder, name).load_state_dict(weights)


def _normalize(x):
    """Feature-wise layer norm without affine parameters, as in V-JEPA 2."""
    return F.layer_norm(x, (x.size(-1),))


# -----------------------------
# TRAINING LOOP (ACTION-CONDITIONED WORLD MODEL)
# -----------------------------
class RolloutActionJEPA(pl.LightningModule):
    """
    Action-conditioned world model on top of a frozen V-JEPA encoder
    (V-JEPA 2-AC style).

    State at step j = latent of the context_frames-frame window ending at
    frame j + context_frames - 1, so every step has the same 16 tokens the
    encoder was pretrained on. The action that moves step j to step j + 1
    is the one recorded with the newest frame of window j.

    Each batch holds context_frames + ac_num_steps consecutive frames, so
    there are K + 1 windows (K = ac_num_steps) and K actions.

    Loss = teacher forcing (ground-truth latents in, next latents out)
         + auto-regressive rollout over ac_auto_steps (own predictions in).
    Targets and predictions are layer-normed and compared with L1.
    """

    def __init__(self, encoder, predictor, config):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.config = config

        self.context_frames = int(config["context_frames"])
        self.num_steps = int(config["ac_num_steps"])
        self.auto_steps = int(config["ac_auto_steps"])
        self.val_horizon = int(config["ac_val_horizon"])
        self.lr = float(config["ac_lr"])

        if not 1 <= self.auto_steps <= self.num_steps:
            raise ValueError("ac_auto_steps must be between 1 and ac_num_steps")

        if not 1 <= self.val_horizon <= self.num_steps:
            raise ValueError("ac_val_horizon must be between 1 and ac_num_steps")

        # Only the predictor trains.
        self.encoder.requires_grad_(False)

    # -----------------------------
    # Encoding
    # -----------------------------
    def _encode(self, batch):
        """
        Returns
            z: [B, K + 1, N, D] layer-normed latent of every window
            a: [B, K]           raw ALE action taken in windows 0..K-1
        """
        img, actions, _ = format_batch_for_vjepa(batch, self.config)

        ctx, K = self.context_frames, self.num_steps

        if img.shape[1] != ctx + K:
            raise ValueError(
                f"Expected {ctx + K} frames (context_frames + ac_num_steps), "
                f"got {img.shape[1]}. Build the loader with stack_length={ctx + K}."
            )

        # Sliding windows: [B, T, H, W] -> [B, K + 1, ctx, H, W]
        windows = img.unfold(1, ctx, 1).permute(0, 1, 4, 2, 3)
        B, W, _, H, Wd = windows.shape

        # The frozen encoder stays in eval mode even when Lightning calls train().
        self.encoder.eval()

        with torch.no_grad():
            z = self.encoder(windows.reshape(B * W, ctx, H, Wd))  # [B * W, N, D]
            z = _normalize(z).view(B, W, z.size(1), z.size(2))

        # TODO: verify this alignment against the dataset. It assumes the action
        # recorded with a frame is the one taken after seeing that frame, so it
        # leads to the next window. If it is recorded the other way round, use
        # actions[:, ctx - 2 : ctx - 2 + K].
        # The action recorded with the newest frame of window j.
        a = actions[:, ctx - 1 : ctx - 1 + K].long()

        return z, a

    # -----------------------------
    # Prediction
    # -----------------------------
    def _step(self, z, a):
        """Predict the next latent for every step of z, [B, S, N, D]."""
        return _normalize(self.predictor(z, a))

    def _rollout(self, z0, actions, steps):
        """
        Auto-regressive rollout from the ground-truth latent z0 [B, 1, N, D],
        feeding the model its own predictions. Returns [B, steps, N, D].
        """
        zs = z0

        for _ in range(steps):
            nxt = self._step(zs, actions[:, : zs.size(1)])[:, -1:]
            zs = torch.cat([zs, nxt], dim=1)

        return zs[:, 1:]

    def _teacher_forcing_loss(self, z, a):
        K = self.num_steps

        return F.l1_loss(self._step(z[:, :K], a), z[:, 1 : K + 1])

    # -----------------------------
    # Training
    # -----------------------------
    def training_step(self, batch, batch_idx):
        z, a = self._encode(batch)

        tf_loss = self._teacher_forcing_loss(z, a)

        rollout = self._rollout(z[:, :1], a, self.auto_steps)
        ar_loss = F.l1_loss(rollout, z[:, 1 : self.auto_steps + 1])

        loss = tf_loss + ar_loss

        self.log("train/tf_loss", tf_loss, on_epoch=True)
        self.log("train/ar_loss", ar_loss, on_epoch=True)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)

        return loss

    # -----------------------------
    # Validation
    # -----------------------------
    def validation_step(self, batch, batch_idx):
        z, a = self._encode(batch)
        H = self.val_horizon

        pred = self._rollout(z[:, :1], a, H)

        # Control: same rollout with the actions shuffled across the batch.
        # If the model uses the action, this must be worse than pred.
        shuffled = a[torch.randperm(a.size(0), device=a.device)]
        pred_shuffled = self._rollout(z[:, :1], shuffled, H)

        self.log("val/tf_loss", self._teacher_forcing_loss(z, a), sync_dist=True)
        self.log(
            "val/ar_loss",
            F.l1_loss(pred, z[:, 1 : H + 1]),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/shuffled_action_loss",
            F.l1_loss(pred_shuffled, z[:, 1 : H + 1]),
            sync_dist=True,
        )

        # Per horizon: the model against the "nothing changes" baseline.
        for h in range(1, H + 1):
            self.log(
                f"val/ar_loss_h{h}",
                F.l1_loss(pred[:, h - 1], z[:, h]),
                sync_dist=True,
            )
            self.log(
                f"val/copy_loss_h{h}",
                F.l1_loss(z[:, 0], z[:, h]),
                sync_dist=True,
            )

    # -----------------------------
    # Optimizer
    # -----------------------------
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.predictor.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )


# ============================================================
# Latent Rollout
# ============================================================


class LatentRollout(nn.Module):
    """
    Rolls an encoder latent forward with the learned dynamics model.

    latent:  [B, N, D] encoder output for the current state
    actions: [B, H]    raw ALE action ids to apply, H <= predictor.max_steps
    returns: [B, N, D] predicted (layer-normed) latent after the H actions
    """

    def __init__(self, predictor):
        super().__init__()

        self.predictor = predictor

    @torch.no_grad()
    def forward(self, latent, actions):
        zs = _normalize(latent).unsqueeze(1)  # [B, 1, N, D]

        for h in range(actions.size(1)):
            nxt = _normalize(self.predictor(zs, actions[:, : h + 1]))[:, -1:]
            zs = torch.cat([zs, nxt], dim=1)

        return zs[:, -1]
