import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import block_mask_tubelets_vectorized

ckpt_path = "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_world_model/version_2/checkpoints/epoch=49-step=138850.ckpt"

ckpt = torch.load(ckpt_path, map_location="cpu")
state_dict = ckpt["state_dict"]


class VJEPA(pl.LightningModule):
    def __init__(
        self,
        model,
        pred,
        config,
        mask_ratio=0.5,
        lr=1e-4,
        ema_decay=0.996,
        reg_coeff=0.1,
    ):
        super().__init__()
        self.model = model
        self.teacher = copy.deepcopy(model.student)
        self.pred = pred
        self.config = config
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay
        self.reg_coeff = reg_coeff

        for p in self.teacher.parameters():
            p.requires_grad = False

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.model.student.parameters(), self.teacher.parameters()):
                t.mul_(self.ema_decay).add_(s, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    def _rep_stats(self, x):
        """x: [B, N, D]"""
        return {
            "var": x.var(dim=(0, 1)).mean(),
            "norm": x.norm(dim=-1).mean(),
        }

    def training_step(self, batch, batch_idx):
        img, _ = batch  # [B, T, H, W]
        B = img.shape[0]

        # --------------------------------------------------
        # Tokenize once — shared by student and teacher
        # --------------------------------------------------
        x = img.unsqueeze(2)  # [B, T, 1, H, W]
        all_tokens = self.model.tubelet_embed(x)  # [B, N, D]
        N, D = all_tokens.shape[1], all_tokens.shape[2]

        # --------------------------------------------------
        # Block masking — uniform N_masked across batch
        # --------------------------------------------------
        _, mask_bool = block_mask_tubelets_vectorized(
            all_tokens,
            drop_ratio=self.mask_ratio,
            block_size=self.config.get("block_size", 2),
        )  # mask_bool: [B, N], True = masked

        if mask_bool.sum() == 0:
            return torch.tensor(0.0, device=img.device, requires_grad=True)

        # --------------------------------------------------
        # Teacher — sees all tokens, no grad
        # --------------------------------------------------
        with torch.no_grad():
            target_full = self.teacher(all_tokens)  # [B, N, D]

        # --------------------------------------------------
        # Student — sees only visible tokens
        # --------------------------------------------------
        N_visible = int((~mask_bool[0]).sum())
        visible_tokens = all_tokens[~mask_bool].reshape(B, N_visible, D)
        student_repr = self.model.student(visible_tokens)  # [B, N_visible, D]

        # --------------------------------------------------
        # Predictor
        # --------------------------------------------------
        N_masked = N - N_visible
        pos_embed = self.model.tubelet_embed.pos_embed.expand(B, -1, -1)
        masked_pos = pos_embed[mask_bool].reshape(B, N_masked, D)
        target_masked = target_full[mask_bool].reshape(B, N_masked, D)

        pred = self.pred(queries=masked_pos, context=student_repr)

        # --------------------------------------------------
        # Loss: smooth-L1 + variance regularization
        # --------------------------------------------------
        loss_jepa = F.smooth_l1_loss(pred, target_masked)
        pred_std = pred.std(dim=1)
        loss_reg = F.relu(1.0 - pred_std).mean()
        loss = loss_jepa + self.reg_coeff * loss_reg

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        pred_stats = self._rep_stats(pred)
        tgt_stats = self._rep_stats(target_masked)

        self.log("loss/jepa", loss_jepa, on_epoch=True, prog_bar=True)
        self.log("loss/reg", loss_reg, on_epoch=True)
        self.log("loss/total", loss, on_epoch=True, prog_bar=True)
        self.log("rep/pred_var", pred_stats["var"], on_epoch=True)
        self.log("rep/tgt_var", tgt_stats["var"], on_epoch=True)
        self.log("rep/pred_norm", pred_stats["norm"], on_epoch=True)
        self.log("rep/tgt_norm", tgt_stats["norm"], on_epoch=True)

        with torch.no_grad():
            n_params = sum(1 for _ in self.model.student.parameters())
            drift = sum(
                (s - t).pow(2).mean()
                for s, t in zip(
                    self.model.student.parameters(), self.teacher.parameters()
                )
            )
            self.log("ema/drift", drift / n_params, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.model.tubelet_embed.parameters())
            + list(self.model.student.parameters())
            + list(self.pred.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )


###########################################################################################################################
########################################################################################################################
##############################################################################################################################
'''
class ActionCondtionVJEPA(pl.LightningModule):
    def __init__(
        self,
        model,
        pred,
        config,
        mask_ratio=0.5,
        lr=1e-4,
        ema_decay=0.996,
        reg_coeff=0.1,
    ):
        super().__init__()
        self.model = model
        self.teacher = copy.deepcopy(model.student)
        self.pred = pred
        self.config = config
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay
        self.reg_coeff = reg_coeff

        for p in self.teacher.parameters():
            p.requires_grad = False
        

        student_weights = {
        k.replace("model.student.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.student.")
        }

        embed_weights = {
            k.replace("model.tubelet_embed.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.tubelet_embed.")
        }



        teacher_weights = {
        k.replace("teacher.", ""): v
        for k, v in state_dict.items()
        if k.startswith("teacher.")
        }

        ac_vjepa.model.student.load_state_dict(student_weights)

        ac_vjepa.model.tubelet_embed.load_state_dict(embed_weights)

        ac_vjepa.teacher.load_state_dict(teacher_weights)

        


    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    def _rep_stats(self, x):
        """x: [B, N, D]"""
        return {
            "var": x.var(dim=(0, 1)).mean(),
            "norm": x.norm(dim=-1).mean(),
        }

    def training_step(self, batch, batch_idx):
        img, actions = batch  # img: [B, T, H, W], actions: [B, T, A_dim]
        B, T, H, W = img.shape
        D = self.tubelet_embed.embedding_dim  # embedding dim

        # -------------------------------
        # Tubelet Embedding
        # -------------------------------
        x = img.unsqueeze(2)  # [B, T, 1, H, W]
        all_tokens = self.tubelet_embed(x)  # [B, N, D], N = tokens per frame * T

        # -------------------------------
        # Temporal Masking
        # -------------------------------
        # Student sees first 4 frames only
        first4_tokens = all_tokens[
            :, : 4 * all_tokens.size(1) // T, :
        ]  # [B, N_visible, D]

        # Last frame for teacher target
        last_tokens = all_tokens[:, -all_tokens.size(1) // T :, :]  # [B, N_target, D]

        # -------------------------------
        # Action Embedding & Conditioning
        # -------------------------------
        # Only actions corresponding to first 4 frames
        action_emb = self.action_embed(actions[:, :4, :])  # [B, 4, D]

        # Broadcast actions to match tokens per frame
        tokens_per_frame = first4_tokens.size(1) // 4
        action_broadcast = action_emb.unsqueeze(2).repeat(1, 1, tokens_per_frame, 1)
        action_broadcast = action_broadcast.reshape(B, -1, D)  # [B, N_visible, D]

        # Add actions to tokens
        student_input = first4_tokens + action_broadcast

        # -------------------------------
        # Student Forward Pass
        # -------------------------------
        student_repr = self.student(student_input)  # [B, N_visible, D]

        # -------------------------------
        # Teacher Forward Pass
        # -------------------------------
        with torch.no_grad():  # Teacher is frozen
            teacher_repr = self.teacher(last_tokens)  # [B, N_target, D]

        # -------------------------------
        # Loss
        # -------------------------------
        # Compare last frame prediction with teacher latent
        # Here we can do a simple MSE across all tokens
        # If student predicts multiple tokens per frame, reduce mean per frame
        student_pred_last_frame = student_repr[
            :, -tokens_per_frame:, :
        ]  # predict last frame tokens
        loss = F.mse_loss(student_pred_last_frame, teacher_repr)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.model.tubelet_embed.parameters())
            + list(self.model.student.parameters())
            + list(self.pred.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )

'''
'''

class ActionCondtionVJEPA(pl.LightningModule):
    def __init__(
        self,
        model,
        pred,
        config,
        ckpt_path,
        mask_ratio=0.5,
        lr=1e-4,
        ema_decay=0.996,
        reg_coeff=0.1,
    ):
        """
        Args:
            model: VJEPAEncoder containing student + tubelet_embed
            pred: fresh predictor (will be trained)
            config: dict, must contain "action_dim"
            ckpt_path: path to pretrained V-JEPA checkpoint
        """
        super().__init__()
        self.model = model
        self.pred = pred
        self.config = config
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay
        self.reg_coeff = reg_coeff

        # Teacher starts as copy of student
        self.teacher = copy.deepcopy(model.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Action embedding
        action_dim = config["action_dim"]
        self.action_embed = nn.Linear(
            action_dim, model.student.encoder.layers[0].self_attn.embed_dim
        )

        # ----------------------------
        # Load pretrained checkpoint
        # ----------------------------
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]

        # Load student weights
        student_weights = {
            k.replace("model.student.", ""): v
            for k, v in sd.items()
            if k.startswith("model.student.")
        }
        self.model.student.load_state_dict(student_weights)

        # Load tubelet embedding weights
        embed_weights = {
            k.replace("model.tubelet_embed.", ""): v
            for k, v in sd.items()
            if k.startswith("model.tubelet_embed.")
        }
        self.model.tubelet_embed.load_state_dict(embed_weights)

        # Teacher starts as EMA of student
        self.teacher.load_state_dict(self.model.student.state_dict())

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.model.student.parameters(), self.teacher.parameters()):
                t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer, optimizer_idx=None):
        self.update_teacher()

    def training_step(self, batch, batch_idx):
        img, actions = batch  # img: [B, T, H, W], actions: [B, T, A_dim]
        B, T, H, W = img.shape
        D = self.model.student.encoder.layers[0].self_attn.embed_dim

        # -------------------------------
        # Tubelet Embedding
        # -------------------------------
        x = img.unsqueeze(2)  # [B, T, 1, H, W]
        all_tokens = self.model.tubelet_embed(x)  # [B, N, D]
        tokens_per_frame = all_tokens.size(1) // T

        # -------------------------------
        # Temporal Masking
        # -------------------------------
        num_visible_frames = min(4, T)
        N_visible = num_visible_frames * tokens_per_frame
        first_tokens = all_tokens[:, :N_visible, :]  # Student sees first 4 frames
        last_tokens = all_tokens[:, -tokens_per_frame:, :]  # Teacher sees last frame

        # -------------------------------
        # Action Conditioning
        # -------------------------------
        action_emb = self.action_embed(actions[:, :num_visible_frames, :])  # [B, 4, D]
        action_broadcast = action_emb.unsqueeze(2).repeat(1, 1, tokens_per_frame, 1)
        action_broadcast = action_broadcast.reshape(B, N_visible, D)
        student_input = first_tokens + action_broadcast

        # -------------------------------
        # Forward Pass
        # -------------------------------
        student_repr = self.model.student(student_input)  # [B, N_visible, D]

        with torch.no_grad():
            teacher_repr = self.teacher(last_tokens)  # [B, tokens_per_frame, D]

        student_pred_last_frame = self.pred(student_repr[:, -tokens_per_frame:, :])

        loss = F.mse_loss(student_pred_last_frame, teacher_repr)

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.model.tubelet_embed.parameters())
            + list(self.model.student.parameters())
            + list(self.pred.parameters())
            + list(self.action_embed.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
'''


class ActionConditionVJEPA(pl.LightningModule):
    def __init__(
        self,
        model,
        config,
        ckpt_path,
        latent_pred_dim=None,
        num_visible_frames=4,
        lr=1e-4,
        ema_decay=0.996,
    ):
        """
        Args:
            model: VJEPAEncoder containing student + tubelet_embed
            config: dict, must contain "action_dim"
            ckpt_path: path to pretrained V-JEPA checkpoint
            latent_pred_dim: optional dim for latent predictor
            num_visible_frames: how many frames student sees
        """
        super().__init__()
        self.model = model
        self.config = config
        self.lr = lr
        self.ema_decay = ema_decay
        self.num_visible_frames = num_visible_frames

        # Teacher starts as copy of student
        self.teacher = copy.deepcopy(model.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Action embedding
        action_dim = config["action_dim"]
        self.action_embed = nn.Linear(
            action_dim, model.student.encoder.layers[0].self_attn.embed_dim
        )

        # Autoregressive latent predictor (causal transformer)
        D = model.student.encoder.layers[0].self_attn.embed_dim
        latent_pred_dim = latent_pred_dim or D
        self.latent_predictor = nn.Transformer(
            d_model=D,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=2048,
            batch_first=True,
        )

        # ----------------------------
        # Load pretrained checkpoint
        # ----------------------------
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]

        # Load student weights
        student_weights = {
            k.replace("model.student.", ""): v
            for k, v in sd.items()
            if k.startswith("model.student.")
        }
        self.model.student.load_state_dict(student_weights)

        # Load tubelet embedding weights
        embed_weights = {
            k.replace("model.tubelet_embed.", ""): v
            for k, v in sd.items()
            if k.startswith("model.tubelet_embed.")
        }
        self.model.tubelet_embed.load_state_dict(embed_weights)

        # Teacher starts as EMA of student
        self.teacher.load_state_dict(self.model.student.state_dict())

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.model.student.parameters(), self.teacher.parameters()):
                t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer, optimizer_idx=None):
        self.update_teacher()

    def training_step(self, batch, batch_idx):
        img, actions = batch  # img: [B, T, H, W], actions: [B, T, A_dim]
        B, T, H, W = img.shape
        D = self.model.student.encoder.layers[0].self_attn.embed_dim

        # -------------------------------
        # Tubelet Embedding
        # -------------------------------
        x = img.unsqueeze(2)  # [B, T, 1, H, W]
        all_tokens = self.model.tubelet_embed(x)  # [B, N, D]
        tokens_per_frame = all_tokens.size(1) // T

        # Mean pool tokens per frame to get a single latent per frame
        frame_latents = all_tokens.reshape(B, T, tokens_per_frame, D).mean(
            dim=2
        )  # [B, T, D]

        # -------------------------------
        # Action Embedding
        # -------------------------------
        action_emb = self.action_embed(actions)  # [B, T, D]

        # -------------------------------
        # Build Causal Sequence [z0,a0,z1,a1,...]
        # -------------------------------
        seq = []
        for t in range(T):
            seq.append(frame_latents[:, t : t + 1, :])
            seq.append(action_emb[:, t : t + 1, :])
        seq = torch.cat(seq, dim=1)  # [B, 2*T, D]

        # -------------------------------
        # Predict Future Latent
        # -------------------------------
        # Use teacher latents for target (next frame)
        with torch.no_grad():
            teacher_latents = self.teacher(
                frame_latents[:, -1:, :].reshape(B, 1, D)
            )  # last frame latent

        # Autoregressive prediction
        # Here we shift input by 1 to predict next latent
        tgt_input = seq[:, :-1, :]  # all except last token
        pred_seq = self.latent_predictor(tgt_input, tgt_input)  # [B, 2*T-1, D]

        # Take last predicted latent for last frame
        student_pred_last_frame = pred_seq[:, -2, :]  # predicted latent for last frame

        # -------------------------------
        # Loss
        # -------------------------------
        loss = F.smooth_l1_loss(student_pred_last_frame, teacher_latents.squeeze(1))
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.model.tubelet_embed.parameters())
            + list(self.model.student.parameters())
            + list(self.latent_predictor.parameters())
            + list(self.action_embed.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
