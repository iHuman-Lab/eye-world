import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.utils import block_mask_tubelets_vectorized
from models.vjepa import Predictor, TransformerEncoder, TubeletEmbedding


class VJEPA(pl.LightningModule):
    def __init__(
        self,
        patch_dim,
        config,
        embed_dim=1024,
        depth=10,
        predictor_depth=4,
        heads=16,
        mlp_dim=3072,
        mask_ratio=0.5,
        lr=1e-4,
        ema_decay=0.996,
        reg_coeff=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay
        self.reg_coeff = reg_coeff

        self.tubelet_embed = TubeletEmbedding(
            config=config,
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            img_size=config.get("size_x", 84),
        )

        self.student = TransformerEncoder(embed_dim, depth, heads, mlp_dim)
        self.teacher = TransformerEncoder(embed_dim, depth, heads, mlp_dim)
        # Predictor uses fewer heads (lighter cross-attention)
        self.pred = Predictor(embed_dim, predictor_depth, heads // 2, mlp_dim)

        self._init_teacher()
        for p in self.teacher.parameters():
            p.requires_grad = False

    # -------------------------------------------------

    def _init_teacher(self):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data.copy_(s.data)

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                t.mul_(self.ema_decay).add_(s, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    # -------------------------------------------------

    def _rep_stats(self, x):
        """x: [B, N, D]"""
        return {
            "var": x.var(dim=(0, 1)).mean(),
            "norm": x.norm(dim=-1).mean(),
        }

    # -------------------------------------------------

    def training_step(self, batch, batch_idx):
        img, _ = batch  # [B, T, H, W]
        B = img.shape[0]

        # --------------------------------------------------
        # Tokenize once — shared by student and teacher
        # --------------------------------------------------
        x = img.unsqueeze(2)  # [B, T, 1, H, W]
        all_tokens = self.tubelet_embed(x)  # [B, N, D]
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
        # N_visible is uniform across batch (vectorized masking guarantees this)
        # --------------------------------------------------
        N_visible = int((~mask_bool[0]).sum())
        visible_tokens = all_tokens[~mask_bool].reshape(
            B, N_visible, D
        )  # [B, N_visible, D]
        student_repr = self.student(visible_tokens)  # [B, N_visible, D]

        # --------------------------------------------------
        # Predictor
        # Queries = positional embeddings at masked positions
        # Context = student encoder outputs of visible tokens
        # --------------------------------------------------
        N_masked = N - N_visible
        pos_embed = self.tubelet_embed.pos_embed.expand(B, -1, -1)  # [B, N, D]
        masked_pos = pos_embed[mask_bool].reshape(B, N_masked, D)  # [B, N_masked, D]
        target_masked = target_full[mask_bool].reshape(
            B, N_masked, D
        )  # [B, N_masked, D]

        pred = self.pred(queries=masked_pos, context=student_repr)  # [B, N_masked, D]

        # --------------------------------------------------
        # Loss: smooth-L1 + variance regularization
        # --------------------------------------------------
        loss_jepa = F.smooth_l1_loss(pred, target_masked)
        pred_std = pred.std(dim=1)  # [B, D] — std over patch tokens
        loss_reg = F.relu(1.0 - pred_std).mean()
        loss = loss_jepa + self.reg_coeff * loss_reg

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        pred_stats = self._rep_stats(pred)
        tgt_stats = self._rep_stats(target_masked)

        self.log("loss/jepa", loss_jepa, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss/reg", loss_reg, on_step=True, on_epoch=True)
        self.log("loss/total", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("rep/pred_var", pred_stats["var"], on_epoch=True)
        self.log("rep/tgt_var", tgt_stats["var"], on_epoch=True)
        self.log("rep/pred_norm", pred_stats["norm"], on_epoch=True)
        self.log("rep/tgt_norm", tgt_stats["norm"], on_epoch=True)

        with torch.no_grad():
            n_params = sum(1 for _ in self.student.parameters())
            drift = sum(
                (s - t).pow(2).mean()
                for s, t in zip(self.student.parameters(), self.teacher.parameters())
            )
            self.log("ema/drift", drift / n_params, on_epoch=True)

        return loss

    # -------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.tubelet_embed.parameters())
            + list(self.student.parameters())
            + list(self.pred.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
