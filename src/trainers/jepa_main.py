import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.contrib import compute_padding, extract_tensor_patches

DEBUG = True  # turn off after debugging


# -------------------------------------------------
# 3D Sin-Cos Positional Embedding
# -------------------------------------------------
def get_3d_sincos_pos_embed(grid_size, grid_depth):

    embed_dim = 1024
    print("emm=", embed_dim)
    """Fixed 3D sin-cos positional embedding."""

    def get_1d_sincos(n, dim):
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2)
        omega = 1.0 / (10000**omega)
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        sincos = torch.cat([torch.sin(pos * omega), torch.cos(pos * omega)], dim=1)
        return sincos

    d_embed = get_1d_sincos(grid_depth, 341)
    h_embed = get_1d_sincos(grid_size, 342)
    w_embed = get_1d_sincos(grid_size, 342)

    # Expand embeddings to broadcast
    d = d_embed[:, None, None, :]  # [D, 1, 1, d_dim]
    h = h_embed[None, :, None, :]  # [1, H, 1, h_dim]
    w = w_embed[None, None, :, :]  # [1, 1, W, w_dim]

    # Broadcast and concatenate
    grid = torch.cat(
        [
            d.expand(-1, grid_size, grid_size, -1),
            h.expand(grid_depth, -1, grid_size, -1),
            w.expand(grid_depth, grid_size, -1, -1),
        ],
        dim=-1,
    )  # [D, H, W, embed_dim]
    """
    grid = torch.zeros(grid_depth, grid_size, grid_size, embed_dim)
    for t in range(grid_depth):
        for i in range(grid_size):
            for j in range(grid_size):
                grid[t, i, j] = torch.cat([d_embed[t], h_embed[i], w_embed[j]], dim=0)
    """

    return grid.view(-1, embed_dim).numpy()


# -------------------------------------------------
# Tubelet extraction
# -------------------------------------------------
def extract_tubelets(video, patch_size_x, patch_size_y, tubelet_size):
    """
    video: [B, T, C, H, W]
    returns: tubelets [B, N, tubelet_dim]
    """
    B, T, C, H, W = video.shape
    assert T % tubelet_size == 0, "T must be divisible by tubelet_size"

    # Merge batch & time for 2D patch extraction per frame
    video_reshaped = video.reshape(B * T, C, H, W)
    stride = (patch_size_x, patch_size_y)
    window = (patch_size_x, patch_size_y)
    padding = compute_padding((H, W), window, stride)

    patches = extract_tensor_patches(
        video_reshaped,
        window_size=window,
        stride=stride,
        padding=padding,
        allow_auto_padding=True,
    )  # [B*T, N_spatial, C, patchx, patchy]

    N_spatial = patches.size(1)
    patches = patches.flatten(2)  # [B*T, N_spatial, C*patchx*patchy]
    patches = patches.view(B, T, N_spatial, -1)

    # Merge temporal tubelets
    patches = patches.view(B, T // tubelet_size, tubelet_size, N_spatial, -1)
    tubelet_dim = patches.size(2) * patches.size(4)
    tubelets = patches.permute(0, 1, 3, 2, 4).reshape(B, -1, tubelet_dim)

    if DEBUG:
        print("extract_tubelets | tubelets.shape:", tubelets.shape)

    return tubelets, N_spatial


class Predictor(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, depth)

    def forward(self, queries, context):
        return self.decoder(queries, context)


# -------------------------------------------------
# Tubelet Embedding (Meta-style)
# -------------------------------------------------
class TubeletEmbedding(nn.Module):
    def __init__(self, config, patch_dim, embed_dim=1024, img_size=84):
        """
        config: dict containing patchx, patchy, stack_size, tubelet_size
        patch_dim: number of channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()

        self.patchx = config["patchx"]
        self.patchy = config["patchy"]
        self.stack_size = config["stack_size"]
        self.tubelet_size = config["tubelet_size"]
        self.embed_dim = 1024

        # Number of patches per frame
        self.grid_size_x = img_size // self.patchx
        self.grid_size_y = img_size // self.patchy

        # Number of tubelets along temporal axis
        self.grid_depth = self.stack_size // self.tubelet_size
        self.num_patches = self.grid_size_x * self.grid_size_y * self.grid_depth

        # Linear projection per tubelet
        self.proj = nn.Linear(self.patchx * self.patchy * self.tubelet_size, 1024)

        # 3D positional embedding
        pos_embed = get_3d_sincos_pos_embed(
            max(self.grid_size_x, self.grid_size_y), self.grid_depth
        )
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False
        )
        print("emm=", embed_dim)

    def forward(self, video):
        """
        video: [B, T, C, H, W]
        returns: [B, num_tubelets, embed_dim]
        """

        B, T, C, H, W = video.shape
        assert T == self.stack_size, f"Expected stack_size={self.stack_size}, got {T}"
        assert T % self.tubelet_size == 0, "T must be divisible by tubelet_size"

        # Extract 2D patches per frame
        video_reshaped = video.reshape(B * T, C, H, W)
        stride = (self.patchx, self.patchy)
        window = (self.patchx, self.patchy)
        padding = compute_padding((H, W), window, stride)

        patches = extract_tensor_patches(
            video_reshaped,
            window_size=window,
            stride=stride,
            padding=padding,
            allow_auto_padding=True,
        )  # [B*T, N_spatial, C, patchx, patchy]

        N_spatial = patches.size(1)
        patches = patches.flatten(2)  # [B*T, N_spatial, C*patchx*patchy]
        patches = patches.view(B, T, N_spatial, -1)

        # Merge temporal patches into tubelets (corrected)
        patches = patches.view(B, self.grid_depth, self.tubelet_size, N_spatial, -1)
        patches = patches.permute(
            0, 1, 3, 2, 4
        )  # [B, D, N_spatial, tubelet_size, patch_dim_flat]

        # Correct flattening: merge temporal + patch dimension, keep spatial separate
        B, D, N_spatial, T_size, patch_flat = patches.shape
        tubelets = patches.reshape(B, D * N_spatial, T_size * patch_flat)  # <- FIXED

        # Project each tubelet individually
        tokens = self.proj(tubelets)

        # Add 3D positional embeddings
        print(tokens.shape)
        print(self.pos_embed.shape)

        tokens = tokens + self.pos_embed

        if DEBUG:
            print("TubeletEmbedding | tokens.shape:", tokens.shape)

        return tokens


# -------------------------------------------------
# Transformer Encoder
# -------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)

    def forward(self, x):
        """
        if DEBUG:
            print("TransformerEncoder | input.shape:", x.shape)

        if DEBUG:
            print("TransformerEncoder | output.shape:", out.shape)"""

        out = self.encoder(x)
        return out


# -------------------------------------------------
# Random Mask
# -------------------------------------------------
def random_patch_mask(num_patches, mask_ratio, device):
    num_mask = int(num_patches * mask_ratio)
    perm = torch.randperm(num_patches, device=device)
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    mask[perm[:num_mask]] = True
    return mask


def block_mask_tubelets_vectorized(tubelets, drop_ratio=0.5, block_size=2):
    """
    Vectorized block masking for tubelets.

    tubelets: [B, N, D]  tensor of tubelets
    drop_ratio: fraction of tubelets to mask
    block_size: number of contiguous tubelets to mask as a block

    Returns:
        student_tokens: [B, N_unmasked, D]  tokens to feed to the student
        mask_bool: [B, N] boolean mask where True=masked
    """
    B, N, D = tubelets.shape
    device = tubelets.device

    # number of blocks and total blocks
    total_blocks = N // block_size
    num_mask_blocks = int(total_blocks * drop_ratio)

    # Randomly select block indices to mask for all batches
    perm = torch.rand(B, total_blocks, device=device).argsort(dim=1)
    mask_blocks = perm[:, :num_mask_blocks]  # [B, num_mask_blocks]

    # Convert block indices to patch indices
    block_offsets = torch.arange(block_size, device=device).view(1, 1, block_size)
    mask_idx = (
        mask_blocks.unsqueeze(-1) * block_size + block_offsets
    )  # [B, num_mask_blocks, block_size]
    mask_idx = mask_idx.view(B, -1)

    # Create boolean mask
    mask_bool = torch.zeros(B, N, device=device, dtype=torch.bool)
    mask_bool.scatter_(1, mask_idx, True)

    # Gather student tokens (unmasked)
    student_tokens = [tubelets[b, ~mask_bool[b]] for b in range(B)]
    student_tokens = torch.nn.utils.rnn.pad_sequence(
        student_tokens, batch_first=True
    )  # pad to max length

    return student_tokens, mask_bool


DEBUG = True  # turn off after debugging


# -------------------------------------------------
# Lightning V-JEPA
# -------------------------------------------------
class LightningVJEPA(pl.LightningModule):
    def __init__(
        self,
        patch_dim,
        config,
        embed_dim=1024,
        depth=12,
        predictor_depth=4,
        heads=16,
        mlp_dim=3072,
        mask_ratio=0.3,
        lr=1e-4,
        ema_decay=0.996,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        tubelet_dim = config["tubelet_size"] * (config["patchx"] * config["patchy"])

        # Modules
        self.tubelet_embed = TubeletEmbedding(
            config=config,
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            img_size=config.get("size_x", 84),
        )
        print(embed_dim)
        print("Attention heads:", heads)
        self.student = TransformerEncoder(1024, depth, 16, mlp_dim)
        self.teacher = TransformerEncoder(1024, depth, 16, mlp_dim)
        self.predictor = TransformerEncoder(1024, predictor_depth, 16, mlp_dim)
        self.predictor = Predictor(1024, predictor_depth, 16, mlp_dim)
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay

        self._init_teacher()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if DEBUG:
            print("LightningVJEPA initialized | tubelet_dim:", tubelet_dim)

    # -------------------------------------------------

    def _rep_stats(self, x):
        return {
            "var": x.var(dim=(0, 1)).mean(),
            "norm": x.norm(dim=-1).mean(),
        }

    # -------------------------------------------------

    def forward(self, x):
        """
        x: [B, T, H, W]
        """
        B, T, H, W = x.shape
        x = x.unsqueeze(2)  # [B, T, 1, H, W]

        context_tokens = self.tubelet_embed(x)
        target_tokens = self.tubelet_embed(x)

        if DEBUG:
            print("context_tokens:", context_tokens.shape)
            print("target_tokens:", target_tokens.shape)

        return context_tokens, target_tokens

    # -------------------------------------------------
    """
    def training_step(self, batch, batch_idx):
        img, _ = batch  # [B, T, H, W]

        # Tokenize video into tubelets
        context_tokens, target_tokens = self.forward(img)  # [B, N, D]

        # -------------------------------
        # Block masking BEFORE student sees tokens
        # -------------------------------
        use_masking = self.config.get("use_masking", False)
        if use_masking:
            student_tokens, mask_bool = block_mask_tubelets_vectorized(
                context_tokens,
                drop_ratio=self.mask_ratio,
                block_size=self.config.get("block_size", 2),
            )
        else:
            student_tokens = context_tokens
            mask_bool = torch.zeros(
                context_tokens.size(0),
                context_tokens.size(1),
                dtype=torch.bool,
                device=img.device,
            )

        if DEBUG:
            print("student_tokens.shape:", student_tokens.shape)
            print("masked tokens:", mask_bool.sum().item())

        B, N_full, D = context_tokens.shape

        with torch.no_grad():
            target = self.teacher(target_tokens)  # [B, N_full, D]

        # Student forward
        student_repr = self.student(student_tokens)  # [B, N_unmasked, D]

        # Predictor
        pred_masked = self.predictor(student_repr)  # [B, N_unmasked, D]

        # Select masked targets directly
        target_sel = target[mask_bool]  # [total_masked, D]

        # Now IMPORTANT:
        # You must also select predictions that correspond to masked positions.
        # But currently predictor outputs N_unmasked tokens, not masked ones.

        # So instead, compute loss against unmasked tokens:

        pred_sel = pred_masked.reshape(-1, D)
        target_visible = target[~mask_bool]

        loss = F.l1_loss(pred_sel, target_visible)

        B, N_full, D = context_tokens.shape
        pred_full = torch.zeros(B, N_full, D, device=context_tokens.device)
        pred_sel_list = []
        target_sel_list = []

        with torch.no_grad():
            target = self.teacher(target_tokens)

        for b in range(B):
            mask_b = mask_bool[b]  # [N_full]

            # Student forward (already only unmasked tokens)
            student_repr_b = self.student(
                student_tokens[b].unsqueeze(0)
            )  # [1, N_unmasked, D]

            # Predictor reconstructs masked tokens
            pred_masked_b = self.predictor(student_repr_b)  # [num_masked, D]
            pred_full[b, mask_b] = pred_masked_b.squeeze(0)

            # Collect masked tokens for loss
            pred_sel_list.append(pred_full[b, mask_b])
            target_sel_list.append(target[b, mask_b])

        # Concatenate across batch
        pred_sel = torch.cat(pred_sel_list, dim=0)
        target_sel = torch.cat(target_sel_list, dim=0)

        # Compute loss
        loss = F.l1_loss(pred_sel, target_sel)
        # -------------------------------
        # Logging
        # -------------------------------
        pred_stats = self._rep_stats(pred_full)
        tgt_stats = self._rep_stats(target)
        self.log("rep/pred_var", pred_stats["var"], on_epoch=True)
        self.log("rep/tgt_var", tgt_stats["var"], on_epoch=True)
        self.log("rep/pred_norm", pred_stats["norm"], on_epoch=True)
        self.log("rep/tgt_norm", tgt_stats["norm"], on_epoch=True)
        self.log("loss/l1", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Cosine similarity for masked tokens
        cos_sim = F.cosine_similarity(
            F.normalize(pred_sel, dim=-1), F.normalize(target_sel, dim=-1), dim=-1
        ).mean()
        self.log("align/cos_sim", cos_sim, on_epoch=True)

        # EMA drift between student and teacher
        with torch.no_grad():
            drift = 0.0
            n = 0
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                drift += (s - t).pow(2).mean()
                n += 1
            self.log("ema/student_teacher_mse", drift / n, on_epoch=True)

        if DEBUG:
            print("training_step | loss:", loss.item())

        return loss
        """

    def training_step(self, batch, batch_idx):
        img, _ = batch  # [B, T, H, W]

        # ---------------------------------
        # Tokenize video into tubelets
        # ---------------------------------
        context_tokens, target_tokens = self.forward(img)  # [B, N, D]

        # ---------------------------------
        # Block masking BEFORE student sees tokens
        # ---------------------------------
        use_masking = self.config.get("use_masking", False)

        if use_masking:
            student_tokens, mask_bool = block_mask_tubelets_vectorized(
                context_tokens,
                drop_ratio=self.mask_ratio,
                block_size=self.config.get("block_size", 2),
            )
        else:
            student_tokens = context_tokens
            mask_bool = torch.zeros(
                context_tokens.size(0),
                context_tokens.size(1),
                dtype=torch.bool,
                device=img.device,
            )

        if DEBUG:
            print("student_tokens.shape:", student_tokens.shape)
            print("masked tokens:", mask_bool.sum().item())

        # =====================================================
        # 🔥 BATCHED TRANSFORMER VERSION (NO LOOP AROUND IT)
        # =====================================================

        B, N_full, D = context_tokens.shape

        # -----------------------------
        # Teacher (full tokens)
        # -----------------------------
        with torch.no_grad():
            target_full = self.teacher(target_tokens)  # [B, N, D]

        B, N, D = context_tokens.shape

        # -----------------------------
        # Student (visible only)
        # -----------------------------
        student_repr = self.student(student_tokens)  # [B, N_visible_max, D]

        # -----------------------------
        # Prepare masked queries per sample
        # -----------------------------
        pos_embed = self.tubelet_embed.pos_embed.expand(B, -1, -1)  # [B, N, D]

        pred_list = []
        target_list = []

        for b in range(B):
            mask_b = mask_bool[b]  # [N]
            visible_b = ~mask_b

            # masked queries
            masked_pos_b = pos_embed[b][mask_b]  # [N_masked_b, D]

            # student context (remove padding)
            n_visible = visible_b.sum()
            context_b = student_repr[b, :n_visible]  # [N_visible_b, D]

            # teacher targets
            target_b = target_full[b][mask_b]  # [N_masked_b, D]

            # cross-attention predictor
            pred_b = self.predictor(
                masked_pos_b.unsqueeze(0),  # queries
                context_b.unsqueeze(0),  # memory
            ).squeeze(0)

            pred_list.append(pred_b)
            target_list.append(target_b)

        pred_masked = torch.cat(pred_list, dim=0)
        target_masked = torch.cat(target_list, dim=0)

        # -----------------------------
        # Loss
        # -----------------------------
        loss = F.l1_loss(pred_masked, target_masked)
        # ---------------------------------
        # Logging
        # ---------------------------------
        pred_stats = self._rep_stats(pred_masked)
        tgt_stats = self._rep_stats(target_masked)

        self.log("rep/pred_var", pred_stats["var"], on_epoch=True)
        self.log("rep/tgt_var", tgt_stats["var"], on_epoch=True)
        self.log("rep/pred_norm", pred_stats["norm"], on_epoch=True)
        self.log("rep/tgt_norm", tgt_stats["norm"], on_epoch=True)

        self.log("loss/l1", loss, on_step=True, on_epoch=True, prog_bar=True)
        """
        cos_sim = F.cosine_similarity(
            F.normalize(pred_sel, dim=-1),
            F.normalize(target_sel, dim=-1),
            dim=-1,
        ).mean()
        
        self.log("align/cos_sim", cos_sim, on_epoch=True)
        """
        # EMA drift
        with torch.no_grad():
            drift = 0.0
            n = 0
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                drift += (s - t).pow(2).mean()
                n += 1
            self.log("ema/student_teacher_mse", drift / n, on_epoch=True)

        if DEBUG:
            print("training_step | loss:", loss.item())

        return loss

    # -------------------------------------------------

    def _init_teacher(self):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data.copy_(s.data)

    # -------------------------------------------------

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                t.mul_(self.ema_decay).add_(s, alpha=1.0 - self.ema_decay)

    # -------------------------------------------------

    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    # -------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.tubelet_embed.parameters())
            + list(self.student.parameters())
            + list(self.predictor.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
