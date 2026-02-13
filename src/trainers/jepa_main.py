import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.contrib import compute_padding, extract_tensor_patches

# from utils import extract_video_patches

DEBUG = True  # <-- turn this off once things work


# -------------------------------------------------
# Spatiotemporal Patch Extraction
# -------------------------------------------------


def extract_video_patches_kornia(video, a, b, DEBUG=False):
    """
    video: [B, T, C, H, W]
    returns: [B, T*N, C*a*b]
    """
    B, T, C, H, W = video.shape

    # Merge batch and time
    video = video.reshape(B * T, C, H, W)

    stride = (a, b)
    window_size = (a, b)
    padding = compute_padding((H, W), window_size, stride)

    patches = extract_tensor_patches(
        video,
        window_size=window_size,
        stride=stride,
        padding=padding,
        allow_auto_padding=True,
    )
    # [B*T, N, C, a, b]

    if DEBUG:
        print("patches raw:", patches.shape)

    patches = patches.flatten(2)  # [B*T, N, C*a*b]
    N = patches.size(1)

    patches = patches.reshape(B, T * N, -1)

    if DEBUG:
        print("final patches:", patches.shape)

    return patches


"would the transformer be able to tell that the tublets are a stack of patches with out specific temporial embedding"


# -------------------------------------------------
# Patch Embedding
# -------------------------------------------------
class VideoPatchEmbedding(nn.Module):
    def __init__(self, patch_dim, embed_dim, max_frames=16, max_patches=256):
        super().__init__()

        self.proj = nn.Linear(patch_dim, embed_dim)

        # Separate spatial + temporal embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, max_patches, embed_dim))
        self.temporal_pos = nn.Parameter(torch.randn(1, max_frames, embed_dim))

    def forward(self, stack, config):
        if DEBUG:
            print("\n[VideoPatchEmbedding]")
            print("  stack shape:", stack.shape)

        B, T, C, H, W = stack.shape

        # -------------------------------------------------
        # Extract patches (still flattened)
        # -------------------------------------------------
        patches = extract_video_patches_kornia(
            stack,
            config["patchx"],
            config["patchy"],
        )
        # [B, T*N, patch_dim]

        # Compute N (patches per frame)

        N = patches.shape[1] // T

        # -------------------------------------------------
        # Reshape to separate time dimension
        # -------------------------------------------------
        patches = patches.view(B, T, N, -1)  # [B, T, N, patch_dim]

        # Linear projection
        tokens = self.proj(patches)  # [B, T, N, D]

        # -------------------------------------------------
        # Add spatial + temporal embeddings
        # -------------------------------------------------
        tokens = (
            tokens
            + self.spatial_pos[:, :N].unsqueeze(1)  # [1,1,N,D]
            + self.temporal_pos[:, :T].unsqueeze(2)  # [1,T,1,D]
        )

        # Flatten back for transformer
        tokens = tokens.view(B, T * N, -1)

        if DEBUG:
            print("  embedded tokens shape:", tokens.shape)

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
        if DEBUG:
            print("\n[TransformerEncoder]")
            print("  input shape:", x.shape)

        out = self.encoder(x)

        if DEBUG:
            print("  output shape:", out.shape)

        return out


# -------------------------------------------------
# Masking
# -------------------------------------------------
"""def random_patch_mask(num_patches, mask_ratio, device):
    if DEBUG:
        print("\n[random_patch_mask]")
        print("  num_patches:", num_patches)
        print("  mask_ratio:", mask_ratio)

    num_mask = int(num_patches * mask_ratio)
    perm = torch.randperm(num_patches, device=device)
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    mask[perm[:num_mask]] = True

    if DEBUG:
        print("  masked patches:", mask.sum().item())
        print("  context patches:", (~mask).sum().item())

    return mask"""


def random_patch_mask(num_patches, mask_ratio, device):
    num_mask = int(num_patches * mask_ratio)
    perm = torch.randperm(num_patches, device=device)
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    mask[perm[:num_mask]] = True
    return mask


# -------------------------------------------------
# Lightning V-JEPA 2 World Model
# -------------------------------------------------
class LightningVJEPA(pl.LightningModule):
    def __init__(
        self,
        patch_dim,
        config,  # pass config so Stack knows stack_size
        embed_dim=768,
        depth=12,
        predictor_depth=4,
        heads=12,
        mlp_dim=3072,
        mask_ratio=0.3,
        lr=1e-4,
        ema_decay=0.996,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # -----------------------------
        # Modules
        # -----------------------------
        self.patch_embed = VideoPatchEmbedding(patch_dim, embed_dim)
        self.student = TransformerEncoder(embed_dim, depth, heads, mlp_dim)
        self.teacher = TransformerEncoder(embed_dim, depth, heads, mlp_dim)
        self.predictor = TransformerEncoder(embed_dim, predictor_depth, heads, mlp_dim)

        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay

        self._init_teacher()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if DEBUG:
            print("\n[LightningVJEPA initialized]")
            print("  patch_dim:", patch_dim)
            print("  embed_dim:", embed_dim)

    def _rep_stats(self, x):
        """
        x: [B, N, D]
        Returns variance + norm across batch and tokens
        """
        return {
            "var": x.var(dim=(0, 1)).mean(),
            "norm": x.norm(dim=-1).mean(),
        }

    def forward(self, x):
        """
        x: [B, T, H, W]  (e.g. [32, 4, 84, 84])
        Student sees frames t0..t(T-2)
        Teacher sees frames t1..t(T-1)
        """
        if DEBUG:
            print("[forward] input x:", x.shape)

        B, T, H, W = x.shape

        # add channel dimension (grayscale)
        x = x.unsqueeze(2)  # [B, T, 1, H, W]

        # temporal shift
        # context_frames = x[:, :-1]  # t0, t1, t2
        # target_frames = x[:, 1:]  # t1, t2, t3
        context_frames = x  # t0, t1, t2
        target_frames = x  # t1, t2, t3

        if DEBUG:
            print("[forward] context_frames:", context_frames.shape)
            print("[forward] target_frames:", target_frames.shape)

        # patch embedding
        context_tokens = self.patch_embed(context_frames, self.config)  # [B, Nc, D]

        target_tokens = self.patch_embed(target_frames, self.config)  # [B, Nt, D]

        if DEBUG:
            print("[forward] context_tokens:", context_tokens.shape)
            print("[forward] target_tokens:", target_tokens.shape)

        return context_tokens, target_tokens

    # -------------------------------------------------
    """ 
   def training_step(self, batch, batch_idx):
        img, _ = batch  # ignore gaze for now

        if DEBUG:
            print("\n==============================")
            print("[training_step] batch_idx:", batch_idx)
            print("img shape:", img.shape)

        # Forward
        context_tokens, target_tokens = self.forward(img)

        # -----------------------------
        # Masking (only on target)
        # -----------------------------
        B, Nt, D = target_tokens.shape

        use_masking = self.config.get("use_masking", False)

        if use_masking:
            mask = random_patch_mask(Nt, self.mask_ratio, img.device)

            # keep ONLY masked target tokens
            target_in = target_tokens[:, mask, :]
        else:
            mask = None
            target_in = target_tokens

        student_in = context_tokens  # student sees full context

        target_in = target_tokens
        # target_in = target_tokens[:, mask, :]  # teacher predicts masked patches kept for future use with eyegaze

        # -----------------------------
        # Student + predictor
        # -----------------------------
        student_repr = self.student(student_in)
        pred = self.predictor(student_repr)

        # -----------------------------
        # Teacher
        # -----------------------------
        with torch.no_grad():
            target = self.teacher(target_in)

        # -----------------------------
        # Loss
        # -----------------------------
        loss = F.mse_loss(
            F.normalize(pred, dim=-1),
            F.normalize(target, dim=-1),
        )

        self.log("train_loss", loss, prog_bar=True)

        if DEBUG:
            print("loss:", loss.item())
            print("==============================\n")

        return loss
        """

    def training_step(self, batch, batch_idx):
        img, _ = batch

        if DEBUG:
            print("\n" + "=" * 80)
            print(f"[training_step] batch_idx: {batch_idx}")
            print("img shape:", img.shape)

        # -------------------------------------------------
        # Forward (patch embedding + temporal shift)
        # -------------------------------------------------
        context_tokens, target_tokens = self.forward(img)
        B, Nt, D = context_tokens.shape

        if DEBUG:
            print("[tokens]")
            print("  context_tokens:", context_tokens.shape)
            print("  target_tokens :", target_tokens.shape)

        use_masking = self.config.get("use_masking", False)

        # -------------------------------------------------
        # Mask (INDICES ONLY)
        # -------------------------------------------------
        if use_masking:
            mask = random_patch_mask(Nt, self.mask_ratio, img.device)
            idx = mask.nonzero(as_tuple=True)[0]

            if DEBUG:
                print("[masking]")
                print("  use_masking:", use_masking)
                print("  mask_ratio :", self.mask_ratio)
                print("  masked tokens:", idx.numel(), "/", Nt)
        else:
            idx = slice(None)
            if DEBUG:
                print("[masking] OFF (using all tokens)")

        # -------------------------------------------------
        # Student (FULL sequence)
        # -------------------------------------------------
        if DEBUG:
            print("[student]")
            print("  input :", context_tokens.shape)

        student_repr = self.student(context_tokens)  # [B, Nt, D]

        if DEBUG:
            print("  output:", student_repr.shape)

        # -------------------------------------------------
        # Predictor (FULL sequence)
        # -------------------------------------------------
        if DEBUG:
            print("[predictor]")
            print("  input :", student_repr.shape)

        pred = self.predictor(student_repr)  # [B, Nt, D]

        if DEBUG:
            print("  output:", pred.shape)

        # -------------------------------------------------
        # Teacher (FULL sequence, NO MASK)
        # -------------------------------------------------
        with torch.no_grad():
            if DEBUG:
                print("[teacher]")
                print("  input :", target_tokens.shape)

            target = self.teacher(target_tokens)  # [B, Nt, D]

            if DEBUG:
                print("  output:", target.shape)

            # -------------------------------------------------
        pred_stats = self._rep_stats(pred)
        tgt_stats = self._rep_stats(target)

        self.log("rep/pred_var", pred_stats["var"], on_epoch=True)
        self.log("rep/tgt_var", tgt_stats["var"], on_epoch=True)

        self.log("rep/pred_norm", pred_stats["norm"], on_epoch=True)
        self.log("rep/tgt_norm", tgt_stats["norm"], on_epoch=True)

        # -------------------------------------------------
        # Loss (MASKED INDICES ONLY)
        # -------------------------------------------------
        pred_sel = pred[:, idx, :]  # [B, Nm, D]
        target_sel = target[:, idx, :]  # [B, Nm, D]

        if DEBUG:
            print("[loss]")
            print("  pred_sel  :", pred_sel.shape)
            print("  target_sel:", target_sel.shape)
        """
        loss = F.l1_loss(
            F.normalize(pred_sel, dim=-1),
            F.normalize(target_sel, dim=-1),
        )

        
        loss = F.mse_loss(
            F.normalize(pred_sel, dim=-1),
            F.normalize(target_sel, dim=-1),
        )
        """
        loss = F.l1_loss(pred_sel, target_sel)

        if DEBUG:
            print("  loss:", loss.item())
            print("=" * 80)

        # -------------------------------------------------
        # Overall loss + alignment  <<< NEW CODE
        # -------------------------------------------------
        self.log(
            "loss/mse",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        cos_sim = F.cosine_similarity(
            F.normalize(pred_sel, dim=-1),
            F.normalize(target_sel, dim=-1),
            dim=-1,
        ).mean()

        self.log("align/cos_sim", cos_sim, on_epoch=True)

        # -------------------------------------------------
        # Student vs Teacher drift (EMA health)  <<< NEW CODE
        # -------------------------------------------------
        with torch.no_grad():
            drift = 0.0
            n = 0
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                drift += (s - t).pow(2).mean()
                n += 1
            drift = drift / n

        self.log("ema/student_teacher_mse", drift, on_epoch=True)

        return loss

    # -----------
    # --------------------------------------
    def _init_teacher(self):
        if DEBUG:
            print("\n[Initializing teacher weights]")
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data.copy_(s.data)

    def update_teacher(self):
        if DEBUG:
            print("[EMA update teacher]")
        with torch.no_grad():
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                t.mul_(self.ema_decay).add_(s, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    # -------------------------------------------------
    def configure_optimizers(self):
        if DEBUG:
            print("[configure_optimizers]")
        return torch.optim.AdamW(
            list(self.patch_embed.parameters())
            + list(self.student.parameters())
            + list(self.predictor.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
