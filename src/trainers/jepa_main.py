import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.pre_process_jepa import Resize, Stack

# from utils import extract_video_patches

DEBUG = True  # <-- turn this off once things work


class JEPAWrapper(torch.utils.data.IterableDataset):
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.config = config

    def __iter__(self):
        for sample in self.base_dataset:
            stacked_frames, eye_gazes = sample  # preserve gaze data
            yield stacked_frames, eye_gazes


# -------------------------------------------------
# Spatiotemporal Patch Extraction
# -------------------------------------------------
def extract_video_patches(video, a, b):
    if DEBUG:
        print("\n[extract_video_patches]")
        print("  input video shape:", video.shape)

    T, C, H, W = video.shape
    assert H % a == 0 and W % b == 0, "Patch size must divide H and W"

    patches = video.unfold(2, a, a).unfold(3, b, b)
    # (T, C, H//a, W//b, a, b)

    if DEBUG:
        print("  after unfold:", patches.shape)

    patches = patches.permute(2, 3, 0, 1, 4, 5)
    # (H//a, W//b, T, C, a, b)

    if DEBUG:
        print("  after permute:", patches.shape)

    patches = patches.reshape(-1, T * C * a * b)

    if DEBUG:
        print("  final patches shape:", patches.shape)

    return patches


# -------------------------------------------------
# Patch Embedding
# -------------------------------------------------
class VideoPatchEmbedding(nn.Module):
    def __init__(self, patch_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, stack, config):
        if DEBUG:
            print("\n[VideoPatchEmbedding]")
            print("  stack shape:", stack.shape)

        patches = extract_video_patches(
            stack,
            config["patchx"],
            config["patchy"],
        )

        tokens = self.proj(patches)

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
def random_patch_mask(num_patches, mask_ratio, device):
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
        mask_ratio=0.6,
        lr=1e-4,
        ema_decay=0.996,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # -----------------------------
        # Stack inside the model
        # -----------------------------
        self.stack_processor = Stack(config)
        self.resize = Resize(config)

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

    def forward(self, img, gaze):
        """
        Forward pass with resizing + stacking + patch embedding.
        """
        # Step 1: Resize and grayscale
        resized_img, _ = self.resize((img, gaze))  # Output shape: [1, 84, 84]

        # Step 2: Update the stack with the resized image
        stacked, _ = self.stack_processor((resized_img, gaze))  # Shape: [T, 1, 84, 84]

        if DEBUG:
            print("[forward] stacked shape:", stacked.shape)

        # Step 3: Split stack
        context_frames = stacked[:-1]  # all except last frame -> student sees this
        target_frame = stacked[-1:]  # last frame -> teacher sees this

        # Step 4: Patch embedding
        context_tokens = self.patch_embed(context_frames, self.config)  # student input
        target_tokens = self.patch_embed(target_frame, self.config)  # teacher input

        return context_tokens, target_tokens

    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        img, gaze = batch

        if DEBUG:
            print("\n==============================")
            print("[training_step] batch_idx:", batch_idx)
            print("img shape:", img.shape)

        # Forward pass handles stacking internally
        tokens = self.forward(img, gaze)
        tokens = tokens.unsqueeze(0)  # Add batch dimension

        N = tokens.size(1)
        device = tokens.device

        # Masking
        mask = random_patch_mask(N, self.mask_ratio, device)
        context_tokens = tokens[:, ~mask]

        # Student + predictor
        student_repr = self.student(context_tokens)
        pred = self.predictor(student_repr)

        # Teacher
        with torch.no_grad():
            target_tokens = tokens[:, mask]
            target = self.teacher(target_tokens)

        # Loss
        loss = F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1))
        self.log("train_loss", loss, prog_bar=True)

        if DEBUG:
            print("loss:", loss.item())
            print("==============================\n")

        return loss

    # -------------------------------------------------
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
