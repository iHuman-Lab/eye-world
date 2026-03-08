import copy

import torch.nn as nn
from kornia.contrib import compute_padding, extract_tensor_patches

from models.utils import get_3d_sincos_pos_embed

DEBUG = False  # set True for debugging


# -------------------------------------------------
# VJEPA Encoder — inference model (tubelet embed + student)
# -------------------------------------------------
class VJEPAEncoder(nn.Module):
    def __init__(self, tubelet_embed, student):
        super().__init__()
        self.tubelet_embed = tubelet_embed
        self.student = student

    def forward(self, video):
        """
        video: [B, T, H, W]
        returns: [B, N, D] — patch representations
        """
        x = video.unsqueeze(2)  # [B, T, 1, H, W]
        tokens = self.tubelet_embed(x)  # [B, N, D]
        return self.student(tokens)  # [B, N, D]


# -------------------------------------------------
# Tubelet Embedding
# -------------------------------------------------
class TubeletEmbedding(nn.Module):
    def __init__(self, config, patch_dim, embed_dim=1024, img_size=84):
        """
        config:    dict with patchx, patchy, stack_size, tubelet_size
        patch_dim: number of channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()
        self.patchx = config["patchx"]
        self.patchy = config["patchy"]
        self.stack_size = config["stack_size"]
        self.tubelet_size = config["tubelet_size"]
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim

        self.grid_h = img_size // self.patchy
        self.grid_w = img_size // self.patchx
        self.grid_depth = self.stack_size // self.tubelet_size
        self.num_patches = self.grid_h * self.grid_w * self.grid_depth

        tubelet_input_dim = patch_dim * self.patchx * self.patchy * self.tubelet_size
        self.proj = nn.Linear(tubelet_input_dim, embed_dim)

        pos_embed = get_3d_sincos_pos_embed(
            self.grid_h, self.grid_w, self.grid_depth, embed_dim
        )  # [N, embed_dim]
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # [1, N, embed_dim]

    def forward(self, video):
        """
        video: [B, T, C, H, W]
        returns: [B, num_patches, embed_dim]
        """
        B, T, C, H, W = video.shape
        assert T == self.stack_size, f"Expected T={self.stack_size}, got {T}"

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

        # Merge frames into tubelets along temporal axis
        patches = patches.view(B, self.grid_depth, self.tubelet_size, N_spatial, -1)
        patches = patches.permute(
            0, 1, 3, 2, 4
        )  # [B, D, N_spatial, tubelet_size, patch_flat]
        B_, D, Ns, Ts, Pf = patches.shape
        tubelets = patches.reshape(B_, D * Ns, Ts * Pf)  # [B, N, tubelet_input_dim]

        tokens = self.proj(tubelets) + self.pos_embed  # [B, N, embed_dim]

        if DEBUG:
            print("TubeletEmbedding | tokens:", tokens.shape)
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
        return self.encoder(x)


# -------------------------------------------------
# Predictor (cross-attention decoder)
# -------------------------------------------------
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
        """
        queries: [B, N_masked, D]  — positional embeddings at masked positions
        context: [B, N_visible, D] — student encoder outputs
        returns: [B, N_masked, D]
        """
        return self.decoder(tgt=queries, memory=context)
