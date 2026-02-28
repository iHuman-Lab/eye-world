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

    grid = torch.zeros(grid_depth, grid_size, grid_size, embed_dim)
    for t in range(grid_depth):
        for i in range(grid_size):
            for j in range(grid_size):
                grid[t, i, j] = torch.cat([d_embed[t], h_embed[i], w_embed[j]], dim=0)
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
        self.embed_dim = embed_dim

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
        if DEBUG:
            print("TransformerEncoder | input.shape:", x.shape)
        out = self.encoder(x)
        if DEBUG:
            print("TransformerEncoder | output.shape:", out.shape)
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
        heads=12,
        mlp_dim=3072,
        mask_ratio=0.3,
        lr=1e-4,
        ema_decay=0.996,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Compute tubelet dim correctly
        print(patch_dim)

        tubelet_dim = config["tubelet_size"] * (config["patchx"] * config["patchy"])

        # Modulesimport pytorch_lightning as pl


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

    grid = torch.zeros(grid_depth, grid_size, grid_size, embed_dim)
    for t in range(grid_depth):
        for i in range(grid_size):
            for j in range(grid_size):
                grid[t, i, j] = torch.cat([d_embed[t], h_embed[i], w_embed[j]], dim=0)
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
        self.embed_dim = embed_dim

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

    '''def forward(self, video):
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

        # Merge temporal patches into tubelets
        patches = patches.view(B, self.grid_depth, self.tubelet_size, N_spatial, -1)
        patches = patches.permute(
            0, 1, 3, 2, 4
        )  # [B, D, N_spatial, tubelet_size, patch_dim_flat]
        tubelets = patches.reshape(B, -1, self.tubelet_size * patches.size(-1))

        # Project each tubelet individually
        tokens = self.proj(tubelets)

        # Add 3D positional embeddings
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]

        if DEBUG:
            print("TubeletEmbedding | tokens.shape:", tokens.shape)

        return tokens
'''


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
            print("TransformerEncoder | input.shape:", x.shape)
        out = self.encoder(x)
        if DEBUG:
            print("TransformerEncoder | output.shape:", out.shape)
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
        heads=12,
        mlp_dim=3072,
        mask_ratio=0.3,
        lr=1e-4,
        ema_decay=0.996,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Compute tubelet dim correctly
        print(patch_dim)

        tubelet_dim = config["tubelet_size"] * (config["patchx"] * config["patchy"])

        # Modules
        self.tubelet_embed = TubeletEmbedding(
            config=config,
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            img_size=config.get("size_x", 84),  # optional, default=84
        )
        self.student = TransformerEncoder(1024, depth, 16, mlp_dim)
        self.teacher = TransformerEncoder(1024, depth, 16, mlp_dim)
        self.predictor = TransformerEncoder(1024, predictor_depth, 16, mlp_dim)

        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay

        self._init_teacher()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if DEBUG:
            print("LightningVJEPA | initialized | tubelet_dim:", tubelet_dim)

    def _rep_stats(self, x):
        return {"var": x.var(dim=(0, 1)).mean(), "norm": x.norm(dim=-1).mean()}

    def forward(self, x):
        """
        x: [B, T, H, W]
        """
        B, T, H, W = x.shape
        x = x.unsqueeze(2)  # [B, T, C=1, H, W]

        context_frames = x
        target_frames = x

        # Pass only video tensor
        context_tokens = self.tubelet_embed(context_frames)
        target_tokens = self.tubelet_embed(target_frames)

        if DEBUG:
            print("LightningVJEPA | context_tokens.shape:", context_tokens.shape)
            print("LightningVJEPA | target_tokens.shape:", target_tokens.shape)

        return context_tokens, target_tokens

    def training_step(self, batch, batch_idx):
        img, _ = batch
        context_tokens, target_tokens = self.forward(img)
        B, Nt, D = context_tokens.shape

        use_masking = self.config.get("use_masking", False)
        idx = slice(None)
        if use_masking:
            mask = random_patch_mask(Nt, self.mask_ratio, img.device)
            idx = mask.nonzero(as_tuple=True)[0]
            if DEBUG:
                print("training_step | masked tokens:", idx.numel(), "/", Nt)

        # Student
        student_repr = self.student(context_tokens)
        # Predictor
        pred = self.predictor(student_repr)
        # Teacher
        with torch.no_grad():
            target = self.teacher(target_tokens)

        # Loss
        pred_sel = pred[:, idx, :]
        target_sel = target[:, idx, :]
        loss = F.l1_loss(pred_sel, target_sel)

        # Logs
        pred_stats = self._rep_stats(pred)
        tgt_stats = self._rep_stats(target)
        self.log("rep/pred_var", pred_stats["var"], on_epoch=True)
        self.log("rep/tgt_var", tgt_stats["var"], on_epoch=True)
        self.log("rep/pred_norm", pred_stats["norm"], on_epoch=True)
        self.log("rep/tgt_norm", tgt_stats["norm"], on_epoch=True)
        self.log("loss/mse", loss, on_step=True, on_epoch=True, prog_bar=True)

        cos_sim = F.cosine_similarity(
            F.normalize(pred_sel, dim=-1), F.normalize(target_sel, dim=-1), dim=-1
        ).mean()
        self.log("align/cos_sim", cos_sim, on_epoch=True)

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

    def _init_teacher(self):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data.copy_(s.data)

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                t.mul_(self.ema_decay).add_(s, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.tubelet_embed.parameters())
            + list(self.student.parameters())
            + list(self.predictor.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )

        self.tubelet_embed = TubeletEmbedding(
            config=config,
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            img_size=config.get("size_x", 84),  # optional, default=84
        )
        self.student = TransformerEncoder(1024, depth, 16, mlp_dim)
        self.teacher = TransformerEncoder(1024, depth, 16, mlp_dim)
        self.predictor = TransformerEncoder(1024, predictor_depth, 16, mlp_dim)

        self.mask_ratio = mask_ratio
        self.lr = lr
        self.ema_decay = ema_decay

        self._init_teacher()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if DEBUG:
            print("LightningVJEPA | initialized | tubelet_dim:", tubelet_dim)

    def _rep_stats(self, x):
        return {"var": x.var(dim=(0, 1)).mean(), "norm": x.norm(dim=-1).mean()}

    def forward(self, x):
        """
        x: [B, T, H, W]
        """
        B, T, H, W = x.shape
        x = x.unsqueeze(2)  # [B, T, C=1, H, W]

        context_frames = x
        target_frames = x

        # Pass only video tensor
        context_tokens = self.tubelet_embed(context_frames)
        target_tokens = self.tubelet_embed(target_frames)

        if DEBUG:
            print("LightningVJEPA | context_tokens.shape:", context_tokens.shape)
            print("LightningVJEPA | target_tokens.shape:", target_tokens.shape)

        return context_tokens, target_tokens

    def training_step(self, batch, batch_idx):
        img, _ = batch
        context_tokens, target_tokens = self.forward(img)
        B, Nt, D = context_tokens.shape

        use_masking = self.config.get("use_masking", False)
        idx = slice(None)
        if use_masking:
            mask = random_patch_mask(Nt, self.mask_ratio, img.device)
            idx = mask.nonzero(as_tuple=True)[0]
            if DEBUG:
                print("training_step | masked tokens:", idx.numel(), "/", Nt)

        # Student
        student_repr = self.student(context_tokens)
        # Predictor
        pred = self.predictor(student_repr)
        # Teacher
        with torch.no_grad():
            target = self.teacher(target_tokens)

        # Loss
        pred_sel = pred[:, idx, :]
        target_sel = target[:, idx, :]
        loss = F.l1_loss(pred_sel, target_sel)

        # Logs
        pred_stats = self._rep_stats(pred)
        tgt_stats = self._rep_stats(target)
        self.log("rep/pred_var", pred_stats["var"], on_epoch=True)
        self.log("rep/tgt_var", tgt_stats["var"], on_epoch=True)
        self.log("rep/pred_norm", pred_stats["norm"], on_epoch=True)
        self.log("rep/tgt_norm", tgt_stats["norm"], on_epoch=True)
        self.log("loss/mse", loss, on_step=True, on_epoch=True, prog_bar=True)

        cos_sim = F.cosine_similarity(
            F.normalize(pred_sel, dim=-1), F.normalize(target_sel, dim=-1), dim=-1
        ).mean()
        self.log("align/cos_sim", cos_sim, on_epoch=True)

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

    def _init_teacher(self):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data.copy_(s.data)

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                t.mul_(self.ema_decay).add_(s, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer):
        self.update_teacher()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.tubelet_embed.parameters())
            + list(self.student.parameters())
            + list(self.predictor.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
