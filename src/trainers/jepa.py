import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from models.utils import atari_to_gym, block_mask_tubelets_vectorized
from trainers.utils import format_batch_for_vjepa

config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)
a = atari_to_gym(1)


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


class ActionConditionVJEPA(pl.LightningModule):
    def __init__(
        self,
        model,
        action_embed,
        config,
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
        self.action_embed = action_embed

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

        ckpt_path = config["ckpt_path"]
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")

            # Load weights
            student_weights = {
                k.replace("model.student.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.student.")
            }
            self.model.student.load_state_dict(student_weights)

            # Load tubelet embedding weights
            embed_weights = {
                k.replace("model.tubelet_embed.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.tubelet_embed.")
            }
            self.model.tubelet_embed.load_state_dict(embed_weights)
        except FileNotFoundError:
            pass

        # Teacher starts as EMA of student
        self.teacher.load_state_dict(self.model.student.state_dict())

    # --------------------------------------------------
    # Gaze helpers
    # --------------------------------------------------

    def _get_tubelet_geometry(self, frames, token_count):
        """
        Uses V-JEPA config where:

            patchx = patch width in pixels
            patchy = patch height in pixels
            tubelet_size = frames per tubelet

        Example:
            H = 84
            W = 84
            patchx = 21
            patchy = 21
            tubelet_size = 4
            T = 4

            temporal_grid = 1
            grid_x = 84 // 21 = 4
            grid_y = 84 // 21 = 4

            expected_tokens = 1 * 4 * 4 = 16
        """

        if frames.dim() == 4:
            B, T, H, W = frames.shape
        elif frames.dim() == 5:
            B, T, C, H, W = frames.shape
        else:
            raise ValueError(f"Unexpected frame shape: {frames.shape}")

        frames_per_tubelet = int(self.config["tubelet_size"])

        # These are pixel patch sizes, not patch counts.
        patch_w_pixels = int(self.config["patchx"])
        patch_h_pixels = int(self.config["patchy"])

        if T % frames_per_tubelet != 0:
            raise RuntimeError(
                f"Frame count T={T} is not divisible by tubelet_size={frames_per_tubelet}."
            )

        if W < patch_w_pixels or H < patch_h_pixels:
            raise RuntimeError(
                f"Patch size is larger than frame size. "
                f"Frame H,W=({H},{W}), patchy,patchx=({patch_h_pixels},{patch_w_pixels})."
            )

        temporal_grid = T // frames_per_tubelet
        grid_x = W // patch_w_pixels
        grid_y = H // patch_h_pixels

        expected_tokens = temporal_grid * grid_y * grid_x

        if expected_tokens != token_count:
            raise RuntimeError(
                f"Tubelet geometry mismatch. Expected {expected_tokens} tokens from "
                f"temporal_grid={temporal_grid}, grid_y={grid_y}, grid_x={grid_x}, "
                f"patchy={patch_h_pixels}, patchx={patch_w_pixels}, "
                f"but encoder produced {token_count} tokens. "
                f"Check whether tubelet_embed crops/pads frames or uses a different flattening order."
            )

        return {
            "frames_per_tubelet": frames_per_tubelet,
            "temporal_grid": temporal_grid,
            "grid_y": grid_y,
            "grid_x": grid_x,
            "patch_h_pixels": patch_h_pixels,
            "patch_w_pixels": patch_w_pixels,
            "height": H,
            "width": W,
        }

    def _slice_gaze(self, gaze_locations, start, end):
        """
        Expected gaze shapes:
            [B, T, 2]
            [B, T, G, 2]

        Coordinates are expected as:
            [x, y]
        """
        if gaze_locations is None:
            return None

        return gaze_locations[:, start:end]

    def gaze_to_heatmap(self, frames, gaze_locations):
        """
        Converts gaze points into image-space heatmaps.

        Args:
            frames:
                [B, T, H, W] or [B, T, C, H, W]

            gaze_locations:
                [B, T, 2] or [B, T, G, 2]

        Returns:
            heatmap:
                [B, T, H, W]
        """
        if frames.dim() == 4:
            B, T, H, W = frames.shape
        elif frames.dim() == 5:
            B, T, C, H, W = frames.shape
        else:
            raise ValueError(f"Unexpected frame shape: {frames.shape}")

        device = frames.device
        dtype = frames.dtype

        heatmap = torch.zeros(B, T, H, W, device=device, dtype=dtype)

        if gaze_locations is None:
            return heatmap

        gaze_locations = gaze_locations.to(device=device, dtype=dtype)

        if gaze_locations.dim() == 3:
            gaze_locations = gaze_locations.unsqueeze(2)

        sigma = float(self.config.get("gaze_sigma", 4.0))

        gaze_is_normalized = bool(self.config.get("gaze_is_normalized", False))

        gaze_source_width = self.config.get("gaze_source_width", None)
        gaze_source_height = self.config.get("gaze_source_height", None)

        yy = torch.arange(H, device=device, dtype=dtype).view(H, 1)
        xx = torch.arange(W, device=device, dtype=dtype).view(1, W)

        B_gaze, T_gaze, G, _ = gaze_locations.shape
        T_use = min(T, T_gaze)

        for b in range(B):
            for t in range(T_use):
                points = gaze_locations[b, t]

                for g in range(G):
                    x = points[g, 0]
                    y = points[g, 1]

                    if not torch.isfinite(x) or not torch.isfinite(y):
                        continue

                    # Case 1: gaze is normalized to [0, 1].
                    if gaze_is_normalized:
                        x = x * (W - 1)
                        y = y * (H - 1)

                    # Case 2: gaze is in original Atari/ALE resolution.
                    elif (
                        gaze_source_width is not None and gaze_source_height is not None
                    ):
                        x = x / float(gaze_source_width - 1) * float(W - 1)
                        y = y / float(gaze_source_height - 1) * float(H - 1)

                    # Case 3: gaze is already in resized frame coordinates.
                    else:
                        pass

                    if x < 0 or x >= W or y < 0 or y >= H:
                        continue

                    dist2 = (xx - x) ** 2 + (yy - y) ** 2
                    heatmap[b, t] += torch.exp(-dist2 / (2.0 * sigma * sigma))

        return heatmap

    def heatmap_to_tubelet_scores(self, heatmap, token_count):
        """
        Converts image-space heatmaps to tubelet heat scores.

        Returns:
            token_heat:
                [B, N]

            token_labels:
                [N, 3], where each label is:
                    [temporal_tubelet_index, patch_y, patch_x]
        """
        B, T, H, W = heatmap.shape

        geometry = self._get_tubelet_geometry(
            frames=heatmap,
            token_count=token_count,
        )

        frames_per_tubelet = geometry["frames_per_tubelet"]
        temporal_grid = geometry["temporal_grid"]
        grid_y = geometry["grid_y"]
        grid_x = geometry["grid_x"]
        patch_h = geometry["patch_h_pixels"]
        patch_w = geometry["patch_w_pixels"]

        T_crop = temporal_grid * frames_per_tubelet
        H_crop = grid_y * patch_h
        W_crop = grid_x * patch_w

        heatmap = heatmap[:, :T_crop, :H_crop, :W_crop]

        # [B, T, H, W]
        # -> [B, temporal_grid, frames_per_tubelet, grid_y, patch_h, grid_x, patch_w]
        heatmap = heatmap.view(
            B,
            temporal_grid,
            frames_per_tubelet,
            grid_y,
            patch_h,
            grid_x,
            patch_w,
        )

        # Average heat inside each tubelet.
        # Result: [B, temporal_grid, grid_y, grid_x]
        tubelet_heat = heatmap.mean(dim=(2, 4, 6))

        # Flatten order: temporal, y, x.
        # This must match your tubelet_embed token order.
        token_heat = tubelet_heat.reshape(B, -1)

        labels = []

        for tt in range(temporal_grid):
            for yy in range(grid_y):
                for xx in range(grid_x):
                    labels.append([tt, yy, xx])

        token_labels = torch.tensor(
            labels,
            device=heatmap.device,
            dtype=torch.long,
        )

        if token_heat.shape[1] != token_count:
            raise RuntimeError(
                f"Produced {token_heat.shape[1]} gaze tubelet scores, "
                f"but expected {token_count}."
            )

        return token_heat, token_labels

    def make_gaze_tubelet_mask(self, frames, gaze_locations, token_count):
        """
        Builds a top-k gaze mask.

        Returns:
            mask:
                [B, N] bool

            topk_indices:
                [B, K]

            topk_labels:
                [B, K, 3]

            token_heat:
                [B, N]
        """
        heatmap = self.gaze_to_heatmap(
            frames=frames,
            gaze_locations=gaze_locations,
        )

        token_heat, token_labels = self.heatmap_to_tubelet_scores(
            heatmap=heatmap,
            token_count=token_count,
        )

        k = int(self.config.get("gaze_topk", 4))
        k = min(k, token_count)

        _, topk_indices = torch.topk(
            token_heat,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        )

        mask = torch.zeros_like(token_heat, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)

        topk_labels = token_labels[topk_indices]

        return mask, topk_indices, topk_labels, token_heat

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.model.student.parameters(), self.teacher.parameters()):
                t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)

    def on_after_optimizer_step(self, optimizer, optimizer_idx=None):
        self.update_teacher()

    def training_step(self, batch, batch_idx):
        img, actions, gaze_locations = format_batch_for_vjepa(
            batch,
            self.config,
        )

        if isinstance(img, list):
            img = torch.stack(img)

        context_frames = int(self.config["context_frames"])

        student_frames = img[:, :context_frames]
        teacher_frames = img[:, context_frames:]

        student_x = student_frames.unsqueeze(2)
        teacher_x = teacher_frames.unsqueeze(2)

        # --------------------------------------------------
        # Frozen student encoder
        # --------------------------------------------------
        with torch.no_grad():
            student_tokens = self.model.tubelet_embed(student_x)
            student_tokens = self.model.student(student_tokens)

        # --------------------------------------------------
        # Frozen teacher encoder
        # --------------------------------------------------
        with torch.no_grad():
            teacher_tokens = self.model.tubelet_embed(teacher_x)
            teacher_latents = self.teacher(teacher_tokens)

        if student_tokens.shape[1] != teacher_latents.shape[1]:
            raise RuntimeError(
                f"Student and teacher token counts must match for tubelet-level gaze loss. "
                f"Got student_tokens={student_tokens.shape}, "
                f"teacher_latents={teacher_latents.shape}. "
                f"With stack_length=8, context_frames=4, and tubelet_size=4, "
                f"both sides should usually produce 441 tokens if patchx=21 and patchy=21."
            )

        predictor_mask = None
        loss_mask = None

        # --------------------------------------------------
        # Gaze mask for predictor input
        # --------------------------------------------------
        if self.config.get("use_gaze_for_predictor", False):
            predictor_gaze = self._slice_gaze(
                gaze_locations,
                start=0,
                end=context_frames,
            )

            (
                predictor_mask,
                predictor_topk_indices,
                predictor_topk_labels,
                predictor_token_heat,
            ) = self.make_gaze_tubelet_mask(
                frames=student_frames,
                gaze_locations=predictor_gaze,
                token_count=student_tokens.shape[1],
            )

            self.last_predictor_gaze_indices = predictor_topk_indices.detach().cpu()
            self.last_predictor_gaze_labels = predictor_topk_labels.detach().cpu()

        # --------------------------------------------------
        # Gaze mask for loss

        if self.config.get("use_gaze_for_loss", False):
            loss_gaze = self._slice_gaze(
                gaze_locations,
                start=context_frames,
                end=img.shape[1],
            )

            (
                loss_mask,
                loss_topk_indices,
                loss_topk_labels,
                loss_token_heat,
            ) = self.make_gaze_tubelet_mask(
                frames=teacher_frames,
                gaze_locations=loss_gaze,
                token_count=teacher_latents.shape[1],
            )

            # --------------------------------------------------
            # Normalize Gaussian scores into loss weights
            # --------------------------------------------------
            eps = 1e-8

            loss_token_weights = loss_token_heat.float()
            loss_token_weights = loss_token_weights / (
                loss_token_weights.sum(dim=1, keepdim=True) + eps
            )

            self.last_loss_gaze_indices = loss_topk_indices.detach().cpu()
            self.last_loss_gaze_labels = loss_topk_labels.detach().cpu()
            self.last_loss_gaze_weights = loss_token_weights.detach().cpu()

        """   # --------------------------------------------------
        if self.config.get("use_gaze_for_loss", False):
            loss_gaze = self._slice_gaze(
                gaze_locations,
                start=context_frames,
                end=img.shape[1],
            )

            (
                loss_mask,
                loss_topk_indices,
                loss_topk_labels,
                loss_token_heat,
            ) = self.make_gaze_tubelet_mask(
                frames=teacher_frames,
                gaze_locations=loss_gaze,
                token_count=teacher_latents.shape[1],
            )

            self.last_loss_gaze_indices = loss_topk_indices.detach().cpu()
            self.last_loss_gaze_labels = loss_topk_labels.detach().cpu()
            """
        # --------------------------------------------------
        # Predictor input
        # --------------------------------------------------
        predictor_tokens = student_tokens

        if predictor_mask is not None:
            # Important:
            # Keep all tubelet positions, but zero out unselected ones.
            predictor_tokens = predictor_tokens * predictor_mask.unsqueeze(-1).to(
                predictor_tokens.dtype
            )

        # --------------------------------------------------
        # Action token
        # --------------------------------------------------
        last_action = actions[:, context_frames - 1]
        last_action = atari_to_gym(last_action)
        action_token = self.action_embed(last_action).unsqueeze(1)

        predictor_input = torch.cat(
            [
                predictor_tokens,
                action_token,
            ],
            dim=1,
        )

        # --------------------------------------------------
        # Predict future tubelet latents
        # --------------------------------------------------
        prediction = self.latent_predictor(
            predictor_input,
            predictor_input,
        )

        # Remove action-token output.
        # Shape becomes [B, N, D].
        prediction = prediction[:, :-1, :]

        target = teacher_latents

        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Prediction and target shape mismatch. "
                f"prediction={prediction.shape}, target={target.shape}"
            )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------
        if loss_mask is not None:
            per_token_loss = F.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
            ).mean(dim=-1)

            loss_mask_float = loss_mask.to(per_token_loss.dtype)

            loss = (
                per_token_loss * loss_mask_float
            ).sum() / loss_mask_float.sum().clamp_min(1.0)

        else:
            loss = F.smooth_l1_loss(
                prediction,
                target,
            )

        self.log(
            "loss/action_jepa",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.latent_predictor.parameters())
            + list(self.action_embed.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )


import pytorch_lightning as pl

"""


def training_step(self, batch, batch_idx):

    # --------------------------------------------------
    # Format batch
    # --------------------------------------------------
    img, actions, gaze_locations = format_batch_for_vjepa(
        batch,
        self.config,
    )

    if isinstance(img, list):
        img = torch.stack(img)

    context_frames = self.config["context_frames"]

    student_frames = img[:, :context_frames]
    teacher_frames = img[:, context_frames:]

    student_x = student_frames.unsqueeze(2)
    teacher_x = teacher_frames.unsqueeze(2)

    # --------------------------------------------------
    # Student encoder
    # --------------------------------------------------
    student_tokens = self.model.tubelet_embed(student_x)
    student_tokens = self.model.student(student_tokens)

    # --------------------------------------------------
    # Teacher encoder (EMA)
    # --------------------------------------------------
    with torch.no_grad():
        teacher_tokens = self.model.tubelet_embed(teacher_x)
        teacher_latents = self.teacher(teacher_tokens)

    # --------------------------------------------------
    # Compute gaze token indices (optional)
    # --------------------------------------------------
    gaze_indices = None

    if (
        self.config["use_gaze_for_predictor"]
        or
        self.config["use_gaze_for_loss"]
    ):

        heatmap = eye_gaze_to_density_image(
            student_frames.shape,
            gaze_locations,
            self.config,
        )

        gaze_indices = self.compute_gaze_indices(heatmap)

    # --------------------------------------------------
    # Predictor input
    # --------------------------------------------------
    predictor_tokens = student_tokens

    if (
        self.config["use_gaze_for_predictor"]
        and gaze_indices is not None
    ):
        predictor_tokens = self.gather_tokens(
            predictor_tokens,
            gaze_indices,
        )

    # --------------------------------------------------
    # Action token
    # --------------------------------------------------
    last_action = actions[:, context_frames - 1]

    action_token = self.action_embed(last_action)
    action_token = action_token.unsqueeze(1)

    predictor_input = torch.cat(
        [
            predictor_tokens,
            action_token,
        ],
        dim=1,
    )

    # --------------------------------------------------
    # Predict future tubelets
    # --------------------------------------------------
    prediction = self.latent_predictor(
        predictor_input,
        predictor_input,
    )

    # Remove the action token prediction
    prediction = prediction[:, :-1]

    # --------------------------------------------------
    # Build target
    # --------------------------------------------------
    target = teacher_latents

    if gaze_indices is not None:

        target = self.gather_tokens(
            target,
            gaze_indices,
        )

    # --------------------------------------------------
    # Compute loss
    # --------------------------------------------------
    loss = F.smooth_l1_loss(
        prediction,
        target,
    )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    self.log(
        "loss/action_jepa",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )

    return loss

"""


"""

class ActionConditionVJEPA(pl.LightningModule):
    def __init__(
        self,
        model,
        action_embed,
        config,
        latent_pred_dim=None,
        num_visible_frames=4,
        lr=1e-4,
        ema_decay=0.996,
    ):
        super().__init__()

        self.model = model
        self.config = config
        self.lr = lr
        self.ema_decay = ema_decay
        self.num_visible_frames = num_visible_frames

        # --------------------------------------------------
        # Load pretrained V-JEPA checkpoint
        # --------------------------------------------------
        ckpt_path = config["ckpt_path"]

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")

            student_weights = {
                k.replace("model.student.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.student.")
            }
            self.model.student.load_state_dict(student_weights)

            embed_weights = {
                k.replace("model.tubelet_embed.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.tubelet_embed.")
            }
            self.model.tubelet_embed.load_state_dict(embed_weights)

            print(f"Loaded pretrained checkpoint: {ckpt_path}")

        except FileNotFoundError:
            print("No pretrained checkpoint found.")

        # --------------------------------------------------
        # Freeze pretrained encoder
        # --------------------------------------------------
        for p in self.model.student.parameters():
            p.requires_grad = False

        for p in self.model.tubelet_embed.parameters():
            p.requires_grad = False

        # --------------------------------------------------
        # Teacher (EMA copy)
        # --------------------------------------------------
        self.teacher = copy.deepcopy(self.model.student)

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.teacher.load_state_dict(self.model.student.state_dict())

        # --------------------------------------------------
        # Action embedding
        # --------------------------------------------------
        self.action_embed = action_embed

        # --------------------------------------------------
        # Latent predictor
        # --------------------------------------------------
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

    def update_teacher(self):
        with torch.no_grad():
            for s, t in zip(
                self.model.student.parameters(),
                self.teacher.parameters(),
            ):
                t.data.mul_(self.ema_decay).add_(
                    s.data,
                    alpha=1.0 - self.ema_decay,
                )

    def on_after_optimizer_step(self, optimizer, optimizer_idx=None):
        self.update_teacher()

    def training_step(self, batch, batch_idx):

        # --------------------------------------------------
        # Format batch
        # --------------------------------------------------
        img, actions, gaze_locations = format_batch_for_vjepa(
            batch,
            self.config,
        )

        if isinstance(img, list):
            img = torch.stack(img)

        context_frames = self.config["context_frames"]

        student_frames = img[:, :context_frames]
        teacher_frames = img[:, context_frames:]

        student_x = student_frames.unsqueeze(2)
        teacher_x = teacher_frames.unsqueeze(2)

        # --------------------------------------------------
        # Student encoder (frozen)
        # --------------------------------------------------
        with torch.no_grad():
            student_tokens = self.model.tubelet_embed(student_x)
            student_tokens = self.model.student(student_tokens)

        # --------------------------------------------------
        # Teacher encoder
        # --------------------------------------------------
        with torch.no_grad():
            teacher_tokens = self.model.tubelet_embed(teacher_x)
            teacher_latents = self.teacher(teacher_tokens)

        # --------------------------------------------------
        # Eye gaze processing
        # --------------------------------------------------
        gaze_indices = None

        if (
            self.config["use_gaze_for_predictor"]
            or self.config["use_gaze_for_loss"]
        ):

            heatmap = eye_gaze_to_density_image(
                student_frames.shape,
                gaze_locations,
                self.config,
            )

            gaze_indices = self.compute_gaze_indices(
                heatmap
            )

        # --------------------------------------------------
        # Predictor input
        # --------------------------------------------------
        predictor_tokens = student_tokens

        if (
            self.config["use_gaze_for_predictor"]
            and gaze_indices is not None
        ):
            predictor_tokens = self.gather_tokens(
                predictor_tokens,
                gaze_indices,
            )

        # --------------------------------------------------
        # Action token
        # --------------------------------------------------
        last_action = actions[:, context_frames - 1]

        action_token = self.action_embed(
            last_action
        ).unsqueeze(1)

        predictor_input = torch.cat(
            [
                predictor_tokens,
                action_token,
            ],
            dim=1,
        )

        # --------------------------------------------------
        # Predict future latents
        # --------------------------------------------------
        prediction = self.latent_predictor(
            predictor_input,
            predictor_input,
        )

        # Remove prediction corresponding to action token
        prediction = prediction[:, :-1]

        # --------------------------------------------------
        # Build target
        # --------------------------------------------------
        target = teacher_latents

        if (
            self.config["use_gaze_for_loss"]
            and gaze_indices is not None
        ):
            target = self.gather_tokens(
                target,
                gaze_indices,
            )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------
        loss = F.smooth_l1_loss(
            prediction,
            target,
        )

        self.log(
            "loss/action_jepa",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.latent_predictor.parameters())
            + list(self.action_embed.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )
"""
