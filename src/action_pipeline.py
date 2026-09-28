import os
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from models.vjepa import (
    ActionEmbedding,
    TransformerEncoder,
    TubeletEmbedding,
    VJEPAEncoder,
)
from trainers.jepa import ActionConditionVJEPA
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


# ============================================================
# Atari Environment Wrapper
# ============================================================
import sys
from collections import deque

import ale_py  # ⚠️ force registration of ALE namespace

# from config import *
# from gym_manager import GymManager

# from dataset.pre_process import Resize, StackWithLabels
# from models.vjepa import ActionEmbedding, TransformerEncoder, TubeletEmbedding

# import gym

print("Python version:", sys.version)
print("Gym version:", gym.__version__)
# import gymnasium.atari

# Dummy usage to prevent deletion
_ = ale_py.__name__

print("Python version:", sys.version)
print("Gym version:", gym.__version__)


class Resize:
    def __init__(self, config):
        self.size = 84

    def __call__(self, sample):
        obs, _, _ = sample

        obs = torch.tensor(obs, dtype=torch.float32) / 255.0

        if obs.ndim == 3:
            obs = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]

        obs = obs.unsqueeze(0).unsqueeze(0)
        obs = F.interpolate(obs, size=(self.size, self.size), mode="bilinear")
        obs = obs.squeeze(0)

        return obs, None, None


class StackWithLabels:
    def __init__(self, stack_size=4):
        self.frames = deque(maxlen=stack_size)
        self.stack_size = stack_size

    def reset(self, first_frame):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(first_frame)
        return self._get_stack()

    def step(self, frame):
        self.frames.append(frame)
        return self._get_stack()

    def _get_stack(self):
        return torch.stack(list(self.frames), dim=0)
        # return torch.cat(list(self.frames), dim=0)


class RuntimePreprocessor:
    def __init__(self, config):
        self.resize = Resize(config)
        self.stack = StackWithLabels(4)

    def reset(self, obs):
        obs, _, _ = self.resize((obs, None, None))
        return self.stack.reset(obs)

    def step(self, obs):
        obs, _, _ = self.resize((obs, None, None))
        return self.stack.step(obs)


class GymManager:
    """
    Online Atari environment.

    reset() returns the preprocessed 4-frame stack.
    step(action) returns:
        next_preprocessed_stack, reward, done
    """

    def __init__(
        self,
        config,
        env_name="ALE/MsPacman-v5",
    ):
        self.env = gym.make(env_name)
        self.preprocessor = RuntimePreprocessor(config)

    def reset(self):
        obs, _ = self.env.reset()
        return self.preprocessor.reset(obs)

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(int(action))
        done = terminated or truncated
        state = self.preprocessor.step(obs)
        return state, float(reward), bool(done)

    @property
    def num_actions(self):
        return self.env.action_space.n

    def close(self):
        self.env.close()


# ============================================================
# Utility Helpers
# ============================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(config):
    requested = config.get("score_delta_device", None)

    if requested is not None:
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    return torch.device("cpu")


def get_amp_settings(config, device):
    """
    Uses bf16 when available, otherwise fp16 on CUDA.
    Falls back to fp32 on CPU.
    """

    if device.type != "cuda":
        return False, torch.float32, None

    precision = config.get("score_delta_precision", "bf16")

    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, None

    if precision in ["bf16", "fp16", "16", "mixed"]:
        scaler = torch.cuda.amp.GradScaler()
        return True, torch.float16, scaler

    return False, torch.float32, None


def safe_float_tensor(x, device):
    x = torch.as_tensor(x)

    if not torch.is_floating_point(x):
        x = x.float()

    # If the preprocessor already normalizes, this will do nothing useful.
    # If it gives uint8-like 0..255 tensors, this normalizes them.
    if x.numel() > 0 and x.max().item() > 2.0:
        x = x / 255.0

    return x.to(device=device, dtype=torch.float32)


def prepare_state_batch(states, config, device):
    """
    Converts a list of preprocessed frame stacks into a model-ready tensor.

    Supported common shapes per state:
        [T, H, W]
        [T, C, H, W]
        [H, W, T]

    Returns:
        Tensor with shape either:
            [B, T, H, W]
        or:
            [B, T, C, H, W]
    """

    context_frames = int(config.get("context_frames", 4))
    processed = []

    for state in states:
        x = safe_float_tensor(state, device=device)

        if x.ndim == 2:
            # Single image; repeat as context.
            x = x.unsqueeze(0).repeat(context_frames, 1, 1)

        elif x.ndim == 3:
            # Either [T, H, W] or [H, W, T].
            if x.shape[0] == context_frames:
                pass
            elif x.shape[-1] == context_frames:
                x = x.permute(2, 0, 1).contiguous()
            else:
                raise ValueError(
                    f"Could not infer 3D state shape {tuple(x.shape)}. "
                    f"Expected [T,H,W] or [H,W,T] with T={context_frames}."
                )

        elif x.ndim == 4:
            # Usually [T, C, H, W].
            if x.shape[0] == context_frames:
                pass
            else:
                raise ValueError(
                    f"Could not infer 4D state shape {tuple(x.shape)}. "
                    f"Expected [T,C,H,W] with T={context_frames}."
                )

        else:
            raise ValueError(
                f"Unsupported state shape {tuple(x.shape)} from GymManager."
            )

        processed.append(x)

    return torch.stack(processed, dim=0)


def clean_checkpoint_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("teacher."):
            continue

        key = key.removeprefix("module.")

        cleaned[key] = value

    return cleaned


def load_matching_weights(model, checkpoint_path):
    """
    Loads only matching checkpoint weights.

    This lets you load your previous ActionConditionVJEPA checkpoint
    while ignoring the new score_predictor weights.
    """

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = clean_checkpoint_state_dict(state_dict)

    model_state = model.state_dict()

    loadable = {}
    skipped = []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            loadable[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    print("Loaded checkpoint:", checkpoint_path)
    print("Loaded keys:", len(loadable))
    print("Skipped keys:", len(skipped))
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return missing, unexpected, skipped


def save_score_delta_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    global_step,
    episode_count,
    config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "global_step": global_step,
        "episode_count": episode_count,
        "score_predictor": model.score_predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": dict(config),
    }

    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    torch.save(payload, path)


def load_score_delta_checkpoint(
    path,
    model,
    optimizer=None,
    scaler=None,
    device="cpu",
):
    ckpt = torch.load(path, map_location=device)

    model.score_predictor.load_state_dict(ckpt["score_predictor"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    global_step = int(ckpt.get("global_step", 0))
    episode_count = int(ckpt.get("episode_count", 0))

    print("Resumed score-delta checkpoint:", path)
    print("Resume global_step:", global_step)
    print("Resume episode_count:", episode_count)

    return global_step, episode_count


# ============================================================
# Full-token Score Delta Predictor
# ============================================================


class ScoreDeltaPredictor(nn.Module):
    """
    Predicts normalized score delta:

        delta_score = future_score - current_score

    Inputs:
        current_latent : [B, N, D]
        future_latent  : [B, N, D]
        current_score  : [B]

    Important:
        The action list is NOT input to this network.

    This model does not pool the tubelet tokens before processing.
    It attends over all current tokens, future tokens, and difference tokens.
    """

    def __init__(
        self,
        embed_dim=768,
        heads=12,
        depth=6,
        mlp_dim=3072,
        dropout=0.0,
        score_norm=10000.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.score_norm = float(score_norm)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Types:
        # 0 = cls
        # 1 = score
        # 2 = current latent
        # 3 = future latent
        # 4 = latent difference
        self.type_embed = nn.Parameter(torch.zeros(5, embed_dim))

        self.score_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.current_norm = nn.LayerNorm(embed_dim)
        self.future_norm = nn.LayerNorm(embed_dim)
        self.diff_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.out_norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(
        self,
        current_latent,
        future_latent,
        current_score,
    ):
        if current_latent.ndim != 3:
            raise ValueError(
                f"current_latent must be [B,N,D], got {tuple(current_latent.shape)}"
            )

        if future_latent.ndim != 3:
            raise ValueError(
                f"future_latent must be [B,N,D], got {tuple(future_latent.shape)}"
            )

        if current_latent.shape != future_latent.shape:
            raise ValueError(
                f"current_latent and future_latent must have same shape, got "
                f"{tuple(current_latent.shape)} and {tuple(future_latent.shape)}"
            )

        batch_size = current_latent.shape[0]

        current = self.current_norm(current_latent)
        future = self.future_norm(future_latent)
        diff = self.diff_norm(future_latent - current_latent)

        score = current_score.float().view(batch_size, 1)
        score = score / self.score_norm
        score_token = self.score_embed(score).unsqueeze(1)

        cls = self.cls_token.expand(batch_size, -1, -1)

        cls = cls + self.type_embed[0].view(1, 1, -1)
        score_token = score_token + self.type_embed[1].view(1, 1, -1)
        current = current + self.type_embed[2].view(1, 1, -1)
        future = future + self.type_embed[3].view(1, 1, -1)
        diff = diff + self.type_embed[4].view(1, 1, -1)

        tokens = torch.cat(
            [
                cls,
                score_token,
                current,
                future,
                diff,
            ],
            dim=1,
        )

        tokens = self.encoder(tokens)

        cls_out = tokens[:, 0]
        cls_out = self.out_norm(cls_out)

        delta = self.head(cls_out).squeeze(-1)

        return delta


# ============================================================
# Latent Rollout + Score Model
# ============================================================


class OnlineScoreDeltaJEPA(nn.Module):
    """
    Wraps your pretrained JEPA world model and adds a trainable
    score-delta predictor.

    Frozen:
        model.tubelet_embed
        model.student
        action_embed
        latent_predictor

    Trainable:
        score_predictor
    """

    def __init__(
        self,
        base_model,
        config,
        embed_dim=768,
    ):
        super().__init__()

        self.model = base_model.model
        self.action_embed = base_model.action_embed
        self.latent_predictor = base_model.latent_predictor

        self.config = config
        self.context_frames = int(config.get("context_frames", 4))
        self.cycle_steps = int(
            config.get("cycle_steps", config.get("rollout_steps", 4))
        )

        score_depth = int(config.get("score_delta_depth", 6))
        score_heads = int(config.get("score_delta_heads", 12))
        score_mlp_dim = int(config.get("score_delta_mlp_dim", 3072))
        score_dropout = float(config.get("score_delta_dropout", 0.0))
        score_norm = float(config.get("score_norm", 10000.0))

        self.score_delta_norm = float(config.get("score_delta_norm", 100.0))

        self.score_predictor = ScoreDeltaPredictor(
            embed_dim=embed_dim,
            heads=score_heads,
            depth=score_depth,
            mlp_dim=score_mlp_dim,
            dropout=score_dropout,
            score_norm=score_norm,
        )

        self.freeze_world_model()

    def freeze_world_model(self):
        self.model.tubelet_embed.requires_grad_(False)
        self.model.student.requires_grad_(False)
        self.action_embed.requires_grad_(False)
        self.latent_predictor.requires_grad_(False)

        self.model.tubelet_embed.eval()
        self.model.student.eval()
        self.action_embed.eval()
        self.latent_predictor.eval()

    def train(self, mode=True):
        super().train(mode)

        # Keep frozen parts in eval mode even when the wrapper is in train mode.
        self.model.tubelet_embed.eval()
        self.model.student.eval()
        self.action_embed.eval()
        self.latent_predictor.eval()

        return self

    @torch.no_grad()
    def encode_current(self, frames):
        """
        frames can be:
            [B, T, H, W]
            [B, T, C, H, W]

        Returns:
            current_latent [B, N, D]
        """

        if frames.ndim == 4:
            # [B, T, H, W] -> [B, T, 1, H, W]
            context = frames[:, : self.context_frames].unsqueeze(2)

        elif frames.ndim == 5:
            # [B, T, C, H, W]
            context = frames[:, : self.context_frames]

        else:
            raise ValueError(
                f"Expected frames [B,T,H,W] or [B,T,C,H,W], got {tuple(frames.shape)}"
            )

        z = self.model.tubelet_embed(context)
        z = self.model.student(z)

        return z

    @torch.no_grad()
    def predict_next_latent(self, latent, action):
        """
        One JEPA latent transition.

        latent : [B, N, D]
        action : [B]
        """

        action_token = self.action_embed(action)

        if action_token.ndim > 2:
            action_token = action_token.mean(dim=tuple(range(1, action_token.ndim - 1)))

        action_token = action_token.unsqueeze(1)

        sequence = torch.cat(
            [
                latent,
                action_token,
            ],
            dim=1,
        )

        out = self.latent_predictor(sequence, sequence)

        return out[:, :-1, :]

    @torch.no_grad()
    def rollout_latent(self, current_latent, actions):
        """
        Rolls current_latent forward cycle_steps using action sequence.

        current_latent : [B, N, D]
        actions        : [B, cycle_steps]

        Returns:
            future_latent [B, N, D]
        """

        latent = current_latent

        steps = min(actions.shape[1], self.cycle_steps)

        for t in range(steps):
            latent = self.predict_next_latent(
                latent,
                actions[:, t],
            )

        return latent

    def predict_normalized_delta(
        self,
        current_latent,
        future_latent,
        current_score,
    ):
        return self.score_predictor(
            current_latent,
            future_latent,
            current_score,
        )

    def predict_score_delta(
        self,
        current_latent,
        future_latent,
        current_score,
    ):
        normalized_delta = self.predict_normalized_delta(
            current_latent,
            future_latent,
            current_score,
        )

        return normalized_delta * self.score_delta_norm


# ============================================================
# Online Action Sampler
# ============================================================


class RandomActionSequenceSampler:
    """
    Generates random action sequences.

    Later you can replace this with:
        - policy network
        - planner
        - MPC search
        - CEM search
        - human demonstrations
    """

    def __init__(self, num_actions):
        self.num_actions = int(num_actions)

    def sample(self, batch_size, cycle_steps):
        return np.random.randint(
            low=0,
            high=self.num_actions,
            size=(batch_size, cycle_steps),
            dtype=np.int64,
        )


# ============================================================
# Online Training Loop
# ============================================================


def train_online_score_delta(
    model,
    config,
    env_name,
    device,
    writer,
    start_global_step=0,
    start_episode_count=0,
    scaler=None,
):
    model.to(device)
    model.train()

    cycle_steps = int(config.get("cycle_steps", config.get("rollout_steps", 4)))
    num_envs = int(config.get("score_delta_num_envs", 16))
    total_updates = int(config.get("score_delta_updates", 1_000_000))
    lr = float(config.get("score_delta_lr", 1e-4))
    weight_decay = float(config.get("score_delta_weight_decay", 1e-4))
    grad_clip = float(config.get("score_delta_grad_clip", 1.0))

    log_every = int(config.get("score_delta_log_every", 10))
    save_every = int(config.get("score_delta_save_every", 5000))
    checkpoint_dir = Path(
        config.get(
            "score_delta_checkpoint_dir",
            f"tb_logs/{env_name}/vjepa_score_delta_online/checkpoints",
        )
    )

    skip_terminal_rollouts = bool(
        config.get("score_delta_skip_terminal_rollouts", True)
    )

    use_amp, amp_dtype, local_scaler = get_amp_settings(config, device)

    if scaler is None:
        scaler = local_scaler

    optimizer = torch.optim.AdamW(
        model.score_predictor.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25_000,
        gamma=0.95,
    )

    envs = [GymManager(config, env_name=env_name) for _ in range(num_envs)]

    action_sampler = RandomActionSequenceSampler(envs[0].num_actions)

    states = []
    current_scores = []
    episode_returns = []
    episode_lengths = []

    for env in envs:
        states.append(env.reset())
        current_scores.append(0.0)
        episode_returns.append(0.0)
        episode_lengths.append(0)

    global_step = int(start_global_step)
    episode_count = int(start_episode_count)
    env_step_count = 0

    running_loss = 0.0
    running_abs_error = 0.0
    running_pred_delta = 0.0
    running_actual_delta = 0.0
    running_valid_fraction = 0.0

    start_time = time.time()

    try:
        for update_idx in range(total_updates):
            model.train()

            # ------------------------------------------------
            # Build online batch from current environment states
            # ------------------------------------------------

            batch_states = prepare_state_batch(
                states,
                config=config,
                device=device,
            )

            batch_scores = torch.tensor(
                current_scores,
                device=device,
                dtype=torch.float32,
            )

            action_sequences_np = action_sampler.sample(
                batch_size=num_envs,
                cycle_steps=cycle_steps,
            )

            action_sequences = torch.tensor(
                action_sequences_np,
                device=device,
                dtype=torch.long,
            )

            # ------------------------------------------------
            # Frozen JEPA:
            # current frames -> current latent
            # current latent + actions -> predicted future latent
            # ------------------------------------------------

            with torch.no_grad():
                current_latent = model.encode_current(batch_states)
                future_latent = model.rollout_latent(
                    current_latent,
                    action_sequences,
                )

                current_latent = current_latent.detach()
                future_latent = future_latent.detach()

            # ------------------------------------------------
            # Trainable score predictor
            # ------------------------------------------------

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                predicted_delta_normalized = model.predict_normalized_delta(
                    current_latent=current_latent,
                    future_latent=future_latent,
                    current_score=batch_scores,
                )

                predicted_delta_normalized = predicted_delta_normalized.squeeze(-1)

            ####################################################################################################
            # #################################################################################################
            # ------------------------------------------------
            # Execute the same actions in the real environment
            # to get the true score delta.
            # ------------------------------------------------

            actual_deltas = []
            valid_mask = []

            for env_idx, env in enumerate(envs):
                score_before = current_scores[env_idx]
                rollout_delta = 0.0
                done_happened = False
                next_state = states[env_idx]

                for action in action_sequences_np[env_idx]:
                    next_state, reward, done = env.step(int(action))

                    rollout_delta += reward
                    current_scores[env_idx] += reward
                    episode_returns[env_idx] += reward
                    episode_lengths[env_idx] += 1
                    env_step_count += 1

                    if done:
                        done_happened = True

                        writer.add_scalar(
                            "episode/return",
                            episode_returns[env_idx],
                            episode_count,
                        )

                        writer.add_scalar(
                            "episode/length",
                            episode_lengths[env_idx],
                            episode_count,
                        )

                        episode_count += 1

                        next_state = env.reset()
                        current_scores[env_idx] = 0.0
                        episode_returns[env_idx] = 0.0
                        episode_lengths[env_idx] = 0

                        break

                states[env_idx] = next_state

                actual_delta = current_scores[env_idx] - score_before

                # If the episode terminated and reset, current_scores is 0 now,
                # so use the accumulated rollout_delta instead.
                if done_happened:
                    actual_delta = rollout_delta

                actual_deltas.append(actual_delta)

                if skip_terminal_rollouts and done_happened:
                    valid_mask.append(False)
                else:
                    valid_mask.append(True)

            actual_delta = torch.tensor(
                actual_deltas,
                device=device,
                dtype=torch.float32,
            )

            target_delta_normalized = actual_delta / model.score_delta_norm
            if predicted_delta_normalized.shape != target_delta_normalized.shape:
                raise RuntimeError(
                    "Prediction and target shapes do not match: "
                    f"prediction={predicted_delta_normalized.shape}, "
                    f"target={target_delta_normalized.shape}"
                )
            valid_mask = torch.tensor(
                valid_mask,
                device=device,
                dtype=torch.bool,
            )

            if valid_mask.any():
                predicted_valid = predicted_delta_normalized[valid_mask]
                target_valid = target_delta_normalized[valid_mask]

                loss = F.smooth_l1_loss(
                    predicted_valid.float(),
                    target_valid.float(),
                )

                if scaler is not None:
                    scaler.scale(loss).backward()

                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.score_predictor.parameters(),
                            grad_clip,
                        )

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    loss.backward()

                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.score_predictor.parameters(),
                            grad_clip,
                        )

                    optimizer.step()
                scheduler.step()

                predicted_delta = (
                    predicted_delta_normalized.detach() * model.score_delta_norm
                )

                abs_error = torch.abs(predicted_delta.float() - actual_delta.float())

                valid_fraction = valid_mask.float().mean().item()

                loss_value = loss.detach().item()
                abs_error_value = abs_error[valid_mask].mean().item()
                pred_value = predicted_delta[valid_mask].mean().item()
                actual_value = actual_delta[valid_mask].mean().item()

            else:
                loss_value = 0.0
                abs_error_value = 0.0
                pred_value = 0.0
                actual_value = 0.0
                valid_fraction = 0.0

            global_step += 1
            if global_step % 1000 == 0:
                print(
                    "prediction shape:",
                    predicted_delta_normalized.shape,
                    "target shape:",
                    target_delta_normalized.shape,
                    "prediction std:",
                    predicted_delta_normalized.std().item(),
                    "target std:",
                    target_delta_normalized.std().item(),
                )
            running_loss += loss_value
            running_abs_error += abs_error_value
            running_pred_delta += pred_value
            running_actual_delta += actual_value
            running_valid_fraction += valid_fraction

            # ------------------------------------------------
            # TensorBoard logging
            # ------------------------------------------------

            if global_step % log_every == 0:
                denom = float(log_every)

                elapsed = max(time.time() - start_time, 1e-6)
                env_steps_per_second = env_step_count / elapsed

                writer.add_scalar(
                    "train/loss",
                    running_loss / denom,
                    global_step,
                )

                writer.add_scalar(
                    "train/abs_score_delta_error",
                    running_abs_error / denom,
                    global_step,
                )

                writer.add_scalar(
                    "train/predicted_score_delta",
                    running_pred_delta / denom,
                    global_step,
                )

                writer.add_scalar(
                    "train/actual_score_delta",
                    running_actual_delta / denom,
                    global_step,
                )

                writer.add_scalar(
                    "train/valid_fraction",
                    running_valid_fraction / denom,
                    global_step,
                )

                writer.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )

                writer.add_scalar(
                    "system/env_steps_per_second",
                    env_steps_per_second,
                    global_step,
                )

                writer.add_scalar(
                    "system/env_step_count",
                    env_step_count,
                    global_step,
                )

                writer.add_scalar(
                    "system/episode_count",
                    episode_count,
                    global_step,
                )

                print(
                    f"[step {global_step}] "
                    f"loss={running_loss / denom:.6f} "
                    f"abs_err={running_abs_error / denom:.3f} "
                    f"pred_delta={running_pred_delta / denom:.3f} "
                    f"actual_delta={running_actual_delta / denom:.3f} "
                    f"valid={running_valid_fraction / denom:.3f} "
                    f"env_sps={env_steps_per_second:.1f}"
                )

                running_loss = 0.0
                running_abs_error = 0.0
                running_pred_delta = 0.0
                running_actual_delta = 0.0
                running_valid_fraction = 0.0

            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------

            if global_step % save_every == 0:
                checkpoint_path = checkpoint_dir / f"score_delta_step={global_step}.pt"

                save_score_delta_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=global_step,
                    episode_count=episode_count,
                    config=config,
                )

                latest_path = checkpoint_dir / "latest.pt"

                save_score_delta_checkpoint(
                    path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    global_step=global_step,
                    episode_count=episode_count,
                    config=config,
                )

                print("Saved:", checkpoint_path)

    finally:
        latest_path = checkpoint_dir / "latest.pt"

        save_score_delta_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            global_step=global_step,
            episode_count=episode_count,
            config=config,
        )

        print("Saved final latest checkpoint:", latest_path)

        for env in envs:
            env.close()

        writer.flush()
        writer.close()


# ============================================================
# Main Online Score-Delta Training Block
# ============================================================

with skip_run("run", "online_jepa_score_delta_trainer") as check, check():
    game = config["games"][0]

    env_name = config.get("env_name", "ALE/MsPacman-v5")

    seed = int(config.get("seed", 123))
    set_seed(seed)

    device = select_device(config)

    print("Using device:", device)
    print("Game:", game)
    print("Env:", env_name)

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    log_dir = config.get(
        "score_delta_log_dir",
        f"tb_logs/{game}/vjepa_score_delta_online",
    )

    writer = SummaryWriter(log_dir=log_dir)

    print("TensorBoard log dir:", log_dir)

    # --------------------------------------------------------
    # Build pretrained JEPA world model
    # --------------------------------------------------------

    patch_dim = 1 if config.get("grey_scale_v", True) else 3
    embed_dim = int(config.get("embed_dim", 768))
    heads = int(config.get("heads", 12))
    mlp_dim = int(config.get("mlp_dim", 3072))
    depth = int(config.get("encoder_depth", 12))

    tubelet_embed = TubeletEmbedding(
        config=config,
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        img_size=config.get("size_x", 84),
    )

    student = TransformerEncoder(
        embed_dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    net = VJEPAEncoder(
        tubelet_embed=tubelet_embed,
        student=student,
    )

    action_embed = ActionEmbedding()

    base_model = ActionConditionVJEPA(
        model=net,
        action_embed=action_embed,
        config=config,
        lr=1e-4,
        ema_decay=0.996,
    )

    model = OnlineScoreDeltaJEPA(
        base_model=base_model,
        config=config,
        embed_dim=embed_dim,
    )

    # --------------------------------------------------------
    # Load pretrained JEPA checkpoint
    # --------------------------------------------------------
    """
    pretrained_checkpoint = config.get(
        "pretrained_jepa_checkpoint",
        "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_action_world_model/version_4/checkpoints/epoch=49-step=164350.ckpt",
    )
    """
    pretrained_checkpoint = "tb_logs/ms_pacman/vjepa_rollout_world_model/version_2/checkpoints/epoch=49-step=164350.ckpt"
    load_matching_weights(
        model=model,
        checkpoint_path=pretrained_checkpoint,
    )

    model.freeze_world_model()

    # --------------------------------------------------------
    # Optional resume for score-delta model
    # --------------------------------------------------------

    optimizer_dummy = None
    scaler_dummy = None

    start_global_step = 0
    start_episode_count = 0

    resume_path = config.get("score_delta_resume_checkpoint", None)

    if resume_path is not None and os.path.exists(resume_path):
        # Optimizer is created inside train_online_score_delta.
        # This resume path only restores score_predictor here.
        # If you want optimizer resume too, move optimizer creation outside.
        ckpt = torch.load(resume_path, map_location=device)
        model.score_predictor.load_state_dict(ckpt["score_predictor"])
        start_global_step = int(ckpt.get("global_step", 0))
        start_episode_count = int(ckpt.get("episode_count", 0))

        print("Resumed score predictor from:", resume_path)
        print("Starting global step:", start_global_step)
        print("Starting episode count:", start_episode_count)

    # --------------------------------------------------------
    # Train online from Atari
    # --------------------------------------------------------

    train_online_score_delta(
        model=model,
        config=config,
        env_name=env_name,
        device=device,
        writer=writer,
        start_global_step=start_global_step,
        start_episode_count=start_episode_count,
    )


##################################################################################################
#####################################################################################################
##########################################################################################################


import torch


class CEMActionPlanner:
    """
    Cross-Entropy Method planner for discrete Atari actions.

    It generates action lists for your JEPA pipeline.

    It does NOT train the JEPA model.
    It does NOT train the reward model.
    It searches for high-reward action sequences using the frozen JEPA rollout
    and the trained score-delta predictor.

    Required model methods:
        model.encode_current(frames)
        model.rollout_latent(current_latent, actions)
        model.predict_score_delta(current_latent, future_latent, current_score)

    Expected action shape:
        actions: [num_candidates, horizon]
    """

    def __init__(
        self,
        num_actions,
        horizon=8,
        num_candidates=512,
        cem_iterations=4,
        elite_fraction=0.1,
        eval_batch_size=128,
        momentum=0.25,
        min_prob=0.01,
        temperature=1.0,
        device="cuda",
        warm_start=False,
    ):
        self.num_actions = int(num_actions)
        self.horizon = int(horizon)
        self.num_candidates = int(num_candidates)
        self.cem_iterations = int(cem_iterations)
        self.elite_fraction = float(elite_fraction)
        self.eval_batch_size = int(eval_batch_size)
        self.momentum = float(momentum)
        self.min_prob = float(min_prob)
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self.warm_start = bool(warm_start)

        self.num_elites = max(
            1,
            int(self.num_candidates * self.elite_fraction),
        )

        self.prior_probs = None

    def reset(self):
        self.prior_probs = None

    def _initial_probs(self):
        if self.warm_start and self.prior_probs is not None:
            probs = self.prior_probs.clone()
        else:
            probs = torch.full(
                (self.horizon, self.num_actions),
                1.0 / self.num_actions,
                device=self.device,
            )

        return probs

    def _sample_sequences(self, probs):
        """
        probs: [horizon, num_actions]

        returns:
            action_sequences: [num_candidates, horizon]
        """

        sampled = []

        for t in range(self.horizon):
            actions_t = torch.multinomial(
                probs[t],
                num_samples=self.num_candidates,
                replacement=True,
            )
            sampled.append(actions_t)

        return torch.stack(sampled, dim=1)

    def _evaluate_sequences(
        self,
        model,
        current_latent,
        current_score,
        action_sequences,
    ):
        """
        Evaluates candidate action sequences using:

            current latent
            + candidate action sequence
            + JEPA latent rollout
            + score-delta predictor

        Returns:
            predicted_delta: [num_candidates]
        """

        values = []

        current_score_tensor = torch.as_tensor(
            current_score,
            device=self.device,
            dtype=torch.float32,
        ).view(1)

        for chunk in action_sequences.split(self.eval_batch_size, dim=0):
            batch_size = chunk.shape[0]

            latent_batch = current_latent.expand(
                batch_size,
                -1,
                -1,
            ).contiguous()

            score_batch = current_score_tensor.expand(batch_size)

            future_latent = model.rollout_latent(
                latent_batch,
                chunk,
            )

            predicted_delta = model.predict_score_delta(
                current_latent=latent_batch,
                future_latent=future_latent,
                current_score=score_batch,
            )

            values.append(predicted_delta.detach().float())

        return torch.cat(values, dim=0)

    def _update_probs_from_elites(self, probs, elite_sequences):
        """
        elite_sequences: [num_elites, horizon]
        """

        one_hot = F.one_hot(
            elite_sequences,
            num_classes=self.num_actions,
        ).float()

        elite_probs = one_hot.mean(dim=0)

        elite_probs = elite_probs.clamp_min(self.min_prob)
        elite_probs = elite_probs / elite_probs.sum(
            dim=-1,
            keepdim=True,
        )

        new_probs = self.momentum * probs + (1.0 - self.momentum) * elite_probs

        new_probs = new_probs.clamp_min(self.min_prob)
        new_probs = new_probs / new_probs.sum(
            dim=-1,
            keepdim=True,
        )

        return new_probs

    def _warm_start_next_step(self, probs):
        """
        After choosing one action, shift the probability plan forward.

        Example:
            old:
                t0, t1, t2, t3

            new:
                t1, t2, t3, uniform
        """

        if not self.warm_start:
            self.prior_probs = None
            return

        uniform = torch.full(
            (1, self.num_actions),
            1.0 / self.num_actions,
            device=self.device,
        )

        self.prior_probs = torch.cat(
            [
                probs[1:].detach(),
                uniform,
            ],
            dim=0,
        )

    @torch.inference_mode()
    def plan(
        self,
        model,
        current_latent,
        current_score,
    ):
        """
        Returns:
            first_action: int
            best_sequence: torch.LongTensor [horizon]
            best_value: float
            final_probs: torch.Tensor [horizon, num_actions]
        """

        model.eval()

        current_latent = current_latent.to(self.device)

        probs = self._initial_probs()

        best_sequence = None
        best_value = None

        for _ in range(self.cem_iterations):
            action_sequences = self._sample_sequences(probs)

            values = self._evaluate_sequences(
                model=model,
                current_latent=current_latent,
                current_score=current_score,
                action_sequences=action_sequences,
            )

            if self.temperature != 1.0:
                values_for_selection = values / self.temperature
            else:
                values_for_selection = values

            elite_indices = torch.topk(
                values_for_selection,
                k=self.num_elites,
                dim=0,
            ).indices

            elite_sequences = action_sequences[elite_indices]

            probs = self._update_probs_from_elites(
                probs,
                elite_sequences,
            )

            max_idx = torch.argmax(values)

            if best_value is None or values[max_idx].item() > best_value:
                best_value = values[max_idx].item()
                best_sequence = action_sequences[max_idx].detach().clone()

        first_action = int(best_sequence[0].item())

        self._warm_start_next_step(probs)

        return first_action, best_sequence, best_value, probs


# ============================================================
# Planner Episode Runner
# ============================================================


def run_planner_episode(
    model,
    gym_manager,
    config,
    device,
    writer=None,
    global_step_start=0,
    max_steps=None,
):
    """
    Runs one Atari episode using:

        current state
            -> JEPA encoder
            -> CEM action planner
            -> JEPA latent rollout
            -> score-delta model
            -> selected real Atari action

    Returns:
        episode_return, episode_length, global_step
    """

    model.to(device)
    model.eval()

    planner = CEMActionPlanner(
        num_actions=gym_manager.num_actions,
        horizon=config.get("planner_horizon", config.get("cycle_steps", 8)),
        num_candidates=config.get("planner_num_candidates", 512),
        cem_iterations=config.get("planner_cem_iterations", 4),
        elite_fraction=config.get("planner_elite_fraction", 0.1),
        eval_batch_size=config.get("planner_eval_batch_size", 128),
        momentum=config.get("planner_momentum", 0.25),
        min_prob=config.get("planner_min_prob", 0.01),
        temperature=config.get("planner_temperature", 1.0),
        device=device,
        warm_start=True,
    )

    state = gym_manager.reset()

    current_score = 0.0
    episode_return = 0.0
    episode_length = 0
    global_step = int(global_step_start)

    done = False

    while not done:
        if max_steps is not None and episode_length >= max_steps:
            break

        frames = prepare_state_batch(
            [state],
            config=config,
            device=device,
        )

        with torch.no_grad():
            current_latent = model.encode_current(frames)

            action, planned_sequence, predicted_delta, action_probs = planner.plan(
                model=model,
                current_latent=current_latent,
                current_score=current_score,
            )

        next_state, reward, done = gym_manager.step(action)

        current_score += reward
        episode_return += reward
        episode_length += 1
        global_step += 1

        if writer is not None:
            writer.add_scalar(
                "planner/reward",
                reward,
                global_step,
            )

            writer.add_scalar(
                "planner/current_score",
                current_score,
                global_step,
            )

            writer.add_scalar(
                "planner/predicted_delta",
                predicted_delta,
                global_step,
            )

            writer.add_scalar(
                "planner/chosen_action",
                action,
                global_step,
            )

            entropy = (
                torch.distributions.Categorical(probs=action_probs[0].detach().cpu())
                .entropy()
                .item()
            )

            writer.add_scalar(
                "planner/action_entropy_t0",
                entropy,
                global_step,
            )

            writer.add_text(
                "planner/planned_sequence",
                str(planned_sequence.detach().cpu().tolist()),
                global_step,
            )

        state = next_state

    if writer is not None:
        writer.add_scalar(
            "planner/episode_return",
            episode_return,
            global_step,
        )

        writer.add_scalar(
            "planner/episode_length",
            episode_length,
            global_step,
        )

    return episode_return, episode_length, global_step


# ============================================================
# Build Model for Planner Evaluation
# ============================================================


def build_score_delta_model_for_planning(config, device):
    """
    Rebuilds the JEPA + score-delta model for planner use.

    This is needed because your training block may be skipped, so the
    variables `model` and `writer` may not exist at the bottom of the file.
    """

    patch_dim = 1 if config.get("grey_scale_v", True) else 3
    embed_dim = int(config.get("embed_dim", 768))
    heads = int(config.get("heads", 12))
    mlp_dim = int(config.get("mlp_dim", 3072))
    depth = int(config.get("encoder_depth", 12))

    tubelet_embed = TubeletEmbedding(
        config=config,
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        img_size=config.get("size_x", 84),
    )

    student = TransformerEncoder(
        embed_dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    net = VJEPAEncoder(
        tubelet_embed=tubelet_embed,
        student=student,
    )

    action_embed = ActionEmbedding()

    base_model = ActionConditionVJEPA(
        model=net,
        action_embed=action_embed,
        config=config,
        lr=1e-4,
        ema_decay=0.996,
    )

    model = OnlineScoreDeltaJEPA(
        base_model=base_model,
        config=config,
        embed_dim=embed_dim,
    )
    pretrained_checkpoint = "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_rollout_world_model/version_2/checkpoints/epoch=49-step=164350.ckpt"
    # pretrained_checkpoint = config.get(
    #    "pretrained_jepa_checkpoint",
    #    "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_rollout_world_model/version_2/checkpoints/epoch=49-step=164350.ckpt",
    # )
    # pretrained_checkpoint = "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_rollout_world_model/version_2/checkpoints/epoch=49-step=164350.ckpt"
    load_matching_weights(
        model=model,
        checkpoint_path=pretrained_checkpoint,
    )

    env_name = config.get("env_name", "ALE/MsPacman-v5")

    score_delta_checkpoint = config.get("score_delta_eval_checkpoint", None)

    if score_delta_checkpoint is None:
        score_delta_checkpoint = config.get("score_delta_resume_checkpoint", None)

    if score_delta_checkpoint is None:
        score_delta_checkpoint = (
            Path(
                config.get(
                    "score_delta_checkpoint_dir",
                    f"tb_logs/{env_name}/vjepa_score_delta_online/checkpoints",
                )
            )
            / "latest.pt"
        )

    score_delta_checkpoint = Path(score_delta_checkpoint)

    if not score_delta_checkpoint.exists():
        raise FileNotFoundError(
            f"Could not find score-delta checkpoint: {score_delta_checkpoint}\n"
            f"Set config['score_delta_eval_checkpoint'] to the correct .pt file."
        )

    ckpt = torch.load(
        score_delta_checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.score_predictor.load_state_dict(
        ckpt["score_predictor"],
    )

    print("Loaded score-delta checkpoint:", score_delta_checkpoint)

    model.freeze_world_model()
    model.to(device)
    model.eval()

    return model


# ============================================================
# Run Planner
# ============================================================


with skip_run("run", "cem_planner_eval") as check, check():
    game = config["games"][0]
    env_name = config.get("env_name", "ALE/MsPacman-v5")

    device = select_device(config)

    planner_log_dir = config.get(
        "planner_log_dir",
        f"tb_logs/{game}/vjepa_cem_planner_eval",
    )

    writer = SummaryWriter(log_dir=planner_log_dir)

    model = build_score_delta_model_for_planning(
        config=config,
        device=device,
    )

    gym_manager = GymManager(
        config=config,
        env_name=env_name,
    )

    global_step = 0

    num_eval_episodes = int(config.get("planner_eval_episodes", 5))
    max_steps = config.get("planner_max_steps", None)

    if max_steps is not None:
        max_steps = int(max_steps)

    try:
        for episode_idx in range(num_eval_episodes):
            episode_return, episode_length, global_step = run_planner_episode(
                model=model,
                gym_manager=gym_manager,
                config=config,
                device=device,
                writer=writer,
                global_step_start=global_step,
                max_steps=max_steps,
            )

            print(
                f"[planner episode {episode_idx}] "
                f"return={episode_return:.1f} "
                f"length={episode_length}"
            )

            writer.add_scalar(
                "planner_eval/episode_return",
                episode_return,
                episode_idx,
            )

            writer.add_scalar(
                "planner_eval/episode_length",
                episode_length,
                episode_idx,
            )

    finally:
        gym_manager.close()
        writer.flush()
        writer.close()
# ============================================================
# CEM Probability State Save / Load
# ============================================================


def save_cem_probability_state(
    path,
    planner,
    global_step,
    episode_idx,
    current_score,
    last_action=None,
    last_planned_sequence=None,
    last_action_probs=None,
):
    """
    Saves the temporary CEM action distribution.

    Important:
        planner.prior_probs is the warm-start distribution for the NEXT step.
        last_action_probs is the final distribution used for the CURRENT step.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "global_step": int(global_step),
        "episode_idx": int(episode_idx),
        "current_score": float(current_score),
        "horizon": int(planner.horizon),
        "num_actions": int(planner.num_actions),
        "prior_probs": (
            planner.prior_probs.detach().cpu()
            if planner.prior_probs is not None
            else None
        ),
        "last_action": int(last_action) if last_action is not None else None,
        "last_planned_sequence": (
            last_planned_sequence.detach().cpu()
            if last_planned_sequence is not None
            else None
        ),
        "last_action_probs": (
            last_action_probs.detach().cpu() if last_action_probs is not None else None
        ),
    }

    torch.save(payload, path)


def load_cem_probability_state(
    path,
    device,
    expected_horizon,
    expected_num_actions,
):
    """
    Loads saved CEM warm-start probabilities.

    Returns:
        prior_probs or None
    """

    path = Path(path)

    if not path.exists():
        return None

    ckpt = torch.load(path, map_location=device)

    if ckpt.get("horizon", None) != expected_horizon:
        print(
            "Skipping saved CEM probabilities because horizon changed:",
            ckpt.get("horizon", None),
            "!=",
            expected_horizon,
        )
        return None

    if ckpt.get("num_actions", None) != expected_num_actions:
        print(
            "Skipping saved CEM probabilities because num_actions changed:",
            ckpt.get("num_actions", None),
            "!=",
            expected_num_actions,
        )
        return None

    prior_probs = ckpt.get("prior_probs", None)

    if prior_probs is None:
        return None

    print("Loaded saved CEM prior probabilities from:", path)

    return prior_probs.to(device=device, dtype=torch.float32)


# ============================================================
# Frozen Valuation Loader for CEM
# ============================================================


def freeze_everything_for_cem(model):
    """
    CEM should not train JEPA or the score/valuation model.

    It only updates a temporary probability distribution over action sequences.
    """

    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    model.freeze_world_model()
    model.score_predictor.eval()

    return model


def resolve_score_delta_checkpoint(config, env_name):
    """
    Finds the trained valuation / score-delta checkpoint.

    Priority:
        1. config["score_delta_eval_checkpoint"]
        2. config["score_delta_resume_checkpoint"]
        3. checkpoint_dir/latest.pt
    """

    # checkpoint = config.get("score_delta_eval_checkpoint", None)
    checkpoint = "tb_logs/ALE/MsPacman-v5/vjepa_score_delta_online/checkpoints/score_delta_step=215000.pt"
    if checkpoint is None:
        checkpoint = config.get("score_delta_resume_checkpoint", None)

    if checkpoint is None:
        checkpoint_dir = Path(
            config.get(
                "score_delta_checkpoint_dir",
                f"tb_logs/{env_name}/vjepa_score_delta_online/checkpoints",
            )
        )

        checkpoint = checkpoint_dir / "latest.pt"

    checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Could not find valuation checkpoint:\n"
            f"    {checkpoint}\n\n"
            f"Set config['score_delta_eval_checkpoint'] to your trained "
            f"score-delta checkpoint."
        )

    return checkpoint


def build_frozen_valuation_model_for_cem(config, device):
    """
    Rebuilds:
        pretrained JEPA world model
        +
        trained score-delta valuation model

    Then freezes everything.

    No optimizer is created here.
    """

    env_name = config.get("env_name", "ALE/MsPacman-v5")

    patch_dim = 1 if config.get("grey_scale_v", True) else 3
    embed_dim = int(config.get("embed_dim", 768))
    heads = int(config.get("heads", 12))
    mlp_dim = int(config.get("mlp_dim", 3072))
    depth = int(config.get("encoder_depth", 12))

    tubelet_embed = TubeletEmbedding(
        config=config,
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        img_size=config.get("size_x", 84),
    )

    student = TransformerEncoder(
        embed_dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    net = VJEPAEncoder(
        tubelet_embed=tubelet_embed,
        student=student,
    )

    action_embed = ActionEmbedding()

    base_model = ActionConditionVJEPA(
        model=net,
        action_embed=action_embed,
        config=config,
        lr=1e-4,
        ema_decay=0.996,
    )

    model = OnlineScoreDeltaJEPA(
        base_model=base_model,
        config=config,
        embed_dim=embed_dim,
    )
    pretrained_checkpoint = "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_rollout_world_model/version_2/checkpoints/epoch=49-step=164350.ckpt"
    # pretrained_checkpoint = config.get(
    #    "pretrained_jepa_checkpoint",
    #    "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_action_world_model/version_4/checkpoints/epoch=49-step=164350.ckpt",
    # )

    load_matching_weights(
        model=model,
        checkpoint_path=pretrained_checkpoint,
    )

    valuation_checkpoint = resolve_score_delta_checkpoint(
        config=config,
        env_name=env_name,
    )

    ckpt = torch.load(
        valuation_checkpoint,
        map_location=device,
    )

    model.score_predictor.load_state_dict(
        ckpt["score_predictor"],
    )

    print("Loaded frozen valuation checkpoint:", valuation_checkpoint)

    model.to(device)
    model = freeze_everything_for_cem(model)

    return model


# ============================================================
# CEM Planner Episode Runner
# ============================================================


def run_cem_planner_episode(
    model,
    gym_manager,
    config,
    device,
    writer=None,
    global_step_start=0,
    episode_idx=0,
    max_steps=None,
    cem_probability_path=None,
):
    """
    Runs one Atari episode using frozen JEPA + frozen valuation model.

    This does NOT train the valuation model.

    CEM updates only:
        planner.prior_probs

    That temporary probability state can be saved and reused later.
    """

    model.eval()

    planner = CEMActionPlanner(
        num_actions=gym_manager.num_actions,
        horizon=config.get("planner_horizon", config.get("cycle_steps", 8)),
        num_candidates=config.get("planner_num_candidates", 512),
        cem_iterations=config.get("planner_cem_iterations", 4),
        elite_fraction=config.get("planner_elite_fraction", 0.1),
        eval_batch_size=config.get("planner_eval_batch_size", 128),
        momentum=config.get("planner_momentum", 0.25),
        min_prob=config.get("planner_min_prob", 0.01),
        temperature=config.get("planner_temperature", 1.0),
        device=device,
        warm_start=True,
    )

    if cem_probability_path is not None:
        loaded_prior_probs = load_cem_probability_state(
            path=cem_probability_path,
            device=device,
            expected_horizon=planner.horizon,
            expected_num_actions=planner.num_actions,
        )

        if loaded_prior_probs is not None:
            planner.prior_probs = loaded_prior_probs

    state = gym_manager.reset()

    current_score = 0.0
    episode_return = 0.0
    episode_length = 0
    global_step = int(global_step_start)

    save_policy_every = int(config.get("cem_save_policy_every", 25))

    done = False

    while not done:
        if max_steps is not None and episode_length >= max_steps:
            break

        frames = prepare_state_batch(
            [state],
            config=config,
            device=device,
        )

        with torch.no_grad():
            current_latent = model.encode_current(frames)

            action, planned_sequence, predicted_delta, action_probs = planner.plan(
                model=model,
                current_latent=current_latent,
                current_score=current_score,
            )

        next_state, reward, done = gym_manager.step(action)

        current_score += reward
        episode_return += reward
        episode_length += 1
        global_step += 1

        if writer is not None:
            writer.add_scalar(
                "cem/reward",
                reward,
                global_step,
            )

            writer.add_scalar(
                "cem/current_score",
                current_score,
                global_step,
            )

            writer.add_scalar(
                "cem/predicted_delta",
                predicted_delta,
                global_step,
            )

            writer.add_scalar(
                "cem/chosen_action",
                action,
                global_step,
            )

            entropy_t0 = (
                torch.distributions.Categorical(probs=action_probs[0].detach().cpu())
                .entropy()
                .item()
            )

            writer.add_scalar(
                "cem/action_entropy_t0",
                entropy_t0,
                global_step,
            )

            writer.add_text(
                "cem/planned_sequence",
                str(planned_sequence.detach().cpu().tolist()),
                global_step,
            )

        if (
            cem_probability_path is not None
            and save_policy_every > 0
            and global_step % save_policy_every == 0
        ):
            save_cem_probability_state(
                path=cem_probability_path,
                planner=planner,
                global_step=global_step,
                episode_idx=episode_idx,
                current_score=current_score,
                last_action=action,
                last_planned_sequence=planned_sequence,
                last_action_probs=action_probs,
            )

        state = next_state

    if cem_probability_path is not None:
        save_cem_probability_state(
            path=cem_probability_path,
            planner=planner,
            global_step=global_step,
            episode_idx=episode_idx,
            current_score=current_score,
            last_action=None,
            last_planned_sequence=None,
            last_action_probs=None,
        )

    if writer is not None:
        writer.add_scalar(
            "cem_episode/return",
            episode_return,
            global_step,
        )

        writer.add_scalar(
            "cem_episode/length",
            episode_length,
            global_step,
        )

    return episode_return, episode_length, global_step


# ============================================================
# Second Skip-Run Block:
# Frozen Valuation + CEM Action Search
# ============================================================

with skip_run("skip", "cem_planner_with_frozen_valuation") as check, check():
    game = config["games"][0]
    env_name = config.get("env_name", "ALE/MsPacman-v5")

    device = select_device(config)

    print("Running CEM planner with frozen valuation model")
    print("Device:", device)
    print("Env:", env_name)

    cem_log_dir = config.get(
        "cem_log_dir",
        f"tb_logs/{game}/vjepa_cem_planner",
    )

    writer = SummaryWriter(log_dir=cem_log_dir)

    print("CEM TensorBoard log dir:", cem_log_dir)

    model = build_frozen_valuation_model_for_cem(
        config=config,
        device=device,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Trainable parameters during CEM:", trainable_params)

    if trainable_params != 0:
        raise RuntimeError(
            "CEM planner should have zero trainable parameters. "
            "The valuation model was not fully frozen."
        )

    gym_manager = GymManager(
        config=config,
        env_name=env_name,
    )

    global_step = 0

    num_eval_episodes = int(config.get("cem_eval_episodes", 5))
    max_steps = config.get("cem_max_steps", None)

    if max_steps is not None:
        max_steps = int(max_steps)

    cem_probability_path = Path(
        config.get(
            "cem_probability_checkpoint",
            f"{cem_log_dir}/cem_action_probs_latest.pt",
        )
    )

    print("CEM probability checkpoint:", cem_probability_path)

    try:
        for episode_idx in range(num_eval_episodes):
            episode_return, episode_length, global_step = run_cem_planner_episode(
                model=model,
                gym_manager=gym_manager,
                config=config,
                device=device,
                writer=writer,
                global_step_start=global_step,
                episode_idx=episode_idx,
                max_steps=max_steps,
                cem_probability_path=cem_probability_path,
            )

            print(
                f"[CEM episode {episode_idx}] "
                f"return={episode_return:.1f} "
                f"length={episode_length}"
            )

            writer.add_scalar(
                "cem_eval/episode_return",
                episode_return,
                episode_idx,
            )

            writer.add_scalar(
                "cem_eval/episode_length",
                episode_length,
                episode_idx,
            )

    finally:
        gym_manager.close()
        writer.flush()
        writer.close()
