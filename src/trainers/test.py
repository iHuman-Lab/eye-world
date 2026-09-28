import sys

import ale_py  # ⚠️ force registration of ALE namespace
import gymnasium as gym

# import gym

print("Python version:", sys.version)
print("Gym version:", gym.__version__)
# import gymnasium.atari

# Dummy usage to prevent deletion
_ = ale_py.__name__
env = gym.make("ALE/MsPacman-v5")
obs, _ = env.reset()
print("Environment ready, observation shape:", obs.shape)
env.close()
import yaml

# =========================
# Config
# =========================
config_path = "configs/config.yaml"
config = yaml.load(open(config_path), Loader=yaml.SafeLoader)


class RuntimePreprocessor:
    def __init__(self, config):
        self.pipeline = ComposePreprocessor(
            [
                Resize(config),
                StackWithLabels(config),
            ]
        )

        self.last_action = 0
        self.last_gaze = [(0, 0)]

    def reset(self):
        self.pipeline.transforms[1].reset()

    def step(self, obs):
        sample = (obs, self.last_gaze, self.last_action)
        stacked_img, _, stacked_action = self.pipeline(sample)
        self.last_action = stacked_action[-1].item()
        return stacked_img


import random
import sys
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

# =========================
# Setup
# =========================

print("Python version:", sys.version)
print("Gym version:", gym.__version__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os

base_dir = "tb_logs"
os.makedirs(base_dir, exist_ok=True)

# find next run number
existing_runs = [
    d
    for d in os.listdir(base_dir)
    if d.startswith("DQN_") and os.path.isdir(os.path.join(base_dir, d))
]

run_numbers = []
for d in existing_runs:
    try:
        run_numbers.append(int(d.split("_")[1]))
    except:
        pass

next_run = max(run_numbers) + 1 if run_numbers else 1

log_dir = os.path.join(base_dir, f"DQN_{next_run}")
writer = SummaryWriter(log_dir=log_dir)

print(f"[TensorBoard] Logging to: {log_dir}")
# logger = TensorBoardLogger("tb_logs", name="multi_game/action_classifier/")
# =========================
# Hyperparameters
# =========================
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
MIN_BUFFER_SIZE = 10_000
TARGET_UPDATE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.08
EPSILON_DECAY = 200_000


import json
import os

params = {
    "gamma": GAMMA,
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "buffer_size": BUFFER_SIZE,
    "epsilon_start": EPSILON_START,
    "epsilon_end": EPSILON_END,
    "epsilon_decay": EPSILON_DECAY,
    "target_update": TARGET_UPDATE,
}

with open(os.path.join(log_dir, "hyperparameters.json"), "w") as f:
    json.dump(params, f, indent=4)

# =========================
# Config
# =========================
config_path = "configs/config.yaml"
config = yaml.load(open(config_path), Loader=yaml.SafeLoader)

# =========================
# Preprocessing
# =========================


class ComposePreprocessor:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    def __init__(self, config):
        self.size = config.get("image_size", 84)

    def __call__(self, sample):
        obs, gaze, action = sample

        obs = torch.tensor(obs, dtype=torch.float32)

        if obs.ndim == 3:
            obs = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]

        obs = obs.unsqueeze(0).unsqueeze(0)

        obs = F.interpolate(
            obs,
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        )

        obs = obs.squeeze(0)

        return (obs, gaze, action)


class StackWithLabels:
    def __init__(self, config):
        self.stack_size = config.get("stack_size", 4)
        self.frames = []

    def reset(self):
        self.frames = []

    def __call__(self, sample):
        obs, gaze, action = sample

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        if obs.ndim == 2:
            obs = obs.unsqueeze(0)

        self.frames.append(obs)

        if len(self.frames) > self.stack_size:
            self.frames.pop(0)

        while len(self.frames) < self.stack_size:
            self.frames.insert(0, obs)

        stacked = torch.cat(self.frames, dim=0)

        return stacked, gaze, torch.tensor([action])


class RuntimePreprocessor:
    def __init__(self, config):
        self.pipeline = ComposePreprocessor(
            [
                Resize(config),
                StackWithLabels(config),
            ]
        )

        self.last_action = 0
        self.last_gaze = [(0, 0)]

    def reset(self):
        self.pipeline.transforms[1].reset()

    def step(self, obs):
        sample = (obs, self.last_gaze, self.last_action)
        stacked_img, _, stacked_action = self.pipeline(sample)
        self.last_action = stacked_action[-1].item()
        return stacked_img


# =========================
# DQN
# =========================


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        c, h, w = input_shape

        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.net(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = x.float()
        return self.head(self.net(x))


# =========================
# Replay Buffer
# =========================


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)

        return (
            torch.stack(s).to(DEVICE),
            torch.tensor(a, device=DEVICE),
            torch.tensor(r, device=DEVICE, dtype=torch.float32),
            torch.stack(s_next).to(DEVICE),
            torch.tensor(d, device=DEVICE, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# Epsilon
# =========================


def get_epsilon(step):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-step / EPSILON_DECAY)


# =========================
# Train Step
# =========================

# /home/cody/Documents/IHL/eye-world/tb_logs/DQN_1/checkpoint_ep3500.pth
# /home/cody/Documents/IHL/eye-world/tb_logs/DQN_11/checkpoint_ep50.pth
# /home/cody/Documents/IHL/eye-world/tb_logs/DQN_13/checkpoint_ep50.pth


def evaluate(
    checkpoint_path="/home/cody/Documents/IHL/eye-world/tb_logs/DQN_13/checkpoint_ep3500.pth",
    num_episodes=100,
):
    env = gym.make("ALE/MsPacman-v5")

    runtime_preprocessor = RuntimePreprocessor(config)

    # --- init model ---
    obs, _ = env.reset()
    state = runtime_preprocessor.step(obs)

    policy_net = DQN(state.shape, env.action_space.n).to(DEVICE)

    # --- load checkpoint ---
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # handle both .ckpt and .pth formats
    if "policy_net" in checkpoint:
        policy_net.load_state_dict(checkpoint["policy_net"])
    else:
        policy_net.load_state_dict(checkpoint)

    policy_net.eval()

    rewards = []

    # =========================
    # Run episodes
    # =========================
    for ep in range(num_episodes):
        obs, _ = env.reset()
        runtime_preprocessor.reset()
        state = runtime_preprocessor.step(obs)

        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                q_values = policy_net(state.unsqueeze(0).to(DEVICE))
                action = torch.argmax(q_values, dim=1).item()  # GREEDY

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = runtime_preprocessor.step(next_obs)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep + 1}: Reward = {total_reward}")

    env.close()

    print("\nAll rewards:", rewards)
    print("Average reward:", sum(rewards) / len(rewards))

    return rewards


evaluate()


def train_step(policy_net, target_net, buffer, optimizer):
    if len(buffer) < MIN_BUFFER_SIZE:
        return None

    s, a, r, s_next, done = buffer.sample(BATCH_SIZE)

    q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_net(s_next).max(1)[0]
        target = r + GAMMA * next_q * (1 - done)

    loss = F.smooth_l1_loss(q, target)
    # loss = F.mse_loss(q, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# =========================
# Training Loop
# =========================


def train():
    env = gym.make("ALE/MsPacman-v5")

    runtime_preprocessor = RuntimePreprocessor(config)

    obs, _ = env.reset()
    state = runtime_preprocessor.step(obs)

    policy_net = DQN(state.shape, env.action_space.n).to(DEVICE)
    target_net = DQN(state.shape, env.action_space.n).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    step = 0
    episode = 0
    recent_rewards = []

    while True:
        obs, _ = env.reset()
        runtime_preprocessor.reset()
        state = runtime_preprocessor.step(obs)

        done = False
        total_reward = 0

        while not done:
            epsilon = get_epsilon(step)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy_net(state.unsqueeze(0).to(DEVICE))
                    action = torch.argmax(q, dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = runtime_preprocessor.step(next_obs)

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            loss = train_step(policy_net, target_net, buffer, optimizer)

            if step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            writer.add_scalar("train/epsilon", epsilon, step)

            if loss is not None and step % 10 == 0:
                writer.add_scalar("train/loss", loss, step)

            step += 1

        episode += 1
        if episode % 50 == 0:
            checkpoint = {
                "policy_net": policy_net.state_dict(),
                "target_net": target_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "episode": episode,
            }

            torch.save(checkpoint, os.path.join(log_dir, f"checkpoint_ep{episode}.pth"))

        print(
            f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}"
        )

        writer.add_scalar("episode/reward", total_reward, episode)
        writer.add_scalar("episode/epsilon", epsilon, episode)
        writer.add_scalar("episode/step_count", step, episode)
        # =========================
        # Save model architecture
        # =========================
        with open(os.path.join(log_dir, "model_info.txt"), "w") as f:
            f.write(str(policy_net))

        recent_rewards.append(total_reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)

        if len(recent_rewards) == 10:
            writer.add_scalar(
                "episode/avg_reward_10", sum(recent_rewards) / 10, episode
            )


if __name__ == "__main__":
    train()


"""checkpoint = torch.load("tb_logs/DQN_1/checkpoint_ep100.pth", map_location=DEVICE)

policy_net.load_state_dict(checkpoint["policy_net"])
target_net.load_state_dict(checkpoint["target_net"])
optimizer.load_state_dict(checkpoint["optimizer"])

step = checkpoint["step"]
episode = checkpoint["episode"]"""


import torch
import torch.nn as nn


class VJEPAWorldModel(nn.Module):
    def __init__(self, vjepa_encoder, action_embed, latent_predictor, config):
        super().__init__()

        self.tubelet_embed = vjepa_encoder.tubelet_embed
        self.encoder = vjepa_encoder.student.encoder

        self.action_embed = action_embed
        self.latent_predictor = latent_predictor

        self.context_frames = config["context_frames"]

    @torch.no_grad()
    def predict_next_latent(self, frames, action):
        """
        frames: [4, 1, H, W]
        action: int

        returns: [D]
        """

        device = next(self.parameters()).device

        # -------------------------
        # Format input
        # -------------------------
        frames = frames.unsqueeze(0).to(device)  # [1, 4, 1, H, W]

        action = torch.tensor([action], device=device)  # [1]

        # -------------------------
        # Encode frames
        # -------------------------
        tokens = self.tubelet_embed(frames)  # [1, N, D]
        tokens = self.encoder(tokens)  # [1, N, D]

        # -------------------------
        # Action embedding
        # -------------------------
        action_emb = self.action_embed(action)  # [1, D]
        action_emb = action_emb.unsqueeze(1)  # [1, 1, D]

        # -------------------------
        # Combine
        # -------------------------
        seq = torch.cat([tokens, action_emb], dim=1)  # [1, N+1, D]

        # -------------------------
        # Predict next latent
        # -------------------------
        pred_seq = self.latent_predictor(seq, seq)  # [1, N+1, D]

        next_latent = pred_seq[:, -1, :]  # [1, D]

        return next_latent.squeeze(0)  # [D]


"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Reuse your existing model pieces
# -------------------------
net = VJEPAEncoder(
    tubelet_embed=TubeletEmbedding(config=config, patch_dim=1, embed_dim=768, img_size=84),
    student=TransformerEncoder(768, depth=12, heads=12, mlp_dim=3072),
)

model = ActionConditionVJEPA(
    model=net,
    action_embed=ActionEmbedding(),
    config=config,
).to(DEVICE)

model.eval()



@torch.no_grad()
def predict_next_latent(model, frames, action):
    device = next(model.parameters()).device

    frames = frames.unsqueeze(0).to(device)   # [1,4,1,H,W]
    action = torch.tensor([action], device=device)

    # encode frames
    tokens = model.model.tubelet_embed(frames)
    tokens = model.model.student.encoder(tokens)

    # action
    action_emb = model.action_embed(action).unsqueeze(1)

    # combine
    seq = torch.cat([tokens, action_emb], dim=1)

    # predict
    pred = model.latent_predictor(seq, seq)

    return pred[:, -1, :].squeeze(0)


frames = torch.randint(0, 256, (4, 1, 84, 84), dtype=torch.float32)
action = 1

latent = predict_next_latent(model, frames, action)

print(latent.shape)   # should be [768]"""
