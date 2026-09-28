from collections import Counter
from pathlib import Path
from typing import Any

import ale_py  # noqa: F401  # force registration of the ALE namespace
import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from .utils import (
    ALE_ACTION_NAMES,
    print_episode_report,
    print_header,
    print_step,
    print_summary_report,
)

# ActionNet only consumes the stacked image, never gaze/action, so these
# placeholders just need to satisfy the caller's pipeline's sample shape.
# eye_gaze_to_density_image expects a sequence of (x, y) points per frame.
_DUMMY_GAZE = [(0, 0)]
_DUMMY_ACTION = 0


class RuntimePreprocessor:
    """
    Wraps the caller's own preprocessing pipeline instance (e.g. the same
    ComposePreprocessor of Resize, StackWithLabels, or whatever else they
    compose, used to build offline training data), so frames seen during
    Gym evaluation are processed identically to training data.

    Some steps (e.g. StackWithLabels) hold per-episode state, like a
    frame deque, that must not leak across episodes. Every preprocessor
    implements reset(), so ComposePreprocessor.reset() clears them all.
    """

    def __init__(self, preprocessor_pipeline):
        self.preprocessor_pipeline = preprocessor_pipeline

    def _process(self, obs):
        stacked_img, _, _ = self.preprocessor_pipeline(
            (obs, _DUMMY_GAZE, _DUMMY_ACTION)
        )

        return stacked_img

    def reset(self, obs):
        self.preprocessor_pipeline.reset()

        return self._process(obs)

    def step(self, obs):
        return self._process(obs)


class RuntimeActionNet:
    def __init__(
        self,
        action_net,
        config: dict,
        action_space: Any,
        legal_action_ids: list,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = Path(config["action_classifier_checkpoint"])

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Action-classifier checkpoint was not found: {checkpoint_path}"
            )

        # The env must be built with full_action_space=True so env action
        # IDs equal the raw ALE IDs (0-17) the classifier was trained on.
        if action_space.n != 18:
            raise RuntimeError(
                f"Expected the full 18-action ALE space, but Gym reports "
                f"{action_space.n}. Create the env with full_action_space=True."
            )

        self.num_actions = int(action_space.n)

        # Only the game's own legal actions may be chosen; the other
        # classes are masked out at inference.
        self.action_mask = torch.full(
            (self.num_actions,),
            float("-inf"),
            device=self.device,
        )
        self.action_mask[list(legal_action_ids)] = 0.0

        print("Legal actions:", list(legal_action_ids))

        self.model = action_net

        checkpoint = self._load_checkpoint(checkpoint_path)

        model_state = self._extract_action_net_state(checkpoint)

        self.model.load_state_dict(
            model_state,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded checkpoint: {checkpoint_path}")

        print(f"Inference device: {self.device}")

        print(f"Gym action count: {action_space.n}")

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> dict:
        try:
            return torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            return torch.load(
                checkpoint_path,
                map_location=self.device,
            )

    def _extract_action_net_state(
        self,
        checkpoint: dict,
    ) -> dict:

        checkpoint_state = checkpoint.get(
            "state_dict",
            checkpoint,
        )

        expected_keys = set(self.model.state_dict().keys())

        extracted_state = {}

        possible_prefixes = (
            "",
            "net.",
            "model.",
            "model.net.",
            "module.",
            "module.net.",
        )

        for original_key, value in checkpoint_state.items():
            for prefix in possible_prefixes:
                if prefix and original_key.startswith(prefix):
                    candidate = original_key[len(prefix) :]
                else:
                    candidate = original_key

                if candidate in expected_keys:
                    extracted_state[candidate] = value
                    break

        missing_keys = expected_keys - set(extracted_state.keys())

        expected_state = self.model.state_dict()

        unexpected_shapes = []

        for key, value in extracted_state.items():
            if value.shape != expected_state[key].shape:
                unexpected_shapes.append(
                    (
                        key,
                        tuple(value.shape),
                        tuple(expected_state[key].shape),
                    )
                )

        if missing_keys:
            missing_text = "\n".join(f"  - {key}" for key in sorted(missing_keys))

            raise RuntimeError(
                f"The checkpoint is missing ActionNet parameters:\n{missing_text}"
            )

        if unexpected_shapes:
            shape_text = "\n".join(
                f"  - {key}: checkpoint={found}, expected={expected}"
                for key, found, expected in unexpected_shapes
            )

            raise RuntimeError(
                f"Checkpoint architecture does not match ActionNet.\n{shape_text}"
            )

        return extracted_state

    def act(
        self,
        data_packet: Any,
    ) -> int:

        state = torch.as_tensor(
            data_packet,
            dtype=torch.float32,
            device=self.device,
        )

        # Four RGB frames stacked along the channel dimension:
        #
        # [12, 84, 84]
        # ->
        # [1, 12, 84, 84]
        if state.ndim == 3:
            state = state.unsqueeze(0)

        if state.ndim != 4:
            raise ValueError(
                f"ActionNet expected [batch, 12, H, W], got {tuple(state.shape)}"
            )

        if state.shape[1] != 12:
            raise ValueError(
                "ActionNet expected four RGB frames "
                "(12 channels), "
                f"got {tuple(state.shape)}"
            )

        with torch.inference_mode():
            logits = self.model(state)

        # One action prediction for the four-frame stack.
        # Expected: [B, 1, 18]
        if logits.ndim != 3 or logits.shape[-1] != self.num_actions:
            raise RuntimeError(
                f"Expected ActionNet output [B, 1, {self.num_actions}], "
                f"got {tuple(logits.shape)}"
            )

        action_logits = logits[:, -1, :] + self.action_mask

        action = int(action_logits.argmax(dim=-1).item())

        return action


class GymManager:
    def __init__(
        self,
        config,
        preprocessor_pipeline,
        action_net,
        env_name: str = "ALE/SpaceInvaders-v5",
        record_video: bool = False,
        video_folder: str = "videos/action_classifier",
        episode_trigger=lambda episode_id: True,
    ):
        if record_video:
            video_path = Path(video_folder)
            video_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            base_env = gym.make(
                env_name,
                render_mode="rgb_array",
                full_action_space=True,
            )

            self.env = RecordVideo(
                base_env,
                video_folder=str(video_path),
                episode_trigger=episode_trigger,
                name_prefix="action-classifier",
                disable_logger=False,
            )

            print(
                "Video directory:",
                video_path.resolve(),
            )

        else:
            self.env = gym.make(env_name, full_action_space=True)

        # Legal (minimal) ALE action IDs for this particular game.
        legal_action_ids = [
            int(a) for a in self.env.unwrapped.ale.getMinimalActionSet()
        ]

        print(
            "Environment actions:",
            self.env.unwrapped.get_action_meanings(),
        )

        self.preprocessor = RuntimePreprocessor(preprocessor_pipeline)

        self.action_net = RuntimeActionNet(
            action_net,
            config=config,
            action_space=self.env.action_space,
            legal_action_ids=legal_action_ids,
        )

        self.state = None

        # -----------------------------------------------------
        # LIFE TRACKING
        # -----------------------------------------------------
        self.lives = None

    def reset(self):
        observation, info = self.env.reset()

        # -----------------------------------------------------
        # TRACK STARTING NUMBER OF LIVES
        # -----------------------------------------------------
        self.lives = info.get("lives")

        print(
            "Starting lives:",
            self.lives,
        )

        self.state = self.preprocessor.reset(observation)

        return self.state

    def step(self):
        # -----------------------------------------------------
        # MODEL SELECTS ACTION
        # -----------------------------------------------------

        action = self.action_net.act(self.state)

        # -----------------------------------------------------
        # EXECUTE EXACTLY THAT ACTION
        # -----------------------------------------------------

        (observation, reward, terminated, truncated, info) = self.env.step(action)

        # Expose the exact action sent to the env so callers can log it.
        info = dict(info)
        info["action"] = action

        done = terminated or truncated

        # -----------------------------------------------------
        # LIFE TRACKING
        # -----------------------------------------------------

        current_lives = info.get("lives")

        life_lost = (
            not done
            and self.lives is not None
            and current_lives is not None
            and current_lives < self.lives
        )

        if life_lost:
            print(f"Life lost: {self.lives} -> {current_lives}")

        # Store the latest life count.
        if current_lives is not None:
            self.lives = current_lives

        # -----------------------------------------------------
        # NEXT STATE
        # -----------------------------------------------------

        self.state = self.preprocessor.step(observation)

        return (
            self.state,
            reward,
            done,
            info,
        )

    def close(self):
        self.env.close()


def run_episode(manager, max_steps: int, log_first_steps: int = 0) -> dict:
    """Play one episode and return its raw stats (no reporting)."""
    manager.reset()

    total_reward = 0.0
    step_count = 0
    final_info = {}
    done = False
    action_counts = Counter()

    while not done and step_count < max_steps:
        _, reward, done, info = manager.step()

        action = int(info["action"])

        if not 0 <= action < len(ALE_ACTION_NAMES):
            raise RuntimeError(
                f"Invalid action: {action}. Expected 0-{len(ALE_ACTION_NAMES) - 1}."
            )

        action_counts[action] += 1

        if step_count < log_first_steps:
            print_step(step_count, action, float(reward))

        total_reward += float(reward)
        step_count += 1
        final_info = info

    return {
        "steps": step_count,
        "reward": total_reward,
        "action_counts": action_counts,
        "final_info": final_info,
        "hit_max_steps": step_count >= max_steps and not done,
    }


def evaluate_policy(manager, num_episodes: int, max_steps: int) -> dict:
    """
    Run the GymManager's action classifier for num_episodes episodes, then
    print the per-episode and overall reports (see evaluate/utils.py).

    Returns {"episodes": [stats, ...], "rewards": [...],
             "total_action_counts": Counter}.
    """
    episodes = []

    try:
        for episode in range(num_episodes):
            print_header(f"Episode {episode + 1}/{num_episodes}")

            episodes.append(
                run_episode(
                    manager,
                    max_steps,
                    log_first_steps=30 if episode == 0 else 0,
                )
            )
    finally:
        manager.close()

    # Reports are generated once, after all episodes have finished.
    total_action_counts = Counter()

    for episode, stats in enumerate(episodes, start=1):
        total_action_counts.update(stats["action_counts"])

        print_episode_report(
            episode,
            stats["steps"],
            stats["reward"],
            stats["action_counts"],
            stats["final_info"],
            stats["hit_max_steps"],
            max_steps,
        )

    rewards = [stats["reward"] for stats in episodes]

    print_summary_report(rewards, total_action_counts)

    if hasattr(manager.action_net, "report"):
        manager.action_net.report()

    return {
        "episodes": episodes,
        "rewards": rewards,
        "total_action_counts": total_action_counts,
    }
