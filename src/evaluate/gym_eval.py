from collections import Counter, deque
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


# =========================================================
# Config-derived settings
# =========================================================


def env_id_from_config(config: dict) -> str:
    """
    Gym env id for config["games"][0].

    Dataset game names are snake_case and the ALE v5 ids are CamelCase of
    the same words, e.g. space_invaders -> ALE/SpaceInvaders-v5 and
    ms_pacman -> ALE/MsPacman-v5.
    """
    game = config["games"][0]
    camel = "".join(word.capitalize() for word in game.split("_"))

    return f"ALE/{camel}-v5"


def expected_channels(config: dict) -> int:
    """Channels of the stacked state: stack_length frames, 1 or 3 channels each."""
    return int(config["stack_length"]) * (1 if config.get("grey_scale", True) else 3)


class RuntimePreprocessor:
    """
    Wraps the caller's own preprocessing pipeline instance.

    For example:
        ComposePreprocessor(
            [
                Resize(config),
                StackWithLabels(config),
            ]
        )

    This ensures the Gym frames are processed the same way
    as the offline training data.

    Some preprocessing steps maintain per-episode state.
    Calling reset() clears that state between episodes.
    """

    def __init__(
        self,
        preprocessor_pipeline,
    ):
        self.preprocessor_pipeline = preprocessor_pipeline

    def _process(
        self,
        obs,
    ):
        stacked_img, _, _ = self.preprocessor_pipeline(
            (
                obs,
                _DUMMY_GAZE,
                _DUMMY_ACTION,
            )
        )

        return stacked_img

    def reset(
        self,
        obs,
    ):
        self.preprocessor_pipeline.reset()

        return self._process(obs)

    def step(
        self,
        obs,
    ):
        return self._process(obs)


# ================================================================
# Runtime action network
# ================================================================


class RuntimeActionNet:
    def __init__(
        self,
        action_net,
        config: dict,
        action_space: Any,
        legal_action_ids: list,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -----------------------------------------------------
        # Load action-classifier checkpoint
        # -----------------------------------------------------

        checkpoint_path = Path(config["action_classifier_checkpoint"])

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Action-classifier checkpoint was not found: {checkpoint_path}"
            )

        # -----------------------------------------------------
        # Full ALE action space
        #
        # The classifier predicts raw ALE action IDs 0-17.
        #
        # Therefore the Gym environment must use:
        #
        # full_action_space=True
        # -----------------------------------------------------

        self.num_actions = int(config["num_actions"])

        if self.num_actions != len(ALE_ACTION_NAMES):
            raise RuntimeError(
                f"config num_actions={self.num_actions} but the full ALE "
                f"action space has {len(ALE_ACTION_NAMES)} actions."
            )

        if action_space.n != self.num_actions:
            raise RuntimeError(
                f"Expected the full {self.num_actions}-action ALE space, "
                f"but Gym reports {action_space.n}. "
                "Create the env with full_action_space=True."
            )

        self.expected_channels = expected_channels(config)

        # -----------------------------------------------------
        # Legal action mask
        #
        # Only actions available in the current game
        # can be selected.
        # -----------------------------------------------------

        self.action_mask = torch.full(
            (self.num_actions,),
            float("-inf"),
            device=self.device,
        )

        self.action_mask[list(legal_action_ids)] = 0.0

        print(
            "Legal actions:",
            list(legal_action_ids),
        )

        # -----------------------------------------------------
        # Network
        # -----------------------------------------------------

        self.model = action_net

        # -----------------------------------------------------
        # Load checkpoint
        # -----------------------------------------------------

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

    # =========================================================
    # Load checkpoint
    # =========================================================

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

    # =========================================================
    # Extract ActionNet parameters
    # =========================================================

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

        # -----------------------------------------------------
        # Remove Lightning model prefixes where needed
        # -----------------------------------------------------

        for original_key, value in checkpoint_state.items():
            for prefix in possible_prefixes:
                if prefix and original_key.startswith(prefix):
                    candidate = original_key[len(prefix) :]

                else:
                    candidate = original_key

                if candidate in expected_keys:
                    extracted_state[candidate] = value

                    break

        # -----------------------------------------------------
        # Check for missing parameters
        # -----------------------------------------------------

        missing_keys = expected_keys - set(extracted_state.keys())

        expected_state = self.model.state_dict()

        unexpected_shapes = []

        # -----------------------------------------------------
        # Validate checkpoint parameter shapes
        # -----------------------------------------------------

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

    # =========================================================
    # Predict an action
    #
    # Supports both:
    #
    # CNN:
    #     model(state)
    #
    # JEPA:
    #     model(state, previous_actions)
    # =========================================================

    def act(
        self,
        data_packet: Any,
        previous_actions=None,  # NEW
    ) -> int:

        state = torch.as_tensor(
            data_packet,
            dtype=torch.float32,
            device=self.device,
        )

        # -----------------------------------------------------
        # stack_length frames stacked along the channel dimension
        # (config-derived), e.g. 4 RGB frames:
        #
        # [12, 84, 84]
        #
        # ->
        #
        # [1, 12, 84, 84]
        # -----------------------------------------------------

        if state.ndim == 3:
            state = state.unsqueeze(0)

        if state.ndim != 4:
            raise ValueError(
                f"ActionNet expected [batch, {self.expected_channels}, H, W], "
                f"got {tuple(state.shape)}"
            )

        if state.shape[1] != self.expected_channels:
            raise ValueError(
                f"ActionNet expected {self.expected_channels} channels "
                "(config stack_length x channels per frame), "
                f"got {tuple(state.shape)}"
            )

        # =====================================================
        # NEW: Support both network architectures
        # =====================================================

        with torch.inference_mode():
            if previous_actions is None:
                # ---------------------------------------------
                # Original CNN
                #
                # Input:
                #     state [B, 12, 84, 84]
                #
                # Output:
                #     logits [B, 1, 18]
                # ---------------------------------------------

                logits = self.model(state)

            else:
                # ---------------------------------------------
                # JEPA action-history classifier
                #
                # Input:
                #     state [B, 12, 84, 84]
                #
                #     previous_actions [B, 3]
                #
                # Output:
                #     logits [B, 1, 18]
                # ---------------------------------------------

                history = torch.as_tensor(
                    previous_actions,
                    dtype=torch.long,
                    device=self.device,
                ).reshape(
                    1,
                    -1,
                )

                if history.shape != (1, 3):
                    raise ValueError(
                        "Expected three previous actions "
                        "[1, 3], "
                        f"got {tuple(history.shape)}"
                    )

                logits = self.model(
                    state,
                    history,
                )

        # =====================================================
        # Original logits processing — UNCHANGED
        # =====================================================

        # One action prediction for the four-frame stack.
        #
        # Expected:
        #
        # [B, 1, 18]

        if logits.ndim != 3 or logits.shape[-1] != self.num_actions:
            raise RuntimeError(
                "Expected ActionNet output "
                f"[B, 1, {self.num_actions}], "
                f"got {tuple(logits.shape)}"
            )

        # -----------------------------------------------------
        # Apply legal action mask
        # -----------------------------------------------------
        action_logits = logits[:, -1, :] + self.action_mask

        action = int(action_logits.argmax(dim=-1).item())

        # ============================================================
        # DEBUG: Inspect model predictions
        # ============================================================

        if not hasattr(self, "_debug_steps"):
            self._debug_steps = 0

        if self._debug_steps < 10:
            print("\n----------------------------------------")
            print("MODEL PREDICTION DEBUG")
            print("----------------------------------------")

            print("Previous actions:", previous_actions)

            print("\nLegal action logits:")

            legal_ids = torch.where(torch.isfinite(self.action_mask))[0]

            for action_id in legal_ids.tolist():
                score = logits[
                    0,
                    -1,
                    action_id,
                ].item()

                print(f"{action_id:2d} {ALE_ACTION_NAMES[action_id]:15s} {score:.4f}")

            print("\nSelected action:", action)
            print("----------------------------------------")

            self._debug_steps += 1

        return action


"""
        action_logits = logits[:, -1, :] + self.action_mask

        # -----------------------------------------------------
        # Select highest-scoring legal action
        # -----------------------------------------------------

        action = int(action_logits.argmax(dim=-1).item())

        return action

"""
# ================================================================
# Gym manager
# ================================================================


class GymManager:
    def __init__(
        self,
        config,
        preprocessor_pipeline,
        action_net,
        env_name: str | None = None,
        record_video: bool = False,
        video_folder: str = "videos/action_classifier",
        episode_trigger=lambda episode_id: True,
        action_history_length: int = 0,  # NEW
    ):

        # An explicit env_name overrides; otherwise follow config["games"].
        if env_name is None:
            env_name = env_id_from_config(config)

        print("Gym environment:", env_name)

        # =====================================================
        # Create Gym environment
        # =====================================================

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
            self.env = gym.make(
                env_name,
                full_action_space=True,
            )

        # =====================================================
        # Legal ALE actions
        # =====================================================

        legal_action_ids = [
            int(a) for a in self.env.unwrapped.ale.getMinimalActionSet()
        ]

        print(
            "Environment actions:",
            self.env.unwrapped.get_action_meanings(),
        )

        # =====================================================
        # Preprocessor
        # =====================================================

        self.preprocessor = RuntimePreprocessor(preprocessor_pipeline)

        # =====================================================
        # Action network
        # =====================================================

        self.action_net = RuntimeActionNet(
            action_net,
            config=config,
            action_space=self.env.action_space,
            legal_action_ids=legal_action_ids,
        )

        self.state = None

        # =====================================================
        # Life tracking
        # =====================================================

        self.lives = None

        # =====================================================
        # NEW: Optional previous-action history
        #
        # CNN:
        #     action_history_length=0
        #
        # JEPA:
        #     action_history_length=3
        # =====================================================

        if action_history_length not in (0, 3):
            raise ValueError("action_history_length must be 0 (CNN) or 3 (JEPA)")

        self.action_history = deque(maxlen=3) if action_history_length == 3 else None

    # =========================================================
    # Reset episode
    # =========================================================

    def reset(self):

        observation, info = self.env.reset()

        # -----------------------------------------------------
        # Track starting number of lives
        # -----------------------------------------------------

        self.lives = info.get("lives")

        print(
            "Starting lives:",
            self.lives,
        )

        # =====================================================
        # NEW: Reset previous-action history
        #
        # At the beginning of an episode, there are no
        # previous actions.
        #
        # Initialize with three ALE NOOP actions.
        #
        # Action 0 = NOOP.
        # =====================================================

        if self.action_history is not None:
            self.action_history.clear()

            self.action_history.extend([0, 0, 0])

        # -----------------------------------------------------
        # Reset preprocessing
        # -----------------------------------------------------

        self.state = self.preprocessor.reset(observation)

        return self.state

    # =========================================================
    # Execute one Gym step
    # =========================================================

    def step(self):

        # =====================================================
        # NEW: Read previous three actions
        # =====================================================

        previous_actions = (
            list(self.action_history) if self.action_history is not None else None
        )

        # =====================================================
        # Model selects action
        #
        # CNN:
        #     act(state, None)
        #
        # JEPA:
        #     act(state, previous_actions)
        # =====================================================

        action = self.action_net.act(
            self.state,
            previous_actions=previous_actions,
        )

        # =====================================================
        # Execute exactly that action
        # =====================================================

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        # =====================================================
        # NEW: Update previous-action history
        #
        # Update AFTER predicting and executing.
        #
        # This ensures the current action is never
        # included in its own conditioning history.
        # =====================================================

        if self.action_history is not None:
            self.action_history.append(action)

        # -----------------------------------------------------
        # Expose action to reporting functions
        # -----------------------------------------------------

        info = dict(info)

        info["action"] = action

        done = terminated or truncated

        # =====================================================
        # Life tracking — UNCHANGED
        # =====================================================

        current_lives = info.get("lives")

        life_lost = (
            not done
            and self.lives is not None
            and current_lives is not None
            and current_lives < self.lives
        )

        if life_lost:
            print(f"Life lost: {self.lives} -> {current_lives}")

        # Store latest life count

        if current_lives is not None:
            self.lives = current_lives

        # =====================================================
        # Next state — UNCHANGED
        # =====================================================

        self.state = self.preprocessor.step(observation)

        return (
            self.state,
            reward,
            done,
            info,
        )

    # =========================================================
    # Close environment
    # =========================================================

    def close(self):

        self.env.close()


# ================================================================
# Run a single episode
#
# UNCHANGED
# ================================================================


def run_episode(
    manager,
    max_steps: int,
    log_first_steps: int = 0,
) -> dict:
    """
    Play one episode and return its raw statistics.
    """

    manager.reset()

    total_reward = 0.0

    step_count = 0

    final_info = {}

    done = False

    action_counts = Counter()

    # ---------------------------------------------------------
    # Run until done or max steps reached
    # ---------------------------------------------------------

    while not done and step_count < max_steps:
        _, reward, done, info = manager.step()

        action = int(info["action"])

        if not 0 <= action < len(ALE_ACTION_NAMES):
            raise RuntimeError(
                f"Invalid action: {action}. Expected 0-{len(ALE_ACTION_NAMES) - 1}."
            )

        action_counts[action] += 1

        # -----------------------------------------------------
        # Log first actions
        # -----------------------------------------------------

        if step_count < log_first_steps:
            print_step(
                step_count,
                action,
                float(reward),
            )

        # -----------------------------------------------------
        # Track statistics
        # -----------------------------------------------------

        total_reward += float(reward)

        step_count += 1

        final_info = info

    return {
        "steps": step_count,
        "reward": total_reward,
        "action_counts": action_counts,
        "final_info": final_info,
        "hit_max_steps": (step_count >= max_steps and not done),
    }


# ================================================================
# Evaluate policy over multiple episodes
#
# UNCHANGED
# ================================================================


def evaluate_policy(
    manager,
    num_episodes: int,
    max_steps: int,
) -> dict:
    """
    Run the GymManager's action classifier for num_episodes
    episodes.

    Print:
        Per-episode reports
        Overall summary
        Action frequencies
        Rewards

    Returns:
        {
            "episodes": [...],
            "rewards": [...],
            "total_action_counts": Counter,
        }
    """

    episodes = []

    # =========================================================
    # Run all episodes
    # =========================================================

    try:
        for episode in range(num_episodes):
            print_header(f"Episode {episode + 1}/{num_episodes}")

            episodes.append(
                run_episode(
                    manager,
                    max_steps,
                    log_first_steps=(30 if episode == 0 else 0),
                )
            )

    finally:
        manager.close()

    # =========================================================
    # Report results
    # =========================================================

    total_action_counts = Counter()

    for episode, stats in enumerate(
        episodes,
        start=1,
    ):
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

    # ---------------------------------------------------------
    # Reward summary
    # ---------------------------------------------------------

    rewards = [stats["reward"] for stats in episodes]

    print_summary_report(
        rewards,
        total_action_counts,
    )

    # ---------------------------------------------------------
    # Optional network report
    # ---------------------------------------------------------

    if hasattr(
        manager.action_net,
        "report",
    ):
        manager.action_net.report()

    return {
        "episodes": episodes,
        "rewards": rewards,
        "total_action_counts": total_action_counts,
    }


# TODO: everything between this line and the closing triple quote at the end of
# the file is an old copy of RuntimePreprocessor, RuntimeActionNet, GymManager,
# run_episode and evaluate_policy inside a string, so it never runs. Delete it
# so there is only one version to maintain.
'''
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
'''
