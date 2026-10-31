from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from models.vjepa import TransformerEncoder, TubeletEmbedding

# ================================================================
# Three-action history embedding
# ================================================================


class ActionHistoryEmbedding(nn.Module):
    """
    Converts the previous 3 actions into ONE 768-dimensional token.

    Input:
        actions: [B, 3]

    Example:
        actions[b] = [LEFT, LEFT, FIRE]

    Processing:
        [B, 3]
            ->
        one-hot
        [B, 3, 18]
            ->
        flatten
        [B, 54]
            ->
        MLP
            ->
        [B, 768]
            ->
        [B, 1, 768]

    Temporal ordering is preserved because the three one-hot vectors
    are concatenated rather than averaged.
    """

    def __init__(
        self,
        num_actions: int = 18,
        history_length: int = 3,
        token_dim: int = 768,
    ):
        super().__init__()

        self.num_actions = int(num_actions)
        self.history_length = int(history_length)
        self.token_dim = int(token_dim)

        input_dim = self.num_actions * self.history_length

        self.mlp = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
            ),
            nn.ReLU(),
            nn.Linear(
                128,
                512,
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                self.token_dim,
            ),
        )

    def forward(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:

        # -----------------------------------------------------
        # Expected:
        #
        # actions:
        # [B, 3]
        # -----------------------------------------------------

        if actions.ndim != 2:
            raise ValueError(
                "ActionHistoryEmbedding expected "
                f"[B, {self.history_length}], "
                f"got {tuple(actions.shape)}"
            )

        if actions.shape[1] != self.history_length:
            raise ValueError(
                f"Expected exactly {self.history_length} "
                f"previous actions, "
                f"got {tuple(actions.shape)}"
            )

        actions = actions.long()

        # -----------------------------------------------------
        # Validate raw ALE action IDs
        # -----------------------------------------------------

        if torch.any(actions < 0):
            raise ValueError("Action history contains negative action IDs.")

        if torch.any(actions >= self.num_actions):
            raise ValueError(
                f"Action history contains an action outside 0-{self.num_actions - 1}."
            )

        # -----------------------------------------------------
        # [B, 3]
        #
        # ->
        #
        # [B, 3, 18]
        # -----------------------------------------------------

        one_hot = F.one_hot(
            actions,
            num_classes=self.num_actions,
        ).float()

        # -----------------------------------------------------
        # Preserve temporal ordering.
        #
        # [B, 3, 18]
        #
        # ->
        #
        # [B, 54]
        # -----------------------------------------------------

        one_hot = one_hot.reshape(
            actions.shape[0],
            self.history_length * self.num_actions,
        )

        # -----------------------------------------------------
        # [B, 54]
        #
        # ->
        #
        # [B, 768]
        # -----------------------------------------------------

        token = self.mlp(one_hot)

        # -----------------------------------------------------
        # [B, 768]
        #
        # ->
        #
        # [B, 1, 768]
        # -----------------------------------------------------

        token = token.unsqueeze(1)

        return token


# ================================================================
# Action-conditioned JEPA encoder
# ================================================================


class ActionConditionedJEPAEncoder(nn.Module):
    """
    Uses:

        16 visual JEPA tubelets
        +
        1 action-history token

    as the input to the pretrained student Transformer.

    Input:

        frames:
            [B, 4, 1, 84, 84]

        previous_actions:
            [B, 3]

    Processing:

        frames
            ->
        TubeletEmbedding
            ->
        [B, 16, 768]

        previous_actions
            ->
        ActionHistoryEmbedding
            ->
        [B, 1, 768]

        concatenate
            ->
        [B, 17, 768]

        student Transformer
            ->
        [B, 17, 768]
    """

    def __init__(
        self,
        config: dict,
        num_actions: int = 18,
        jepa_checkpoint: str | Path | None = None,
        freeze_tubelet: bool = True,
        freeze_student: bool = True,
    ):
        super().__init__()

        self.config = config

        self.num_actions = int(num_actions)

        self.num_visual_tokens = 16

        self.num_action_tokens = 1

        self.total_tokens = self.num_visual_tokens + self.num_action_tokens

        self.embed_dim = 768

        self.freeze_tubelet = bool(freeze_tubelet)

        self.freeze_student = bool(freeze_student)

        # -----------------------------------------------------
        # JEPA input channel count
        # -----------------------------------------------------

        self.grey_scale = bool(
            config.get(
                "grey_scale_v",
                True,
            )
        )

        patch_dim = 1 if self.grey_scale else 3

        image_size = int(
            config.get(
                "image_size",
                84,
            )
        )

        # -----------------------------------------------------
        # Original JEPA modules
        # -----------------------------------------------------

        self.tubelet_embed = TubeletEmbedding(
            config=config,
            patch_dim=patch_dim,
            embed_dim=self.embed_dim,
            img_size=image_size,
        )

        self.student = TransformerEncoder(
            dim=self.embed_dim,
            depth=12,
            heads=12,
            mlp_dim=3072,
        )

        # -----------------------------------------------------
        # New three-action token
        # -----------------------------------------------------

        self.action_history_embed = ActionHistoryEmbedding(
            num_actions=self.num_actions,
            history_length=3,
            token_dim=self.embed_dim,
        )

        # -----------------------------------------------------
        # Load pretrained JEPA
        # -----------------------------------------------------

        if jepa_checkpoint is not None:
            self._load_jepa_checkpoint(Path(jepa_checkpoint))

        # -----------------------------------------------------
        # Freeze pretrained portions if requested
        # -----------------------------------------------------

        if self.freeze_tubelet:
            self._freeze_tubelet()

        if self.freeze_student:
            self._freeze_student()

    # =========================================================
    # Load pretrained JEPA checkpoint
    # =========================================================

    def _load_jepa_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> None:

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"JEPA checkpoint was not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        checkpoint_state = checkpoint.get(
            "state_dict",
            checkpoint,
        )

        # -----------------------------------------------------
        # Tubelet embedding
        #
        # Expected original Lightning keys:
        #
        # model.tubelet_embed....
        # -----------------------------------------------------

        tubelet_state = {}

        tubelet_prefixes = (
            "model.tubelet_embed.",
            "tubelet_embed.",
        )

        for key, value in checkpoint_state.items():
            for prefix in tubelet_prefixes:
                if key.startswith(prefix):
                    stripped_key = key[len(prefix) :]

                    tubelet_state[stripped_key] = value

                    break

        if not tubelet_state:
            raise RuntimeError(
                "Could not find TubeletEmbedding weights in the JEPA checkpoint."
            )

        self.tubelet_embed.load_state_dict(
            tubelet_state,
            strict=True,
        )

        # -----------------------------------------------------
        # Student Transformer
        #
        # Expected original Lightning keys:
        #
        # model.student....
        # -----------------------------------------------------

        student_state = {}

        student_prefixes = (
            "model.student.",
            "student.",
        )

        for key, value in checkpoint_state.items():
            for prefix in student_prefixes:
                if key.startswith(prefix):
                    stripped_key = key[len(prefix) :]

                    student_state[stripped_key] = value

                    break

        if not student_state:
            raise RuntimeError(
                "Could not find student "
                "TransformerEncoder weights "
                "in the JEPA checkpoint."
            )

        self.student.load_state_dict(
            student_state,
            strict=True,
        )

        print(
            "Loaded JEPA checkpoint:",
            checkpoint_path,
        )

    # =========================================================
    # Freezing
    # =========================================================

    def _freeze_tubelet(
        self,
    ) -> None:

        for parameter in self.tubelet_embed.parameters():
            parameter.requires_grad = False

        self.tubelet_embed.eval()

        print("Tubelet embedding frozen.")

    def _freeze_student(
        self,
    ) -> None:

        for parameter in self.student.parameters():
            parameter.requires_grad = False

        self.student.eval()

        print("Student transformer frozen.")

    def train(
        self,
        mode: bool = True,
    ):

        super().train(mode)

        # Lightning calls .train() recursively.
        #
        # Restore frozen modules to eval mode.

        if self.freeze_tubelet:
            self.tubelet_embed.eval()

        if self.freeze_student:
            self.student.eval()

        return self

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        frames: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        frames:
            [B, 4, 1, 84, 84]

        previous_actions:
            [B, 3]

        output:
            [B, 17, 768]
        """

        if frames.ndim != 5:
            raise ValueError(
                "ActionConditionedJEPAEncoder expected "
                "frames [B,T,C,H,W], "
                f"got {tuple(frames.shape)}"
            )

        # -----------------------------------------------------
        # Visual tubelets
        #
        # [B, 4, 1, 84, 84]
        #
        # ->
        #
        # [B, 16, 768]
        # -----------------------------------------------------

        if self.freeze_tubelet:
            with torch.no_grad():
                visual_tokens = self.tubelet_embed(frames)

        else:
            visual_tokens = self.tubelet_embed(frames)

        # -----------------------------------------------------
        # HARD-CODED expected JEPA shape
        # -----------------------------------------------------

        if visual_tokens.ndim != 3:
            raise RuntimeError(
                "Expected visual JEPA tokens "
                "[B,16,768], "
                f"got {tuple(visual_tokens.shape)}"
            )

        if visual_tokens.shape[1] != 16:
            raise RuntimeError(
                f"Expected exactly 16 visual tubelets, got {visual_tokens.shape[1]}"
            )

        if visual_tokens.shape[2] != 768:
            raise RuntimeError(
                f"Expected JEPA token dimension 768, got {visual_tokens.shape[2]}"
            )

        # -----------------------------------------------------
        # Previous-three-action token
        #
        # [B, 3]
        #
        # ->
        #
        # [B, 1, 768]
        # -----------------------------------------------------

        action_token = self.action_history_embed(previous_actions)

        # -----------------------------------------------------
        # Add action token to visual tokens
        #
        # [B,16,768]
        # +
        # [B,1,768]
        #
        # ->
        #
        # [B,17,768]
        # -----------------------------------------------------

        student_input = torch.cat(
            [
                visual_tokens,
                action_token,
            ],
            dim=1,
        )

        if student_input.shape[1:] != (
            17,
            768,
        ):
            raise RuntimeError(
                f"Expected student input [B,17,768], got {tuple(student_input.shape)}"
            )

        # -----------------------------------------------------
        # Student Transformer
        #
        # The previous three actions can now participate in
        # self-attention with all visual tubelets.
        # -----------------------------------------------------

        if self.freeze_student:
            with torch.no_grad():
                representation = self.student(student_input)

        else:
            representation = self.student(student_input)

        # -----------------------------------------------------
        # Final representation
        #
        # [B,17,768]
        # -----------------------------------------------------

        if representation.shape[1:] != (
            17,
            768,
        ):
            raise RuntimeError(
                f"Expected student output [B,17,768], got {tuple(representation.shape)}"
            )

        return representation


# ================================================================
# Action classifier
# ================================================================


class ActionNet(nn.Module):
    """
    Action classifier using the action-conditioned JEPA student
    representation.

    Existing classifier input:
        [B, 12, 84, 84]

    This represents:
        4 RGB frames stacked along channels.

    Internally:
        [B,12,84,84]
            ->
        [B,4,3,84,84]
            ->
        grayscale if required
            ->
        [B,4,1,84,84]

        +
        previous actions:
        [B,3]

            ->
        ActionConditionedJEPAEncoder
            ->
        [B,17,768]

            ->
        flatten
            ->
        [B,13056]

            ->
        FC classifier
            ->
        [B,18]

            ->
        [B,1,18]
    """

    def __init__(
        self,
        num_actions: int,
        config: dict,
        jepa_checkpoint: str | Path,
        freeze_tubelet: bool = True,
        freeze_student: bool = True,
    ):
        super().__init__()

        self.num_actions = int(num_actions)

        self.num_predictions = 1

        self.num_visual_tokens = 16

        self.num_action_tokens = 1

        self.total_tokens = 17

        self.embed_dim = 768

        self.flattened_dim = self.total_tokens * self.embed_dim

        # 17 * 768
        if self.flattened_dim != 13056:
            raise RuntimeError("Expected flattened JEPA representation to equal 13056.")

        self.grey_scale = bool(
            config.get(
                "grey_scale_v",
                True,
            )
        )

        # -----------------------------------------------------
        # Action-conditioned JEPA
        # -----------------------------------------------------

        self.encoder = ActionConditionedJEPAEncoder(
            config=config,
            num_actions=self.num_actions,
            jepa_checkpoint=jepa_checkpoint,
            freeze_tubelet=freeze_tubelet,
            freeze_student=freeze_student,
        )

        # -----------------------------------------------------
        # Fully connected classifier
        #
        # [B,17,768]
        #
        # ->
        #
        # [B,13056]
        #
        # ->
        #
        # [B,18]
        # -----------------------------------------------------

        self.action_head = nn.Sequential(
            nn.Flatten(
                start_dim=1,
            ),
            nn.LayerNorm(
                self.flattened_dim,
            ),
            nn.Linear(
                self.flattened_dim,
                4096,
            ),
            nn.LeakyReLU(),
            nn.Dropout(
                p=0.2,
            ),
            nn.Linear(
                4096,
                2048,
            ),
            nn.LeakyReLU(),
            nn.Dropout(
                p=0.2,
            ),
            nn.Linear(
                2048,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Dropout(
                p=0.1,
            ),
            nn.Linear(
                1024,
                512,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                512,
                self.num_actions,
            ),
        )

        print(
            "ActionNet visual tokens:",
            self.num_visual_tokens,
        )

        print(
            "ActionNet action-history tokens:",
            self.num_action_tokens,
        )

        print(
            "ActionNet total student tokens:",
            self.total_tokens,
        )

        print(
            "ActionNet flattened representation:",
            self.flattened_dim,
        )

    # =========================================================
    # Convert existing RGB stack into JEPA format
    # =========================================================

    def _prepare_frames(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # -----------------------------------------------------
        # Expected existing classifier format:
        #
        # [B,12,H,W]
        # -----------------------------------------------------

        if x.ndim != 4:
            raise ValueError(f"ActionNet expected [B,12,H,W], got {tuple(x.shape)}")

        (
            batch_size,
            channels,
            height,
            width,
        ) = x.shape

        if channels != 12:
            raise ValueError(
                "ActionNet expects four RGB frames "
                "(4 x 3 = 12 channels), "
                f"got {tuple(x.shape)}"
            )

        # -----------------------------------------------------
        # Restore temporal dimension
        #
        # [B,12,H,W]
        #
        # ->
        #
        # [B,4,3,H,W]
        # -----------------------------------------------------

        frames = x.reshape(
            batch_size,
            4,
            3,
            height,
            width,
        )

        # -----------------------------------------------------
        # Match grayscale JEPA training
        #
        # [B,4,3,H,W]
        #
        # ->
        #
        # [B,4,1,H,W]
        # -----------------------------------------------------

        if self.grey_scale:
            red = frames[
                :,
                :,
                0:1,
                :,
                :,
            ]

            green = frames[
                :,
                :,
                1:2,
                :,
                :,
            ]

            blue = frames[
                :,
                :,
                2:3,
                :,
                :,
            ]

            frames = 0.299 * red + 0.587 * green + 0.114 * blue

        return frames

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        x: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> torch.Tensor:

        # -----------------------------------------------------
        # Existing stacked image
        #
        # [B,12,84,84]
        #
        # ->
        #
        # [B,4,1,84,84]
        # -----------------------------------------------------

        frames = self._prepare_frames(x)

        # -----------------------------------------------------
        # Action-conditioned JEPA
        #
        # frames:
        # [B,4,1,84,84]
        #
        # previous_actions:
        # [B,3]
        #
        # ->
        #
        # [B,17,768]
        # -----------------------------------------------------

        representation = self.encoder(
            frames,
            previous_actions,
        )

        # -----------------------------------------------------
        # FC action classifier
        #
        # [B,17,768]
        #
        # ->
        #
        # [B,18]
        # -----------------------------------------------------

        logits = self.action_head(representation)

        # -----------------------------------------------------
        # Preserve old ActionNet output interface
        #
        # [B,18]
        #
        # ->
        #
        # [B,1,18]
        # -----------------------------------------------------

        logits = logits.unsqueeze(1)

        return logits
