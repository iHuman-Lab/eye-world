from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from models.vjepa import TransformerEncoder, TubeletEmbedding

# ================================================================
# Previous-three-action embedding
# ================================================================


class ActionHistoryEmbedding(nn.Module):
    """
    Converts the previous 3 ALE actions into ONE token that has the
    same dimensionality as a JEPA visual tubelet.

    Input:
        previous_actions: [B, 3]

    Processing:
        [B, 3]
            ->
        one-hot [B, 3, 18]
            ->
        flatten [B, 54]
            ->
        MLP
            ->
        [B, 768]
            ->
        [B, 1, 768]

    The order of the three actions is preserved because the three
    one-hot vectors are concatenated before the MLP.
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
        previous_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        previous_actions:
            [B, 3]

        returns:
            [B, 1, 768]
        """

        if previous_actions.ndim != 2:
            raise ValueError(
                "ActionHistoryEmbedding expected "
                "[B, 3], "
                f"got {tuple(previous_actions.shape)}"
            )

        if previous_actions.shape[1] != self.history_length:
            raise ValueError(
                f"Expected {self.history_length} previous actions, "
                f"got {tuple(previous_actions.shape)}"
            )

        previous_actions = previous_actions.long()

        # -----------------------------------------------------
        # Sanity check action IDs
        # -----------------------------------------------------

        if torch.any(previous_actions < 0):
            raise ValueError("Previous actions contain negative action IDs.")

        if torch.any(previous_actions >= self.num_actions):
            raise ValueError(
                f"Previous actions contain IDs outside 0-{self.num_actions - 1}."
            )

        # -----------------------------------------------------
        # [B, 3]
        #
        # ->
        #
        # [B, 3, 18]
        # -----------------------------------------------------

        one_hot = F.one_hot(
            previous_actions,
            num_classes=self.num_actions,
        ).float()

        # -----------------------------------------------------
        # [B, 3, 18]
        #
        # ->
        #
        # [B, 54]
        #
        # Action order is preserved.
        # -----------------------------------------------------

        one_hot = one_hot.reshape(
            previous_actions.shape[0],
            self.history_length * self.num_actions,
        )

        # -----------------------------------------------------
        # [B, 54]
        #
        # ->
        #
        # [B, 768]
        # -----------------------------------------------------

        action_token = self.mlp(one_hot)

        # -----------------------------------------------------
        # [B, 768]
        #
        # ->
        #
        # [B, 1, 768]
        # -----------------------------------------------------

        action_token = action_token.unsqueeze(1)

        return action_token


# ================================================================
# Action-conditioned JEPA action classifier
# ================================================================


class ActionNet(nn.Module):
    """
    Action classifier using:

        16 visual JEPA tubelets
            +
        1 previous-three-actions token

    Student input:
        [B, 17, 768]

    Student output:
        [B, 17, 768]

    Then:
        flatten
            ->
        [B, 13056]
            ->
        fully-connected action classifier
            ->
        [B, 18]
            ->
        [B, 1, 18]


    Expected image input from the existing classifier pipeline:

        [B, 12, 84, 84]

    representing:

        4 RGB frames x 3 channels
    """

    def __init__(
        self,
        num_actions: int,
        config: dict,
        jepa_checkpoint: str | Path | None = None,
        freeze_tubelet: bool = True,
        freeze_student: bool = True,
    ):
        super().__init__()

        # =====================================================
        # Basic dimensions
        # =====================================================

        self.num_actions = int(num_actions)

        self.num_predictions = 1

        self.num_visual_tokens = 16
        self.num_action_tokens = 1
        self.total_tokens = 17

        self.embed_dim = 768

        self.flattened_dim = self.total_tokens * self.embed_dim

        # 17 * 768 = 13056

        if self.flattened_dim != 13056:
            raise RuntimeError(
                "Expected flattened dimension "
                "17 * 768 = 13056, "
                f"got {self.flattened_dim}"
            )

        # =====================================================
        # Freeze configuration
        # =====================================================

        self.freeze_tubelet = bool(freeze_tubelet)

        self.freeze_student = bool(freeze_student)

        # =====================================================
        # JEPA input configuration
        # =====================================================

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

        # =====================================================
        # Existing JEPA modules
        # =====================================================

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

        # =====================================================
        # NEW:
        # previous three actions -> one 768-D token
        # =====================================================

        self.action_history_embed = ActionHistoryEmbedding(
            num_actions=self.num_actions,
            history_length=3,
            token_dim=self.embed_dim,
        )

        # =====================================================
        # Optional pretrained JEPA checkpoint
        # =====================================================

        if jepa_checkpoint is not None:
            self._load_jepa_checkpoint(Path(jepa_checkpoint))

        else:
            print(
                "No JEPA checkpoint provided. "
                "Using randomly initialized "
                "TubeletEmbedding and student."
            )

        # =====================================================
        # Freeze pretrained pieces if requested
        # =====================================================

        if self.freeze_tubelet:
            for parameter in self.tubelet_embed.parameters():
                parameter.requires_grad = False

            self.tubelet_embed.eval()

        if self.freeze_student:
            for parameter in self.student.parameters():
                parameter.requires_grad = False

            self.student.eval()

        # =====================================================
        # Action classification head
        #
        # [B, 17, 768]
        #
        # ->
        #
        # [B, 13056]
        #
        # ->
        #
        # [B, num_actions]
        # =====================================================

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

        print("Action-history JEPA ActionNet")

        print(
            "Visual tokens:",
            self.num_visual_tokens,
        )

        print(
            "Action-history tokens:",
            self.num_action_tokens,
        )

        print(
            "Student token count:",
            self.total_tokens,
        )

        print(
            "Embedding dimension:",
            self.embed_dim,
        )

        print(
            "Flattened dimension:",
            self.flattened_dim,
        )

        print(
            "Tubelet frozen:",
            self.freeze_tubelet,
        )

        print(
            "Student frozen:",
            self.freeze_student,
        )

    # =========================================================
    # Load pretrained JEPA
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

        # =====================================================
        # TubeletEmbedding
        # =====================================================

        tubelet_state = {}

        tubelet_prefixes = (
            "model.tubelet_embed.",
            "tubelet_embed.",
        )

        for (
            key,
            value,
        ) in checkpoint_state.items():
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

        # =====================================================
        # Student Transformer
        # =====================================================

        student_state = {}

        student_prefixes = (
            "model.student.",
            "student.",
        )

        for (
            key,
            value,
        ) in checkpoint_state.items():
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
            "Loaded pretrained JEPA:",
            checkpoint_path,
        )

    # =========================================================
    # Keep frozen modules in eval mode
    # =========================================================

    def train(
        self,
        mode: bool = True,
    ):

        super().train(mode)

        if self.freeze_tubelet:
            self.tubelet_embed.eval()

        if self.freeze_student:
            self.student.eval()

        return self

    # =========================================================
    # Convert existing classifier images into JEPA format
    # =========================================================

    def _prepare_frames(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            [B, 12, H, W]

        Output if grayscale JEPA:
            [B, 4, 1, H, W]

        Output if RGB JEPA:
            [B, 4, 3, H, W]
        """

        if images.ndim != 4:
            raise ValueError(
                f"ActionNet expected images [B,12,H,W], got {tuple(images.shape)}"
            )

        (
            batch_size,
            channels,
            height,
            width,
        ) = images.shape

        if channels != 12:
            raise ValueError(
                "Expected four stacked RGB frames "
                "= 12 channels, "
                f"got {tuple(images.shape)}"
            )

        # -----------------------------------------------------
        # [B, 12, H, W]
        #
        # ->
        #
        # [B, 4, 3, H, W]
        # -----------------------------------------------------

        frames = images.reshape(
            batch_size,
            4,
            3,
            height,
            width,
        )

        # -----------------------------------------------------
        # RGB -> grayscale if JEPA expects one channel
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
        images: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        images:
            [B,12,84,84]

        previous_actions:
            [B,3]

        returns:
            [B,1,num_actions]
        """

        # =====================================================
        # 1. Prepare visual input
        # =====================================================

        frames = self._prepare_frames(images)

        # =====================================================
        # 2. Visual tubelet embedding
        #
        # [B,4,1,84,84]
        #
        # ->
        #
        # [B,16,768]
        # =====================================================

        if self.freeze_tubelet:
            # Tubelet embedder is frozen and nothing before it
            # needs gradients, so no_grad is safe here.

            with torch.no_grad():
                visual_tokens = self.tubelet_embed(frames)

        else:
            visual_tokens = self.tubelet_embed(frames)

        # -----------------------------------------------------
        # Verify expected JEPA geometry
        # -----------------------------------------------------

        if visual_tokens.ndim != 3:
            raise RuntimeError(
                f"Expected visual tubelets [B,16,768], got {tuple(visual_tokens.shape)}"
            )

        if visual_tokens.shape[1] != 16:
            raise RuntimeError(
                f"Expected 16 visual tubelets, got {visual_tokens.shape[1]}"
            )

        if visual_tokens.shape[2] != 768:
            raise RuntimeError(
                "Expected tubelet embedding "
                "dimension 768, "
                f"got {visual_tokens.shape[2]}"
            )

        # =====================================================
        # 3. Previous-three-action token
        #
        # [B,3]
        #
        # ->
        #
        # [B,1,768]
        # =====================================================

        action_token = self.action_history_embed(previous_actions)

        # =====================================================
        # 4. Add action-history token to visual tubelets
        #
        # [B,16,768]
        # +
        # [B,1,768]
        #
        # ->
        #
        # [B,17,768]
        # =====================================================

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

        # =====================================================
        # 5. Student Transformer
        #
        # IMPORTANT:
        #
        # Do NOT put this in torch.no_grad(), even when the
        # student parameters are frozen.
        #
        # The action_history_embed is trainable, so gradients
        # must propagate THROUGH the frozen transformer and
        # back into the action-history MLP.
        #
        # Frozen student parameters have requires_grad=False,
        # so they themselves will not update.
        # =====================================================

        representation = self.student(student_input)

        # Expected:
        #
        # [B,17,768]

        if representation.shape[1:] != (
            17,
            768,
        ):
            raise RuntimeError(
                "Expected conditioned student "
                "representation [B,17,768], "
                f"got {tuple(representation.shape)}"
            )

        # =====================================================
        # 6. Fully-connected classifier
        #
        # [B,17,768]
        #
        # ->
        #
        # flatten [B,13056]
        #
        # ->
        #
        # [B,num_actions]
        # =====================================================

        logits = self.action_head(representation)

        # =====================================================
        # 7. Preserve classifier output interface
        #
        # [B,18]
        #
        # ->
        #
        # [B,1,18]
        # =====================================================

        logits = logits.unsqueeze(1)

        return logits
