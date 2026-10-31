import torch
from torch import nn

# ================================================================
# Action network
# ================================================================


class ActionNet(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=32,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )

        # An 84x84 input produces 64 x 7 x 7 = 3136 features.
        self.num_actions = num_actions
        self.num_predictions = 1

        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.LeakyReLU(),
            nn.Linear(
                512,
                self.num_predictions * self.num_actions,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)

        # [batch, 36] -> [batch, 4, 9]
        x = x.reshape(
            x.shape[0],
            self.num_predictions,
            self.num_actions,
        )

        return x


# ================================================================
# Action-conditioned world-model predictor
# ================================================================


class ACPredictor(nn.Module):
    """
    Action-conditioned latent predictor (after V-JEPA 2-AC,
    facebookresearch/vjepa2 src/models/ac_predictor.py).

    One time step is the latent of the current state (N patch tokens)
    plus the action taken in that state. Per step the token layout is

        [action token, patch_1 ... patch_N]

    and attention is block-causal across steps: a token at step s sees
    every token of steps <= s and nothing later. The output at step s is
    the prediction of the latent at step s + 1.

    Input:
        latents: [B, S, N, embed_dim]  encoder latents of steps 0..S-1
        actions: [B, S]                raw ALE action ids (0-17) taken
                                       in those steps
    Output:
        [B, S, N, embed_dim]           predicted latents of steps 1..S
    """

    def __init__(
        self,
        embed_dim: int = 768,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_actions: int = 18,
        num_patches: int = 16,
        max_steps: int = 8,
    ):
        super().__init__()

        self.num_patches = int(num_patches)
        self.max_steps = int(max_steps)

        self.in_proj = nn.Linear(embed_dim, dim)
        self.action_embed = nn.Embedding(num_actions, dim)

        self.patch_pos = nn.Parameter(torch.zeros(self.num_patches, dim))
        self.time_pos = nn.Parameter(torch.zeros(self.max_steps, dim))
        nn.init.trunc_normal_(self.patch_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            activation="gelu",
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, embed_dim)

    @staticmethod
    def block_causal_mask(steps: int, tokens_per_step: int, device) -> torch.Tensor:
        """[L, L] bool mask, True = not allowed: key step later than query step."""
        step = torch.arange(steps, device=device).repeat_interleave(tokens_per_step)

        return step[None, :] > step[:, None]

    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError(
                f"Expected latents [B, S, N, D], got {tuple(latents.shape)}"
            )

        B, S, N, _ = latents.shape

        if N != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {N}")

        if S > self.max_steps:
            raise ValueError(f"Got {S} steps but max_steps={self.max_steps}")

        if actions.shape != (B, S):
            raise ValueError(f"Expected actions [{B}, {S}], got {tuple(actions.shape)}")

        x = self.in_proj(latents)  # [B, S, N, dim]
        x = x + self.patch_pos[None, None] + self.time_pos[:S][None, :, None]

        a = self.action_embed(actions.long()) + self.time_pos[:S]  # [B, S, dim]

        tokens = torch.cat([a.unsqueeze(2), x], dim=2)  # [B, S, 1 + N, dim]
        tokens = tokens.flatten(1, 2)  # [B, S * (1 + N), dim]

        mask = self.block_causal_mask(S, 1 + N, tokens.device)
        out = self.blocks(tokens, mask=mask)

        # Drop the action tokens, keep one prediction per patch.
        out = out.view(B, S, 1 + N, -1)[:, :, 1:]

        return self.out_proj(self.out_norm(out))
