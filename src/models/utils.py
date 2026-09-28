import torch
import torch.nn.functional as F


def flatten_softmax_reshape(x):
    """
    Flatten the tensor, apply softmax, and reshape back to the original shape.

    Args:
        x: Tensor of shape (batch, channels, H, W)

    Returns:
        Tensor of shape (batch, channels, H, W)
    """
    batch_size, channels, H, W = x.size()
    x_flat = x.view(batch_size, -1)
    x_softmax = F.log_softmax(x_flat, dim=1)
    x_reshaped = x_softmax.view(batch_size, channels, H, W)
    return x_reshaped


# -------------------------------------------------
# 3D Sin-Cos Positional Embedding
# -------------------------------------------------
def get_3d_sincos_pos_embed(grid_h, grid_w, grid_depth, embed_dim=1024):
    """Fixed 3D sin-cos positional embedding for a [D, H, W] grid.

    embed_dim is split across depth/height/width axes.
    All component dims are forced even so sincos is exact.
    """
    d_dim = (embed_dim // 3 // 2) * 2  # round down to even
    hw_dim = ((embed_dim - d_dim) // 2 // 2) * 2  # round down to even
    w_dim = embed_dim - d_dim - hw_dim  # absorbs remainder (always even)

    def get_1d_sincos(n, dim):
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
        omega = 1.0 / (10000**omega)
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        return torch.cat([torch.sin(pos * omega), torch.cos(pos * omega)], dim=1)

    d_embed = get_1d_sincos(grid_depth, d_dim)  # [D, d_dim]
    h_embed = get_1d_sincos(grid_h, hw_dim)  # [H, hw_dim]
    w_embed = get_1d_sincos(grid_w, w_dim)  # [W, w_dim]

    d = d_embed[:, None, None, :].expand(-1, grid_h, grid_w, -1)  # [D, H, W, d_dim]
    h = h_embed[None, :, None, :].expand(
        grid_depth, -1, grid_w, -1
    )  # [D, H, W, hw_dim]
    w = w_embed[None, None, :, :].expand(grid_depth, grid_h, -1, -1)  # [D, H, W, w_dim]

    grid = torch.cat([d, h, w], dim=-1)  # [D, H, W, embed_dim]
    return grid.reshape(-1, embed_dim)  # [D*H*W, embed_dim]


# -------------------------------------------------
# Block masking (vectorized, uniform across batch)
# -------------------------------------------------
def block_mask_tubelets_vectorized(tubelets, drop_ratio=0.5, block_size=2):
    """
    Vectorized block masking. All batch items receive the same number of masked
    tokens, which allows clean tensor slicing without padding.

    tubelets:   [B, N, D]
    drop_ratio: fraction of tubelets to mask
    block_size: number of contiguous tubelets per block

    Returns:
        student_tokens: [B, N_visible, D]
        mask_bool:      [B, N] bool, True = masked
    """
    B, N, D = tubelets.shape
    device = tubelets.device

    total_blocks = N // block_size
    num_mask_blocks = int(total_blocks * drop_ratio)

    perm = torch.rand(B, total_blocks, device=device).argsort(dim=1)
    mask_blocks = perm[:, :num_mask_blocks]  # [B, num_mask_blocks]

    block_offsets = torch.arange(block_size, device=device).view(1, 1, block_size)
    mask_idx = (mask_blocks.unsqueeze(-1) * block_size + block_offsets).view(B, -1)

    mask_bool = torch.zeros(B, N, device=device, dtype=torch.bool)
    mask_bool.scatter_(1, mask_idx, True)

    N_visible = int((~mask_bool[0]).sum())
    student_tokens = tubelets[~mask_bool].reshape(B, N_visible, D)

    return student_tokens, mask_bool


def atari_to_gym(actions):
    """
    Convert full ALE action IDs (0-17) to the
    6-action Space Invaders Gym/ALE action space.

    Gym SpaceInvaders:
        0 = NOOP
        1 = FIRE
        2 = RIGHT
        3 = LEFT
        4 = RIGHTFIRE
        5 = LEFTFIRE
    """

    if torch.is_tensor(actions):
        actions = actions.long()

        # Default unsupported actions to NOOP
        gym_actions = torch.zeros_like(actions, dtype=torch.long)

        gym_actions[actions == 0] = 0  # NOOP
        gym_actions[actions == 1] = 1  # FIRE

        gym_actions[actions == 3] = 2  # RIGHT
        gym_actions[actions == 4] = 3  # LEFT

        gym_actions[actions == 11] = 4  # RIGHTFIRE
        gym_actions[actions == 12] = 5  # LEFTFIRE

        return gym_actions

    mapping = {
        0: 0,  # NOOP
        1: 1,  # FIRE
        3: 2,  # RIGHT
        4: 3,  # LEFT
        11: 4,  # RIGHTFIRE
        12: 5,  # LEFTFIRE
    }

    return mapping.get(actions, 0)


'''
def atari_to_gym(actions):
    """
    Convert ALE/Atari action IDs to Gym-style action IDs.

    Input:
        actions: torch.Tensor of shape [B] or scalar int

    Output:
        torch.LongTensor with Gym action IDs
    """

    if torch.is_tensor(actions):
        actions = actions.long()

        gym_actions = torch.zeros_like(actions, dtype=torch.long)

        # Atari 0 NOOP -> Gym 0 NOOP
        gym_actions[actions == 0] = 0

        # Atari 1 FIRE -> Gym 0 NOOP
        gym_actions[actions == 1] = 0

        # Basic movement
        gym_actions[actions == 2] = 1  # UP
        gym_actions[actions == 3] = 2  # RIGHT
        gym_actions[actions == 4] = 3  # LEFT
        gym_actions[actions == 5] = 4  # DOWN

        # Diagonal movement
        gym_actions[actions == 6] = 5  # UPRIGHT
        gym_actions[actions == 7] = 6  # UPLEFT
        gym_actions[actions == 8] = 7  # DOWNRIGHT
        gym_actions[actions == 9] = 8  # DOWNLEFT

        # FIRE + movement -> movement only
        gym_actions[actions == 10] = 1  # UPFIRE
        gym_actions[actions == 11] = 2  # RIGHTFIRE
        gym_actions[actions == 12] = 3  # LEFTFIRE
        gym_actions[actions == 13] = 4  # DOWNFIRE
        gym_actions[actions == 14] = 5  # UPRIGHTFIRE
        gym_actions[actions == 15] = 6  # UPLEFTFIRE
        gym_actions[actions == 16] = 7  # DOWNRIGHTFIRE
        gym_actions[actions == 17] = 8  # DOWNLEFTFIRE

        return gym_actions

    # Fallback for single integer input
    if actions == 0:
        return 0
    if actions == 1:
        return 0
    if actions == 2:
        return 1
    if actions == 3:
        return 2
    if actions == 4:
        return 3
    if actions == 5:
        return 4
    if actions == 6:
        return 5
    if actions == 7:
        return 6
    if actions == 8:
        return 7
    if actions == 9:
        return 8
    if actions == 10:
        return 1
    if actions == 11:
        return 2
    if actions == 12:
        return 3
    if actions == 13:
        return 4
    if actions == 14:
        return 5
    if actions == 15:
        return 6
    if actions == 16:
        return 7
    if actions == 17:
        return 8

    raise ValueError(f"Unknown Atari action: {actions}")
''' """'

# Canonical action names in the order ActionNet's 9-way output uses
# (index == the Gym-style action ID produced by atari_to_gym).
CANONICAL_ACTION_NAMES = (
    "NOOP",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
)
"""


def map_canonical_actions(action_meanings) -> list:
    """
    Build a lookup table from ActionNet's fixed 9-way canonical action
    ID to whatever action IDs the current Gym env actually exposes.

    Different games expose different minimal action sets (e.g. Breakout
    is ["NOOP", "FIRE", "RIGHT", "LEFT"], MsPacman has all 9 directions),
    so this must be recomputed per env rather than assumed to be 1:1.

    Input:
        action_meanings: list[str] from env.unwrapped.get_action_meanings()

    Output:
        list[int] of length 9, where mapping[canonical_id] is the action
        ID to send to env.step().
    """

    noop_id = action_meanings.index("NOOP") if "NOOP" in action_meanings else 0

    mapping = []

    for name in CANONICAL_ACTION_NAMES:
        if name in action_meanings:
            mapping.append(action_meanings.index(name))
            continue

        # Diagonal/absent directions: fall back to a component direction
        # the env does support (e.g. UPRIGHT -> RIGHT), else NOOP.
        matched = noop_id

        for component in ("RIGHT", "LEFT", "UP", "DOWN"):
            if component in name and component in action_meanings:
                matched = action_meanings.index(component)
                break

        mapping.append(matched)

    return mapping
