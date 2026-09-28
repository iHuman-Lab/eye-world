def format_batch_for_vjepa(batch, config):
    stacked_imgs, stacked_gaze, stacked_actions = batch

    # -----------------------------
    # Get shapes
    # -----------------------------
    B, CT, H, W = stacked_imgs.shape
    T = stacked_actions.shape[1]  # sequence length
    C = CT // T  # channels per frame

    # -----------------------------
    # Reshape images: [B, C*T, H, W] → [B, T, C, H, W]
    # -----------------------------
    imgs = stacked_imgs.view(B, T, C, H, W)

    # If grayscale (C=1), squeeze channel dim
    if C == 1:
        imgs = imgs.squeeze(2)  # → [B, T, H, W]
    else:
        # If RGB, you may want to convert or keep as is
        # Option: average channels → grayscale
        imgs = imgs.mean(dim=2)  # → [B, T, H, W]

    # -----------------------------
    # Actions (already correct shape)
    # -----------------------------
    actions = stacked_actions  # [B, T]

    return imgs, actions, stacked_gaze


"""

def atari_to_gym(action):
    # NOOP
    if action == 0:
        return 0

    # FIRE has no direct gym equivalent, so treat it as NOOP
    elif action == 1:
        return 0

    # Basic directions
    elif action == 2:  # UP
        return 1

    elif action == 3:  # RIGHT
        return 2

    elif action == 4:  # LEFT
        return 3

    elif action == 5:  # DOWN
        return 4

    # Diagonal directions
    elif action == 6:  # UPRIGHT
        return 5

    elif action == 7:  # UPLEFT
        return 6

    elif action == 8:  # DOWNRIGHT
        return 7

    elif action == 9:  # DOWNLEFT
        return 8

    # If it does not match anything, default to NOOP
    else:
        return 0
"""

import torch


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
'''
