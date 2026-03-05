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
    d_dim  = (embed_dim // 3 // 2) * 2                  # round down to even
    hw_dim = ((embed_dim - d_dim) // 2 // 2) * 2        # round down to even
    w_dim  = embed_dim - d_dim - hw_dim                  # absorbs remainder (always even)

    def get_1d_sincos(n, dim):
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
        omega = 1.0 / (10000 ** omega)
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        return torch.cat([torch.sin(pos * omega), torch.cos(pos * omega)], dim=1)

    d_embed = get_1d_sincos(grid_depth, d_dim)   # [D, d_dim]
    h_embed = get_1d_sincos(grid_h,    hw_dim)   # [H, hw_dim]
    w_embed = get_1d_sincos(grid_w,    w_dim)    # [W, w_dim]

    d = d_embed[:, None, None, :].expand(-1, grid_h, grid_w, -1)      # [D, H, W, d_dim]
    h = h_embed[None, :, None, :].expand(grid_depth, -1, grid_w, -1)  # [D, H, W, hw_dim]
    w = w_embed[None, None, :, :].expand(grid_depth, grid_h, -1, -1)  # [D, H, W, w_dim]

    grid = torch.cat([d, h, w], dim=-1)   # [D, H, W, embed_dim]
    return grid.reshape(-1, embed_dim)    # [D*H*W, embed_dim]


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

    total_blocks    = N // block_size
    num_mask_blocks = int(total_blocks * drop_ratio)

    perm        = torch.rand(B, total_blocks, device=device).argsort(dim=1)
    mask_blocks = perm[:, :num_mask_blocks]  # [B, num_mask_blocks]

    block_offsets = torch.arange(block_size, device=device).view(1, 1, block_size)
    mask_idx = (mask_blocks.unsqueeze(-1) * block_size + block_offsets).view(B, -1)

    mask_bool = torch.zeros(B, N, device=device, dtype=torch.bool)
    mask_bool.scatter_(1, mask_idx, True)

    N_visible = int((~mask_bool[0]).sum())
    student_tokens = tubelets[~mask_bool].reshape(B, N_visible, D)

    return student_tokens, mask_bool
