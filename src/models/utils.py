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
