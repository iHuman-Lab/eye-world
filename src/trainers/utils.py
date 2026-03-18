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

    return imgs, actions
