import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def gaze_points_to_density(image_shape, gaze_points, sigma, config):
    """
    Convert gaze points to a smoothed density image using a Gaussian filter.

    Args:
        image_shape (tuple): Shape of the output image (H, W).
        gaze_points (list of tuples or array): List of (x, y) gaze coordinates.
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        np.ndarray: Smoothed and normalized density image.
    """
    H, W = image_shape[1], image_shape[2]
    impulses = np.zeros((H, W), dtype=float)

    gaze_points = np.array(gaze_points)
    if gaze_points.ndim > 1:
        # Convert gaze points to array and extract coordinates
        xs, ys = gaze_points[:, 0], gaze_points[:, 1]
    else:
        xs, ys = gaze_points[0], gaze_points[1]

    # Clip coordinates to valid indices
    xs = xs * (W / config["original_size"][0])
    ys = ys * (H / config["original_size"][1])
    xs = np.clip(xs, 0, W - 1).astype(int)
    ys = np.clip(ys, 0, H - 1).astype(int)

    # Place impulses
    impulses[ys, xs] += 1.0

    # Apply Gaussian filter and normalize
    density = gaussian_filter(impulses, sigma=sigma)
    max_val = density.max()
    if max_val != 0:
        density /= max_val

    return density


# TODO: Implement Fixation detection here.
# NOTE: This function should return the fixations for a given frame. No need for a class
# You can use some of the functions from the .utils, but the cluster is not needed.
# The output should look like [n x 2], where n is the number of fixations. Each frame can have
# any number of fixation, we do not control n.
# We do not update fixation from previous frame, we just detect for each frame.
def detect_fixations(gaze_locations, config):
    raise NotImplementedError


def eye_gaze_to_density_image(image_shape, gaze_locations, config):
    """
    Convert eye gaze points to a density image using Gaussian smoothing.

    Args:
        empty_image (np.ndarray): Base image to accumulate gaze density.
        gaze_locations (list): List of gaze sequences; uses last gaze sequence.
        config (dict): Configuration with key "sigma".

    Returns:
        np.ndarray: Density image.
    """

    # Create a new clusterer for this frame
    if config["use_fixations"]:
        impulses = detect_fixations(gaze_locations, config)
    else:
        impulses = gaze_locations[-1]

    sigma = config["sigma"]

    # Use last gaze sequence
    density = gaze_points_to_density(image_shape, impulses, sigma, config)

    density = torch.from_numpy(density)
    density = density.unsqueeze(0)

    return density
