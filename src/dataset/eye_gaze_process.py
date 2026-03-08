import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN


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


# NOTE: Verify the pipeline by training the eye-gaze predictor
def detect_fixations(
    frame_gaze, maxdist=25, missing=0.0, sampling_rate=60, min_duration=0.1
):
    """
    Detect fixations using DBSCAN with spatial and temporal constraints.

    Args:
        frame_gaze: Array of (x, y) gaze points in temporal order
        maxdist: Maximum spatial distance for clustering (pixels)
        missing: Value indicating missing data
        sampling_rate: Samples per second (Hz)
        min_duration: Minimum fixation duration (seconds)

    Returns:
        Array of fixation centroids (x, y)
    """
    # Build points with indices to track time
    valid = [
        (i, p) for i, p in enumerate(frame_gaze) if p[0] != missing and p[1] != missing
    ]

    if len(valid) == 0:
        return np.empty((0, 2))

    indices, points = zip(*valid)
    indices = np.array(indices)
    points = np.array(points)

    # Add time as 3rd dimension, scaled so 1 sample gap = maxdist
    # This prevents clustering of spatially-close but temporally-distant points
    time_scale = maxdist
    times = (indices * time_scale).reshape(-1, 1)

    features = np.hstack([points, times])

    # DBSCAN on (x, y, t) space
    min_samples = max(1, int(min_duration * sampling_rate))
    labels = DBSCAN(eps=maxdist, min_samples=min_samples).fit_predict(features)

    # Filter out noise (label -1) and compute centroids
    unique_labels = [k for k in np.unique(labels) if k != -1]

    if len(unique_labels) == 0:
        return np.empty((0, 2))

    fixations = np.array([points[labels == k].mean(axis=0) for k in unique_labels])

    return fixations


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
        impulses = detect_fixations(
            gaze_locations,
            maxdist=config.get("maxdist", 25),
            missing=config.get("missing", 0.0),
            sampling_rate=config.get("sampling_rate", 60),
            min_duration=config.get("min_duration", 0.1),
        )
    else:
        impulses = gaze_locations[-1]

    sigma = config["sigma"]

    # Use last gaze sequence
    density = gaze_points_to_density(image_shape, impulses, sigma, config)

    density = torch.from_numpy(density)
    density = density.unsqueeze(0)

    return density
