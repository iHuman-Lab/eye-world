import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .utils import OnlineClusterer

''''
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
    """
    H, W = image_shape[1], image_shape[2]
    impulses = np.zeros((H, W), dtype=float)

    gaze_points = np.array(gaze_points)
    if gaze_points.ndim > 1:
        # Convert gaze points to array and extract coordinates
        xs, ys = gaze_points[:, 0], gaze_points[:, 1]
    else:
        xs, ys = gaze_points[0], gaze_points[1]
        max_len = 5
    gaze_window = deque(maxlen=max_len)

    if len(gaze_window) < 5:
        while len(gaze_window) < max_len:
            gaze_window.append(gaze_points)
    else:
        gaze_window.append(gaze_points)
        gaze_window = torch.cat(list(gaze_window), dim=0)

    df = pd.DataFrame({"avg_x": xs, "avg_y": ys, "time": times})
    fixations = detect_fixations(df, maxdist=25, mindur=50)
    """

    H, W = image_shape[0], image_shape[1]
    impulses = np.zeros((H, W), dtype=float)

    # Ensure gaze_points is a 2D array
    gaze_points = np.array(gaze_points)
    if gaze_points.ndim == 1:
        gaze_points = gaze_points.reshape(1, 2)
    xs, ys = gaze_points[:, 0], gaze_points[:, 1]
    # Clip coordinates to valid indices
    xs = xs * (W / config["original_size"][0])
    ys = ys * (H / config["original_size"][1])
    xs = np.clip(xs, 0, W - 1).astype(int)
    ys = np.clip(ys, 0, H - 1).astype(int)


    impulses[ys, xs] += 1.0

    # Apply Gaussian filter and normalize
    density = gaussian_filter(impulses, sigma=sigma)
    max_val = density.max()
    if max_val != 0:
        density /= max_val

    return density
'''


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
    clusterer = OnlineClusterer(maxdist=config.get("maxdist", 25))

    # Update clusters with all points in this frame
    for p in gaze_locations:
        clusterer.update(p)
    impulses = np.array(
        [
            c["centroid"]
            for c in clusterer.clusters
            if c["count"] >= config.get("min_cluster_count", 1)
        ]
    )
    sigma = config["sigma"]

    # Use last gaze sequence
    density = gaze_points_to_density(image_shape, impulses, sigma, config)

    density = torch.from_numpy(density)
    density = density.unsqueeze(0)

    return density


"""
    max_len = config.get("max_len", 5)
    # Initialize deque
    gaze_window = deque(maxlen=max_len)

    for x, y in zip(xs, ys):
        if len(gaze_window) < max_len:
            while len(gaze_window) < max_len:
                gaze_window.append((x, y))
        else:
            gaze_window.append((x, y))

    # Convert deque to DataFrame
    df_window = pd.DataFrame(gaze_window, columns=["avg_x", "avg_y"])

    # time = position in the deque
    df_window["time"] = np.arange(len(df_window))

    # Detect fixations using your provided function
    impulses = detect_fixations(
        df_window, maxdist=config.get("maxdist", 25), mindur=config.get("mindur", 50)
    )
        # Place impulses
    H, W = image_shape[:2]
    impulses = np.zeros((H, W), dtype=np.float32)

    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
"""
