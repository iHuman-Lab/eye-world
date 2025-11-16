import numpy as np


def eye_gaze_to_density_image(empty_image, gaze_locations, config):
    # TODO:
    # 1. Create a multi-variate gaussian distribution at the eye_gaze
    # 2. Create a new variable in config file called sigma. This is the covariance of distribution
    # 3. Place the guassian distribution in the empty image at gaze_location.
    eye_gaze = gaze_locations[-1]
    x0, y0 = eye_gaze[-1]
    sigma = config["sigma"]
    H, W = empty_image.shape[:2]
    Y, X = np.meshgrid(np.arange(H), np.arrange(W), indexing="ij")
    gaussian = np.exp(
        -(((X - x0) ** 2) / (2 * sigma**2) + ((Y - y0) ** 2) / (2 * sigma**2))
    )
    gaussian /= gaussian.max()
    density_image = empty_image.astype(float) + gaussian

    return density_image
