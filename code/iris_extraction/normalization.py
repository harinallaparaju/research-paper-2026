"""
Iris Normalization using Daugman's Rubber-Sheet Model.

Unwraps the annular iris region into a fixed-size rectangular strip
by mapping from Cartesian to polar coordinates.

Reference:
    Daugman, J. (2004). How iris recognition works. IEEE TCSVT, 14(1), 21-30.
"""

import cv2
import numpy as np
from typing import Tuple

# Standard normalized iris dimensions
NORM_HEIGHT = 64   # radial resolution
NORM_WIDTH = 512   # angular resolution


def normalize_iris(
    image: np.ndarray,
    pupil_center: Tuple[int, int],
    pupil_radius: int,
    iris_center: Tuple[int, int],
    iris_radius: int,
    noise_mask: np.ndarray,
    output_height: int = NORM_HEIGHT,
    output_width: int = NORM_WIDTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unwrap iris annulus to a rectangular strip using Daugman's rubber-sheet model.

    For each point (r, theta) in the normalized image:
        x(r, theta) = (1-r) * pupil_x(theta) + r * iris_x(theta)
        y(r, theta) = (1-r) * pupil_y(theta) + r * iris_y(theta)

    This handles non-concentric pupil/iris boundaries correctly.

    Args:
        image: Grayscale input image
        pupil_center: (x, y) of pupil center
        pupil_radius: Pupil radius in pixels
        iris_center: (x, y) of iris center
        iris_radius: Iris radius in pixels
        noise_mask: Binary mask (1=valid, 0=noise), same size as image
        output_height: Radial resolution of normalized image (default 64)
        output_width: Angular resolution of normalized image (default 512)

    Returns:
        (normalized_image, normalized_mask): Both of shape (output_height, output_width)
    """
    h, w = image.shape[:2]
    px, py = pupil_center
    ix, iy = iris_center

    # Create output arrays
    normalized = np.zeros((output_height, output_width), dtype=np.uint8)
    norm_mask = np.zeros((output_height, output_width), dtype=np.uint8)

    # Precompute all theta values (angular positions)
    thetas = np.linspace(0, 2 * np.pi, output_width, endpoint=False)

    # Precompute all radial positions
    rs = np.linspace(0, 1, output_height, endpoint=False)

    # Pupil boundary points as function of theta
    pupil_x = px + pupil_radius * np.cos(thetas)  # (output_width,)
    pupil_y = py + pupil_radius * np.sin(thetas)

    # Iris boundary points as function of theta
    iris_x = ix + iris_radius * np.cos(thetas)    # (output_width,)
    iris_y = iy + iris_radius * np.sin(thetas)

    # Vectorized computation: for each (r, theta), compute the source (x, y)
    # rs: (H,), pupil_x: (W,) → broadcast to (H, W)
    rs_2d = rs[:, np.newaxis]  # (H, 1)

    # Rubber-sheet mapping
    src_x = ((1 - rs_2d) * pupil_x[np.newaxis, :] + rs_2d * iris_x[np.newaxis, :])
    src_y = ((1 - rs_2d) * pupil_y[np.newaxis, :] + rs_2d * iris_y[np.newaxis, :])

    # Bilinear interpolation via cv2.remap (much better than nearest-neighbor)
    map_x = np.clip(src_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(src_y, 0, h - 1).astype(np.float32)

    normalized = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Mask must use nearest-neighbor (binary values)
    if noise_mask.size > 0 and noise_mask.shape == image.shape[:2]:
        norm_mask = cv2.remap(noise_mask, map_x, map_y, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        norm_mask = np.ones((output_height, output_width), dtype=np.uint8)

    return normalized.astype(np.uint8), norm_mask.astype(np.uint8)
