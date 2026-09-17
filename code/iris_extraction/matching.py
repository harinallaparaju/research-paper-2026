"""
Iris Matching using Fractional Hamming Distance with masking and rotation.

Reference:
    Daugman, J. (2004). How iris recognition works. IEEE TCSVT, 14(1), 21-30.
"""

import numpy as np

# Default angular width of the normalized iris strip
NORM_WIDTH = 512


def fractional_hamming_distance(
    code1: np.ndarray, mask1: np.ndarray,
    code2: np.ndarray, mask2: np.ndarray,
    max_shift: int = 32,
    norm_width: int = NORM_WIDTH,
) -> float:
    """
    Compute Fractional Hamming Distance between two IrisCodes with
    masking and rotational alignment.

    FHD = min over shifts of: (XOR disagreements in valid region) / (valid bits)

    Rotation compensation: the flattened code is reshaped to
    (n_segments, norm_width) and each row is circularly shifted
    by the same number of columns. This correctly implements angular
    rotation across all segments (rows × wavelengths × Re/Im).

    Args:
        code1, mask1: Enrollment IrisCode and mask (1D binary arrays)
        code2, mask2: Probe IrisCode and mask (1D binary arrays)
        max_shift: Maximum column shifts to try in each direction (default ±8)
        norm_width: Angular width of the normalized strip (default 512)

    Returns:
        FHD value in [0, 1]. Lower = more similar.
        Returns 1.0 if no valid bits overlap.
    """
    if len(code1) != len(code2):
        raise ValueError(f"Code length mismatch: {len(code1)} vs {len(code2)}")
    if len(code1) % norm_width != 0:
        raise ValueError(f"Code length {len(code1)} not divisible by norm_width {norm_width}")

    n_segments = len(code1) // norm_width

    # Reshape to 2D: (n_segments, norm_width)
    c1_2d = code1.reshape(n_segments, norm_width)
    m1_2d = mask1.reshape(n_segments, norm_width)
    c2_2d = code2.reshape(n_segments, norm_width)
    m2_2d = mask2.reshape(n_segments, norm_width)

    min_dist = 1.0

    for shift in range(-max_shift, max_shift + 1):
        # Circular shift each row by `shift` columns
        c2_shifted = np.roll(c2_2d, shift, axis=1)
        m2_shifted = np.roll(m2_2d, shift, axis=1)

        # Valid region: both masks are 1
        valid = m1_2d & m2_shifted
        n_valid = np.sum(valid)

        if n_valid == 0:
            continue

        # Count disagreements in valid region
        disagreements = np.sum((c1_2d ^ c2_shifted) & valid)
        dist = disagreements / n_valid

        if dist < min_dist:
            min_dist = dist

    return float(min_dist)
