"""
Iris Feature Encoding using Log-Gabor Filters.

Converts a normalized iris strip into a binary IrisCode by applying
Log-Gabor filters and quantizing the phase response.

Supports two encoding modes:
  - '1d' (default): 1D row-wise Log-Gabor, 2 wavelengths, 49152-bit code.
  - '2d' (enhanced): 2D Log-Gabor with multiple orientations, reliability-
    weighted fragile-bit masking, and adaptive CLAHE.  Produces denser,
    more discriminative codes with lower genuine FHD.

Reference:
    Masek, L. (2003). Recognition of human iris patterns for biometric
    identification. Thesis, University of Western Australia.
    Daugman, J. (2004). How iris recognition works. IEEE TCSVT.
    Ma, L. et al. (2004). Efficient iris recognition by characterizing key
    local variations. IEEE TIP.
"""

import numpy as np
import cv2
from typing import Tuple, Optional

# Default encoding parameters
DEFAULT_WAVELENGTHS = [18, 36]            # Log-Gabor center wavelengths (2 octaves)
DEFAULT_BANDWIDTH = 0.30                 # sigma_on_f parameter (narrower = more selective)
DEFAULT_CODE_ROWS = (0, 64)              # Full 64-row strip
DEFAULT_FRAG_THRESHOLD = 0.55            # Fragile-bit amplitude threshold factor

# Enhanced 2D encoding parameters
DEFAULT_2D_WAVELENGTHS = [9, 18, 36]          # 3 wavelengths (72 is too low-freq for 2D)
DEFAULT_2D_ORIENTATIONS = [0, 45, 90, 135]    # 4 orientations in degrees
DEFAULT_2D_BANDWIDTH = 0.55                    # slightly broader for 2D
DEFAULT_FRAG_PERCENTILE = 12                   # fragile-bit percentile threshold (higher = stricter)


def encode_iris(
    normalized: np.ndarray,
    norm_mask: np.ndarray,
    wavelengths: list = None,
    bandwidth: float = DEFAULT_BANDWIDTH,
    code_rows: Tuple[int, int] = DEFAULT_CODE_ROWS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a normalized iris strip into a binary IrisCode.

    Process:
    1. For each row in the selected band, apply 1D Log-Gabor filter
    2. Quantize the complex phase into 2 bits (Re>0, Im>0)
    3. Concatenate across rows and filter scales

    Args:
        normalized: Normalized iris strip (H x W), typically 64x512
        norm_mask: Binary mask (1=valid), same shape as normalized
        wavelengths: List of Log-Gabor center wavelengths
        bandwidth: Log-Gabor bandwidth parameter (sigma/f0)
        code_rows: (start, end) row indices to encode

    Returns:
        (iris_code, code_mask): Both 1D binary numpy arrays
        Code length = num_rows * width * num_wavelengths * 2
    """
    if wavelengths is None:
        wavelengths = DEFAULT_WAVELENGTHS

    h, w = normalized.shape
    row_start, row_end = code_rows
    row_start = max(0, row_start)
    row_end = min(h, row_end)
    selected_rows = range(row_start, row_end)
    n_rows = len(selected_rows)

    # CLAHE enhancement: equalizes local contrast across captures of the same iris
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 16))
    enhanced = clahe.apply(normalized.astype(np.uint8))

    # Mild Gaussian blur to suppress pixel-level sensor noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=1.0)

    code_bits = []
    mask_bits = []

    for wavelength in wavelengths:
        # Build 1D Log-Gabor filter in frequency domain
        log_gabor = _build_log_gabor_1d(w, wavelength, bandwidth)

        for row_idx in selected_rows:
            row = enhanced[row_idx, :].astype(np.float64)
            row_mask = norm_mask[row_idx, :]

            # Apply filter via FFT
            row_fft = np.fft.fft(row)
            filtered = np.fft.ifft(row_fft * log_gabor)

            re_part = np.real(filtered)
            im_part = np.imag(filtered)

            # Phase quantization: 2 bits per pixel
            bit_re = (re_part > 0).astype(np.uint8)
            bit_im = (im_part > 0).astype(np.uint8)

            # Fragile bit masking (Hollingsworth et al., 2009):
            # Bits near the phase boundary flip between captures.
            amplitude = np.sqrt(re_part**2 + im_part**2)
            amp_med = np.median(amplitude[amplitude > 0]) if np.any(amplitude > 0) else 1.0
            frag_thr = DEFAULT_FRAG_THRESHOLD * amp_med
            re_ok = (np.abs(re_part) > frag_thr).astype(np.uint8)
            im_ok = (np.abs(im_part) > frag_thr).astype(np.uint8)

            code_bits.append(bit_re)
            code_bits.append(bit_im)
            mask_bits.append(row_mask & re_ok)
            mask_bits.append(row_mask & im_ok)

    iris_code = np.concatenate(code_bits)
    code_mask = np.concatenate(mask_bits)

    return iris_code, code_mask


def _build_log_gabor_1d(length: int, wavelength: float, bandwidth: float) -> np.ndarray:
    """
    Construct a 1D ANALYTIC Log-Gabor filter in the frequency domain.

    Only positive frequencies have energy, so the IFFT produces an analytic
    (complex) signal with meaningful phase in both Re and Im components.

    The Log-Gabor transfer function:
        G(f) = exp(-log(f/f0)^2 / (2 * log(sigma/f0)^2))

    where f0 = 1/wavelength is the center frequency.

    Args:
        length: Length of the signal (number of columns)
        wavelength: Center wavelength in pixels
        bandwidth: sigma/f0 ratio (typically 0.5)

    Returns:
        1D complex filter in frequency domain, length `length`
    """
    f0 = 1.0 / wavelength
    sigma_on_f = bandwidth

    log_gabor = np.zeros(length, dtype=np.float64)

    # Only positive frequencies (analytic signal)
    for i in range(1, length // 2 + 1):
        f = i / length
        log_gabor[i] = np.exp(-(np.log(f / f0) ** 2) / (2 * np.log(sigma_on_f) ** 2))

    # Negative frequencies are zero → analytic signal after IFFT

    return log_gabor


def get_code_length(
    norm_height: int = 64,
    norm_width: int = 512,
    wavelengths: list = None,
    code_rows: Tuple[int, int] = DEFAULT_CODE_ROWS
) -> int:
    """Calculate the IrisCode length for given parameters."""
    if wavelengths is None:
        wavelengths = DEFAULT_WAVELENGTHS
    n_rows = code_rows[1] - code_rows[0]
    return n_rows * norm_width * len(wavelengths) * 2


# =========================================================================
# ENHANCED 2D ENCODING
# =========================================================================

def encode_iris_2d(
    normalized: np.ndarray,
    norm_mask: np.ndarray,
    wavelengths: list = None,
    orientations: list = None,
    bandwidth: float = DEFAULT_2D_BANDWIDTH,
    code_rows: Tuple[int, int] = DEFAULT_CODE_ROWS,
    frag_percentile: int = DEFAULT_FRAG_PERCENTILE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced iris encoding with 2D Log-Gabor filters.

    Improvements over 1D:
    1. 2D Log-Gabor filters capture orientation-specific texture, producing
       more discriminative bits.
    2. Reliability-weighted fragile bit masking: uses per-pixel confidence
       scores based on filter response amplitude relative to the *local*
       neighbourhood, not just the global median.
    3. Optimised CLAHE (higher clip limit for NIR iris) and bilateral
       filtering (edge-preserving noise reduction).

    Code layout:  for each (wavelength, orientation, row):
                      append 2 bits (Re>0, Im>0) × norm_width columns
    Total bits = n_rows × norm_width × n_wavelengths × n_orientations × 2

    Args:
        normalized: Normalized iris strip (H x W), typically 64x512
        norm_mask: Binary mask (1=valid), same shape
        wavelengths: Center wavelengths for Log-Gabor radial component
        orientations: Filter orientations in degrees
        bandwidth: Log-Gabor bandwidth (sigma/f0)
        code_rows: (start, end) row range to encode
        frag_percentile: Percentile of amplitude below which bits are
                         masked as fragile (higher = stricter, fewer valid bits)

    Returns:
        (iris_code, code_mask): 1D binary arrays
    """
    if wavelengths is None:
        wavelengths = DEFAULT_2D_WAVELENGTHS
    if orientations is None:
        orientations = DEFAULT_2D_ORIENTATIONS

    h, w = normalized.shape
    row_start, row_end = code_rows
    row_start = max(0, row_start)
    row_end = min(h, row_end)
    n_rows = row_end - row_start

    # --- Pre-processing (enhanced) ---
    # CLAHE: slightly higher clip for NIR iris contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 16))
    enhanced = clahe.apply(normalized.astype(np.uint8))

    # Bilateral filter: preserves edges while reducing noise
    # (better than Gaussian blur which smears iris texture boundaries)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=25, sigmaSpace=5)

    # Extract the ROI rows
    roi = enhanced[row_start:row_end, :].astype(np.float64)
    roi_mask = norm_mask[row_start:row_end, :]

    # Build all 2D Log-Gabor filters
    filters = []
    for wl in wavelengths:
        for theta_deg in orientations:
            lg2d = _build_log_gabor_2d(n_rows, w, wl, theta_deg, bandwidth)
            filters.append(lg2d)

    # Apply filters via 2D FFT
    roi_fft = np.fft.fft2(roi)

    code_bits = []
    mask_bits = []

    for lg2d in filters:
        filtered = np.fft.ifft2(roi_fft * lg2d)

        re_part = np.real(filtered)
        im_part = np.imag(filtered)

        # Phase quantization: 2 bits per pixel
        bit_re = (re_part > 0).astype(np.uint8)
        bit_im = (im_part > 0).astype(np.uint8)

        # --- Reliability-weighted fragile bit masking ---
        # Compute per-pixel amplitude (filter response strength)
        amplitude = np.sqrt(re_part**2 + im_part**2)

        # Global-percentile fragile mask (Hollingsworth-style but tuned):
        # Mask bits where the filter response is too weak (near phase boundary).
        # Use the global amplitude distribution across the entire ROI for this
        # filter, which is more stable than per-row thresholding.
        valid_amps = amplitude[roi_mask > 0]
        if len(valid_amps) > 10:
            thr = np.percentile(valid_amps, frag_percentile)
        else:
            thr = 0.0
        re_ok = (np.abs(re_part) > thr).astype(np.uint8)
        im_ok = (np.abs(im_part) > thr).astype(np.uint8)

        # Flatten row-by-row and append
        for r in range(n_rows):
            code_bits.append(bit_re[r, :])
            code_bits.append(bit_im[r, :])
            mask_bits.append(roi_mask[r, :] & re_ok[r, :])
            mask_bits.append(roi_mask[r, :] & im_ok[r, :])

    iris_code = np.concatenate(code_bits)
    code_mask = np.concatenate(mask_bits)

    return iris_code, code_mask


def _build_log_gabor_2d(
    height: int, width: int,
    wavelength: float, theta_deg: float,
    bandwidth: float,
    angular_bandwidth: float = 30.0,
) -> np.ndarray:
    """
    Build a 2D Log-Gabor filter in the frequency domain.

    The 2D Log-Gabor is the product of:
      - Radial Log-Gabor:   G_r(f) = exp(-log(f/f0)^2 / (2 log(sigma_r/f0)^2))
      - Angular Gaussian:   G_a(θ) = exp(-dtheta^2 / (2 sigma_a^2))

    Only positive half-plane frequencies are non-zero (analytic).

    Args:
        height: Number of rows in the patch
        width: Number of columns
        wavelength: Center wavelength in pixels
        theta_deg: Preferred orientation in degrees (0=horizontal)
        bandwidth: Radial bandwidth (sigma/f0)
        angular_bandwidth: Angular bandwidth in degrees (std of Gaussian)

    Returns:
        2D filter in frequency domain, shape (height, width), complex-compatible
    """
    f0 = 1.0 / wavelength
    sigma_on_f = bandwidth
    theta_rad = np.deg2rad(theta_deg)
    sigma_theta = np.deg2rad(angular_bandwidth)

    # Frequency grid
    u = np.fft.fftfreq(width)      # horizontal frequency
    v = np.fft.fftfreq(height)     # vertical frequency
    U, V = np.meshgrid(u, v)

    # Polar coordinates in frequency space
    radius = np.sqrt(U**2 + V**2)
    radius[0, 0] = 1.0  # Avoid log(0)
    angle = np.arctan2(V, U)

    # Radial component: Log-Gabor
    log_r = np.log(radius / f0)
    log_bw = np.log(sigma_on_f)
    radial = np.exp(-(log_r**2) / (2.0 * log_bw**2))

    # Angular component: wrapped Gaussian
    dtheta = angle - theta_rad
    # Wrap to [-pi, pi]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    angular = np.exp(-dtheta**2 / (2.0 * sigma_theta**2))

    # Combine
    lg2d = radial * angular

    # Zero DC
    lg2d[0, 0] = 0.0

    # Analytic signal: zero negative vertical frequencies so the IFFT
    # produces a complex signal with meaningful Re/Im phase components.
    # For very short height (e.g. 32 rows), the vertical frequency
    # resolution is coarse; only zero the clearly-negative half.
    if height > 2:
        lg2d[height // 2 + 1:, :] = 0.0

    return lg2d


def get_code_length_2d(
    norm_height: int = 64,
    norm_width: int = 512,
    wavelengths: list = None,
    orientations: list = None,
    code_rows: Tuple[int, int] = DEFAULT_CODE_ROWS,
) -> int:
    """Calculate the 2D IrisCode length for given parameters."""
    if wavelengths is None:
        wavelengths = DEFAULT_2D_WAVELENGTHS
    if orientations is None:
        orientations = DEFAULT_2D_ORIENTATIONS
    n_rows = code_rows[1] - code_rows[0]
    return n_rows * norm_width * len(wavelengths) * len(orientations) * 2
