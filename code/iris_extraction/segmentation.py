"""
Iris Segmentation for CASIA-Iris-Interval images.

Detects pupil boundary, iris boundary, and eyelid occlusions
from 320x280 grayscale NIR images.

References:
    - Daugman, J. (2004). How iris recognition works. IEEE TCSVT.
    - Masek, L. (2003). Recognition of human iris patterns. Thesis.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SegmentationResult:
    """Result of iris segmentation."""
    pupil_center: Tuple[int, int]   # (x, y)
    pupil_radius: int
    iris_center: Tuple[int, int]    # (x, y)
    iris_radius: int
    noise_mask: np.ndarray          # same size as image, 1=valid, 0=occluded
    success: bool
    failure_reason: Optional[str] = None


def segment_iris(image: np.ndarray) -> SegmentationResult:
    """
    Segment pupil and iris boundaries from a CASIA-Iris-Interval image.

    Args:
        image: Grayscale uint8 image (320x280 for CASIA-Iris-Interval)

    Returns:
        SegmentationResult with boundaries and noise mask
    """
    if image is None or image.size == 0:
        return _fail("Empty image")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape

    # --- Step 1: Find pupil ---
    pupil = _find_pupil(image)
    if pupil is None:
        return _fail("Pupil detection failed")
    px, py, pr = pupil

    # --- Step 2: Find iris boundary ---
    iris = _find_iris(image, px, py, pr)
    if iris is None:
        return _fail("Iris boundary detection failed")
    ix, iy, ir = iris

    # Sanity checks
    if ir <= pr + 10:
        return _fail(f"Iris radius ({ir}) too close to pupil ({pr})")
    if ir > min(h, w) // 2:
        return _fail(f"Iris radius ({ir}) unreasonably large")

    # --- Step 3: Build noise mask (eyelid + reflection masking) ---
    noise_mask = _build_noise_mask(image, px, py, pr, ix, iy, ir)

    return SegmentationResult(
        pupil_center=(px, py),
        pupil_radius=pr,
        iris_center=(ix, iy),
        iris_radius=ir,
        noise_mask=noise_mask,
        success=True
    )


def _fail(reason: str) -> SegmentationResult:
    return SegmentationResult(
        pupil_center=(0, 0), pupil_radius=0,
        iris_center=(0, 0), iris_radius=0,
        noise_mask=np.array([]), success=False,
        failure_reason=reason
    )


def _find_pupil(image: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Detect pupil as the largest dark circular region.
    CASIA-Iris-Interval: pupil is very dark in NIR, with bright LED reflections.
    """
    h, w = image.shape

    # Remove NIR LED reflections by inpainting bright spots
    _, reflection_mask = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    reflection_mask = cv2.dilate(reflection_mask, np.ones((5, 5), np.uint8))
    clean = cv2.inpaint(image, reflection_mask, 5, cv2.INPAINT_TELEA)

    # Blur to smooth
    blurred = cv2.GaussianBlur(clean, (7, 7), 2)

    # Threshold for dark pupil region
    # CASIA pupil is typically <80 intensity
    _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find the largest connected component (should be pupil)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: try Hough circles on the blurred image
        return _hough_pupil_fallback(blurred, h, w)

    # Get largest contour by area
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Minimum area check (pupil should be substantial)
    if area < 500:
        return _hough_pupil_fallback(blurred, h, w)

    # Fit minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cx, cy, radius = int(cx), int(cy), int(radius)

    # Validate: pupil should be roughly in the center region of the image
    if not (w * 0.15 < cx < w * 0.85 and h * 0.15 < cy < h * 0.85):
        return _hough_pupil_fallback(blurred, h, w)

    # Validate radius range (typical CASIA pupil: 20-80 pixels)
    if not (15 <= radius <= 90):
        return _hough_pupil_fallback(blurred, h, w)

    return (cx, cy, radius)


def _hough_pupil_fallback(blurred: np.ndarray, h: int, w: int) -> Optional[Tuple[int, int, int]]:
    """Fallback pupil detection using Hough Circle Transform."""
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=w // 3,
        param1=80, param2=30,
        minRadius=20, maxRadius=80
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Pick the circle closest to the image center
    img_cx, img_cy = w // 2, h // 2
    best = min(circles, key=lambda c: (c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2)
    return (int(best[0]), int(best[1]), int(best[2]))


def _find_iris(image: np.ndarray, px: int, py: int, pr: int) -> Optional[Tuple[int, int, int]]:
    """
    Detect iris outer boundary using a combination of:
    1. Integrodifferential operator (gradient on concentric circles)
    2. Intensity contrast (iris is darker than sclera in NIR)

    The iris center is searched in a small window around the pupil center.
    The radius is found by looking for the maximum contrast change.
    """
    h, w = image.shape
    from scipy.ndimage import gaussian_filter1d

    # Smooth image for boundary detection
    blurred = cv2.GaussianBlur(image, (5, 5), 2).astype(np.float64)

    # Search range for iris radius
    min_r = pr + 20
    max_r = min(int(pr * 4.0), min(h, w) // 2 - 5, 150)

    if min_r >= max_r:
        return None

    best_score = -np.inf
    best_params = None

    # Search over candidate centers near the pupil
    for dx in range(-6, 7, 3):
        for dy in range(-6, 7, 3):
            cx, cy = px + dx, py + dy

            # Compute intensity profile: mean intensity on circles at each radius
            n_radii = max_r - min_r
            intensity_profile = np.zeros(n_radii)

            for idx, r in enumerate(range(min_r, max_r)):
                n_pts = max(100, int(2 * np.pi * r * 0.9))
                thetas = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                xs = np.clip((cx + r * np.cos(thetas)).astype(int), 0, w - 1)
                ys = np.clip((cy + r * np.sin(thetas)).astype(int), 0, h - 1)
                intensity_profile[idx] = np.mean(blurred[ys, xs])

            # Smooth the intensity profile
            smoothed = gaussian_filter1d(intensity_profile, sigma=2)

            # The iris-sclera boundary is where intensity INCREASES sharply
            # (iris is darker, sclera is brighter in NIR)
            # Compute the derivative (rate of intensity change)
            derivative = np.gradient(smoothed)

            # Smooth the derivative
            deriv_smooth = gaussian_filter1d(derivative, sigma=3)

            # Find the peak in the positive derivative (darkest to brightest transition)
            # Weight towards expected iris/pupil ratio (2.0-3.0)
            expected_r = pr * 2.5
            weights = np.exp(-0.5 * ((np.arange(n_radii) + min_r - expected_r) / (pr * 0.8)) ** 2)

            # Score: positive derivative (bright transition) weighted by expected radius
            score_profile = deriv_smooth * weights

            peak_idx = np.argmax(score_profile)
            peak_score = score_profile[peak_idx]

            if peak_score > best_score:
                best_score = peak_score
                best_params = (cx, cy, min_r + peak_idx)

    if best_params is None:
        return None

    # Validate: iris/pupil ratio should be reasonable (1.8 - 3.5)
    iris_r = best_params[2]
    ratio = iris_r / pr
    if ratio < 1.6 or ratio > 4.0:
        # Try concentric with most common ratio (2.5)
        fallback_r = int(pr * 2.5)
        if min_r <= fallback_r <= max_r:
            return (px, py, fallback_r)
        return None

    return best_params


def _build_noise_mask(
    image: np.ndarray,
    px: int, py: int, pr: int,
    ix: int, iy: int, ir: int
) -> np.ndarray:
    """
    Build noise mask: 1 = valid iris texture, 0 = occluded/noise.
    Masks out: pupil, outside iris, eyelids, specular reflections.
    """
    h, w = image.shape
    mask = np.ones((h, w), dtype=np.uint8)

    # Mask outside iris
    iris_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_mask, (ix, iy), ir, 1, -1)
    mask = mask & iris_mask

    # Mask pupil
    cv2.circle(mask, (px, py), pr + 2, 0, -1)

    # Mask specular reflections (very bright spots)
    _, reflections = cv2.threshold(image, 210, 1, cv2.THRESH_BINARY)
    reflections = cv2.dilate(reflections, np.ones((7, 7), np.uint8))
    mask = mask & (1 - reflections)

    # Mask eyelids: upper and lower regions of the iris tend to be occluded
    # Use simple approach: top 25% and bottom 25% of iris annulus are suspect
    # Refine with edge detection for eyelid boundaries
    _mask_eyelids(image, mask, ix, iy, ir)

    return mask


def _fit_parabola_ransac(
    points: np.ndarray,
    n_iter: int = 200,
    threshold: float = 3.0,
    min_inliers: int = 10,
) -> "Optional[Tuple[float, float, float]]":
    """
    Fit y = a*x^2 + b*x + c to 2D points using RANSAC.

    Args:
        points: Nx2 array of (x, y) coordinates
        n_iter: Number of RANSAC iterations
        threshold: Inlier distance in pixels
        min_inliers: Minimum inliers to accept a fit

    Returns:
        (a, b, c) coefficients, or None if fitting fails
    """
    if len(points) < min_inliers:
        return None

    xs = points[:, 0].astype(np.float64)
    ys = points[:, 1].astype(np.float64)
    n = len(xs)
    best_inliers = 0
    best_mask = None

    rng = np.random.RandomState(42)

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        sx, sy = xs[idx], ys[idx]

        A = np.column_stack([sx ** 2, sx, np.ones(3)])
        try:
            params = np.linalg.solve(A, sy)
        except np.linalg.LinAlgError:
            continue

        predicted = params[0] * xs ** 2 + params[1] * xs + params[2]
        residuals = np.abs(ys - predicted)
        inlier_mask = residuals < threshold
        n_inliers = int(inlier_mask.sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_mask = inlier_mask

    if best_mask is None or best_inliers < min_inliers:
        return None

    # Refit using all inliers (least-squares)
    inlier_xs = xs[best_mask]
    inlier_ys = ys[best_mask]
    A = np.column_stack([inlier_xs ** 2, inlier_xs, np.ones(len(inlier_xs))])
    params, _, _, _ = np.linalg.lstsq(A, inlier_ys, rcond=None)

    return (float(params[0]), float(params[1]), float(params[2]))


def _mask_eyelids(
    image: np.ndarray, mask: np.ndarray,
    ix: int, iy: int, ir: int
) -> None:
    """
    Detect and mask eyelid occlusions using intensity-based column-by-column
    texture transition detection.

    For each column in the iris annulus, scans for the transition from
    smooth eyelid skin (low local variance) to textured iris (high variance).
    This approach dramatically reduces impostor variance compared to
    RANSAC parabola fitting.

    Modifies mask in-place.
    """
    h, w = image.shape

    # Region of interest: iris bounding box
    y1 = max(0, iy - ir)
    y2 = min(h, iy + ir)
    x1 = max(0, ix - ir)
    x2 = min(w, ix + ir)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return

    roi_h, roi_w = roi.shape
    mid_y = roi_h // 2

    blurred = cv2.GaussianBlur(roi, (5, 5), 1.5)

    # Upper eyelid: scan from top down per column
    for col in range(roi_w):
        column = blurred[:mid_y, col].astype(np.float64)
        if len(column) < 10:
            continue

        # Compute local variance in sliding window
        win = 7
        if len(column) < win * 2:
            continue
        local_var = np.array([np.var(column[max(0, i - win):i + win])
                              for i in range(len(column))])

        # Eyelid region: low variance (smooth skin). Iris: high variance (texture).
        var_threshold = np.median(local_var) * 0.6 + np.percentile(local_var, 75) * 0.4

        boundary = 0
        for i in range(3, len(column)):
            if local_var[i] > var_threshold:
                boundary = i
                break

        if boundary > 0:
            abs_y = y1 + boundary
            mask[y1:abs_y, x1 + col] = 0

    # Lower eyelid: scan from bottom up per column
    for col in range(roi_w):
        column = blurred[mid_y:, col].astype(np.float64)
        if len(column) < 10:
            continue

        win = 7
        if len(column) < win * 2:
            continue
        local_var = np.array([np.var(column[max(0, i - win):i + win])
                              for i in range(len(column))])

        var_threshold = np.median(local_var) * 0.6 + np.percentile(local_var, 75) * 0.4

        boundary = len(column)
        for i in range(len(column) - 4, -1, -1):
            if local_var[i] > var_threshold:
                boundary = i + 1
                break

        if boundary < len(column):
            abs_y = y1 + mid_y + boundary
            mask[abs_y:y2, x1 + col] = 0

    # Eyelash detection: high-frequency dark regions
    local_mean = cv2.blur(roi.astype(np.float32), (15, 15))
    local_var_2d = cv2.blur((roi.astype(np.float32) - local_mean) ** 2, (15, 15))
    eyelash_mask = ((local_var_2d > 200) & (roi < local_mean - 20)).astype(np.uint8)
    eyelash_mask = cv2.dilate(eyelash_mask, np.ones((3, 3), np.uint8))
    mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2] & (1 - eyelash_mask)
