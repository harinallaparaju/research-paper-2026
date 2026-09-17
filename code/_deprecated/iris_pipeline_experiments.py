"""
Systematic Iris Pipeline Experiment Harness.

Tests ALL possible improvements individually, then combines the best.
Each experiment measures: genuine FHD, impostor FHD, d', mask coverage.

Usage:
    python3 -u iris_pipeline_experiments.py 2>&1 | tee iris_experiment_log.txt
"""

import sys
import os
import gc
import cv2
import numpy as np
import json
import time
import itertools
import multiprocessing as mp
from functools import partial
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from datetime import datetime

N_WORKERS = min(4, max(1, mp.cpu_count() - 1))  # Cap at 4 for 8GB RAM

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_extraction.segmentation import (
    segment_iris, _find_pupil, _find_iris, _build_noise_mask,
    _fit_parabola_ransac, SegmentationResult
)
from iris_extraction.normalization import normalize_iris, NORM_HEIGHT, NORM_WIDTH
from iris_extraction.encoding import (
    encode_iris, _build_log_gabor_1d, DEFAULT_WAVELENGTHS, DEFAULT_BANDWIDTH, DEFAULT_CODE_ROWS
)
from iris_extraction.matching import fractional_hamming_distance

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

CASIA_ROOT = Path(__file__).resolve().parent.parent / "data" / "Iris" / "CASIA-Iris-Interval"
EYE = "L"
MAX_SHIFT = 16  # always use wider rotation search
N_IMPOSTOR_PAIRS = 5000  # random impostor pairs to sample
SEED = 42

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CachedImage:
    """Pre-segmented image with cached boundaries."""
    path: str
    image: np.ndarray
    px: int
    py: int
    pr: int
    ix: int
    iy: int
    ir: int
    # Base mask without eyelid masking (pupil + iris boundary + reflections only)
    base_mask: np.ndarray


def load_all_images() -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """
    Load all CASIA-Iris-Interval left-eye images.
    Returns: {subject_id: [(image_path, image_array), ...]}
    """
    subjects = {}
    for subj_dir in sorted(CASIA_ROOT.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.isdigit():
            continue
        eye_dir = subj_dir / EYE
        if not eye_dir.exists():
            continue
        images = []
        for img_file in sorted(eye_dir.iterdir()):
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.bmp', '.png', '.tiff'):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append((str(img_file), img))
        if images:
            subjects[subj_dir.name] = images
    return subjects


def presegment_all(subjects: Dict[str, List[Tuple[str, np.ndarray]]]) -> Dict[str, List[CachedImage]]:
    """
    Pre-compute pupil/iris boundaries for all images ONCE.
    Returns cached results with base mask (no eyelid masking applied yet).
    """
    cached = {}
    n_ok = 0
    n_fail = 0
    
    for subj_id, images in subjects.items():
        subj_cached = []
        for path, img in images:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape
            
            pupil = _find_pupil(img)
            if pupil is None:
                n_fail += 1
                continue
            px, py, pr = pupil
            
            iris = _find_iris(img, px, py, pr)
            if iris is None:
                n_fail += 1
                continue
            ix, iy, ir = iris
            
            if ir <= pr + 10 or ir > min(h, w) // 2:
                n_fail += 1
                continue
            
            # Build base mask (everything except eyelids)
            base_mask = np.ones((h, w), dtype=np.uint8)
            iris_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(iris_mask, (ix, iy), ir, 1, -1)
            base_mask = base_mask & iris_mask
            cv2.circle(base_mask, (px, py), pr + 2, 0, -1)
            
            _, reflections = cv2.threshold(img, 210, 1, cv2.THRESH_BINARY)
            reflections = cv2.dilate(reflections, np.ones((7, 7), np.uint8))
            base_mask = base_mask & (1 - reflections)
            
            subj_cached.append(CachedImage(
                path=path, image=img,
                px=px, py=py, pr=pr,
                ix=ix, iy=iy, ir=ir,
                base_mask=base_mask,
            ))
            n_ok += 1
        
        if subj_cached:
            cached[subj_id] = subj_cached
    
    print(f"  Pre-segmented: {n_ok} success, {n_fail} fail, {len(cached)} subjects")
    return cached


# ═══════════════════════════════════════════════════════════════════════
# IMPROVED EYELID MASKING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

def mask_eyelids_v1_original(image, mask, ix, iy, ir):
    """Original: RANSAC parabola + 15% flat fallback."""
    from iris_extraction.segmentation import _mask_eyelids
    _mask_eyelids(image, mask, ix, iy, ir)


def mask_eyelids_v2_intensity(image, mask, ix, iy, ir):
    """
    Intensity-based column-by-column eyelid detection.
    For each column in the iris annulus, find the transition from 
    eyelid (darker, more uniform) to iris texture.
    """
    h, w = image.shape
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
        local_var = np.array([np.var(column[max(0,i-win):i+win]) for i in range(len(column))])
        
        # Eyelid region: low variance (smooth skin). Iris: high variance (texture).
        # Find where variance first exceeds threshold
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
        local_var = np.array([np.var(column[max(0,i-win):i+win]) for i in range(len(column))])
        
        var_threshold = np.median(local_var) * 0.6 + np.percentile(local_var, 75) * 0.4
        
        boundary = len(column)
        for i in range(len(column) - 4, -1, -1):
            if local_var[i] > var_threshold:
                boundary = i + 1
                break
        
        if boundary < len(column):
            abs_y = y1 + mid_y + boundary
            mask[abs_y:y2, x1 + col] = 0
    
    # Eyelash detection (same as original)
    local_mean = cv2.blur(roi.astype(np.float32), (15, 15))
    local_var_2d = cv2.blur((roi.astype(np.float32) - local_mean) ** 2, (15, 15))
    eyelash_mask = ((local_var_2d > 200) & (roi < local_mean - 20)).astype(np.uint8)
    eyelash_mask = cv2.dilate(eyelash_mask, np.ones((3, 3), np.uint8))
    mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2] & (1 - eyelash_mask)


def mask_eyelids_v3_gradient_histogram(image, mask, ix, iy, ir):
    """
    Gradient-based: uses horizontal gradient magnitude to find eyelid boundaries.
    The eyelid-iris boundary creates a strong horizontal edge.
    Uses per-row gradient energy to find the transition.
    """
    h, w = image.shape
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
    
    # Compute horizontal gradient (Sobel in y direction)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Per-row energy of horizontal gradient
    row_energy = np.mean(np.abs(gy), axis=1)
    
    # Upper eyelid: find the peak in gradient energy (eyelid boundary)
    upper_energy = row_energy[:mid_y]
    if len(upper_energy) > 5:
        # Smooth
        from scipy.ndimage import gaussian_filter1d
        upper_smooth = gaussian_filter1d(upper_energy, sigma=2)
        peak_idx = np.argmax(upper_smooth)
        # The eyelid boundary is at the peak — mask everything above it
        # Add a small margin below the peak too
        boundary = min(peak_idx + 3, mid_y)
        mask[y1:y1+boundary, x1:x2] = 0
    
    # Lower eyelid
    lower_energy = row_energy[mid_y:]
    if len(lower_energy) > 5:
        from scipy.ndimage import gaussian_filter1d
        lower_smooth = gaussian_filter1d(lower_energy, sigma=2)
        peak_idx = np.argmax(lower_smooth)
        boundary = max(peak_idx - 3, 0)
        mask[y1+mid_y+boundary:y2, x1:x2] = 0
    
    # Eyelash detection
    local_mean = cv2.blur(roi.astype(np.float32), (15, 15))
    local_var_2d = cv2.blur((roi.astype(np.float32) - local_mean) ** 2, (15, 15))
    eyelash_mask = ((local_var_2d > 200) & (roi < local_mean - 20)).astype(np.uint8)
    eyelash_mask = cv2.dilate(eyelash_mask, np.ones((3, 3), np.uint8))
    mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2] & (1 - eyelash_mask)


def mask_eyelids_v4_ransac_improved(image, mask, ix, iy, ir):
    """
    Improved RANSAC: more iterations, wider search, lower inlier threshold,
    and a smarter fallback (20% instead of 15%, plus column-wise fallback).
    """
    h, w = image.shape
    y1 = max(0, iy - ir)
    y2 = min(h, iy + ir)
    x1 = max(0, ix - ir)
    x2 = min(w, ix + ir)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    
    roi_h, roi_w = roi.shape
    mid_y = roi_h // 2
    
    # Better edge detection
    blurred = cv2.GaussianBlur(roi, (5, 5), 1.5)
    # Use wider Canny thresholds to catch more edges
    edges = cv2.Canny(blurred, 20, 60)
    
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    
    # --- Upper eyelid ---
    ey, ex = np.where(edges[:mid_y, :] > 0)
    upper_fitted = False
    if len(ex) > 0:
        # Relaxed constraint: |gy| > 0.5*|gx| (instead of strict |gy| > |gx|)
        hmask = np.abs(gy[:mid_y][ey, ex]) > 0.5 * np.abs(gx[:mid_y][ey, ex])
        hx, hy = ex[hmask], ey[hmask]
        
        if len(hx) >= 8:  # lower threshold from 10 to 8
            pts = np.column_stack([hx, hy])
            params = _fit_parabola_ransac(pts, n_iter=500, threshold=4.0, min_inliers=8)
            if params is not None:
                a, b, c = params
                for col in range(roi_w):
                    boundary = int(a * col**2 + b * col + c)
                    # Add margin: mask a few pixels below the fitted curve too
                    boundary = max(0, min(boundary + 2, mid_y))
                    abs_y = y1 + boundary
                    if abs_y > y1:
                        mask[y1:abs_y, x1 + col] = 0
                upper_fitted = True
    
    if not upper_fitted:
        # Better fallback: 20% instead of 15%
        cutoff = y1 + int(roi_h * 0.20)
        mask[y1:cutoff, x1:x2] = 0
    
    # --- Lower eyelid ---
    ey_l, ex_l = np.where(edges[mid_y:, :] > 0)
    lower_fitted = False
    if len(ex_l) > 0:
        hmask_l = np.abs(gy[mid_y:][ey_l, ex_l]) > 0.5 * np.abs(gx[mid_y:][ey_l, ex_l])
        hx_l, hy_l = ex_l[hmask_l], ey_l[hmask_l] + mid_y
        
        if len(hx_l) >= 8:
            pts_l = np.column_stack([hx_l, hy_l])
            params_l = _fit_parabola_ransac(pts_l, n_iter=500, threshold=4.0, min_inliers=8)
            if params_l is not None:
                a, b, c = params_l
                for col in range(roi_w):
                    boundary = int(a * col**2 + b * col + c)
                    boundary = max(mid_y, min(boundary - 2, roi_h))
                    abs_y = y1 + boundary
                    if abs_y < y2:
                        mask[abs_y:y2, x1 + col] = 0
                lower_fitted = True
    
    if not lower_fitted:
        cutoff = y2 - int(roi_h * 0.20)
        mask[cutoff:y2, x1:x2] = 0
    
    # Eyelash detection (same)
    local_mean = cv2.blur(roi.astype(np.float32), (15, 15))
    local_var_2d = cv2.blur((roi.astype(np.float32) - local_mean) ** 2, (15, 15))
    eyelash_mask = ((local_var_2d > 200) & (roi < local_mean - 20)).astype(np.uint8)
    eyelash_mask = cv2.dilate(eyelash_mask, np.ones((3, 3), np.uint8))
    mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2] & (1 - eyelash_mask)


def mask_eyelids_v5_union(image, mask, ix, iy, ir):
    """
    Union masking: combine multiple strategies. A pixel is valid only if 
    ALL strategies agree it's valid. This is the most conservative approach.
    """
    h, w = image.shape
    
    # Run original RANSAC
    mask1 = mask.copy()
    mask_eyelids_v1_original(image, mask1, ix, iy, ir)
    
    # Run intensity-based
    mask2 = np.ones((h, w), dtype=np.uint8)
    # Rebuild base mask (outside iris + pupil + reflections already masked in caller)
    mask2[:] = mask[:]  # start from pre-eyelid mask
    mask_eyelids_v2_intensity(image, mask2, ix, iy, ir)
    
    # Union: only valid where BOTH say valid
    mask[:] = mask1 & mask2


def mask_eyelids_v6_aggressive_flat(image, mask, ix, iy, ir):
    """
    Aggressive flat masking: mask top 25% and bottom 25% of iris region.
    Simple but consistent — no RANSAC, no variance, no edges.
    """
    h, w = image.shape
    y1 = max(0, iy - ir)
    y2 = min(h, iy + ir)
    x1 = max(0, ix - ir)
    x2 = min(w, ix + ir)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    
    roi_h = y2 - y1
    
    # Flat 25% top and bottom
    top_cut = y1 + int(roi_h * 0.25)
    bot_cut = y2 - int(roi_h * 0.25)
    mask[y1:top_cut, x1:x2] = 0
    mask[bot_cut:y2, x1:x2] = 0
    
    # Eyelash detection still useful
    local_mean = cv2.blur(roi.astype(np.float32), (15, 15))
    local_var_2d = cv2.blur((roi.astype(np.float32) - local_mean) ** 2, (15, 15))
    eyelash_mask = ((local_var_2d > 200) & (roi < local_mean - 20)).astype(np.uint8)
    eyelash_mask = cv2.dilate(eyelash_mask, np.ones((3, 3), np.uint8))
    mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2] & (1 - eyelash_mask)


def mask_eyelids_v7_none(image, mask, ix, iy, ir):
    """No eyelid masking at all — control experiment."""
    pass


def mask_eyelids_v8_adaptive_threshold(image, mask, ix, iy, ir):
    """
    Adaptive threshold on normalized strip intensity.
    After normalization, the eyelid regions in the polar image have distinctive
    intensity patterns. But since we can't normalize here (no encoding yet),
    we use intensity gradient profile in the radial direction.
    
    For each angular column in the iris annulus, find the radial boundary 
    where iris texture transitions to eyelid.
    """
    h, w = image.shape
    y1 = max(0, iy - ir)
    y2 = min(h, iy + ir)
    x1 = max(0, ix - ir)
    x2 = min(w, ix + ir)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    
    roi_h, roi_w = roi.shape
    mid_y = roi_h // 2
    
    blurred = cv2.GaussianBlur(roi, (7, 7), 2.0)
    
    # Use Otsu's method on the upper and lower halves to separate eyelid from iris
    upper_half = blurred[:mid_y, :]
    lower_half = blurred[mid_y:, :]
    
    if upper_half.size > 100:
        # In NIR, eyelids are often slightly different intensity from iris
        # Use adaptive threshold
        upper_adaptive = cv2.adaptiveThreshold(
            upper_half, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, -5
        )
        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        upper_opened = cv2.morphologyEx(upper_adaptive, cv2.MORPH_OPEN, kernel)
        
        # Find the lowest "eyelid" row per column
        for col in range(roi_w):
            col_data = upper_opened[:, col]
            # Scan from top, find last row that looks like eyelid (bright in adaptive)
            last_eyelid_row = 0
            for row in range(len(col_data)):
                if col_data[row] > 0:
                    last_eyelid_row = row
            if last_eyelid_row > 2:
                mask[y1:y1+last_eyelid_row+1, x1+col] = 0
    
    if lower_half.size > 100:
        lower_adaptive = cv2.adaptiveThreshold(
            lower_half, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, -5
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        lower_opened = cv2.morphologyEx(lower_adaptive, cv2.MORPH_OPEN, kernel)
        
        for col in range(roi_w):
            col_data = lower_opened[:, col]
            first_eyelid_row = len(col_data)
            for row in range(len(col_data) - 1, -1, -1):
                if col_data[row] > 0:
                    first_eyelid_row = row
            if first_eyelid_row < len(col_data) - 2:
                mask[y1+mid_y+first_eyelid_row:y2, x1+col] = 0
    
    # Eyelash detection
    local_mean = cv2.blur(roi.astype(np.float32), (15, 15))
    local_var_2d = cv2.blur((roi.astype(np.float32) - local_mean) ** 2, (15, 15))
    eyelash_mask = ((local_var_2d > 200) & (roi < local_mean - 20)).astype(np.uint8)
    eyelash_mask = cv2.dilate(eyelash_mask, np.ones((3, 3), np.uint8))
    mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2] & (1 - eyelash_mask)


# ═══════════════════════════════════════════════════════════════════════
# EYELID FUNCTION REGISTRY (for multiprocessing pickle support)
# ═══════════════════════════════════════════════════════════════════════

EYELID_FN_REGISTRY = {
    'v1_original': mask_eyelids_v1_original,
    'v2_intensity': mask_eyelids_v2_intensity,
    'v3_gradient': mask_eyelids_v3_gradient_histogram,
    'v4_ransac_improved': mask_eyelids_v4_ransac_improved,
    'v5_union': mask_eyelids_v5_union,
    'v6_flat25': mask_eyelids_v6_aggressive_flat,
    'v7_none': mask_eyelids_v7_none,
    'v8_adaptive_thresh': mask_eyelids_v8_adaptive_threshold,
}

# Reverse lookup: function → name
EYELID_FN_NAMES = {v: k for k, v in EYELID_FN_REGISTRY.items()}


# ═══════════════════════════════════════════════════════════════════════
# CUSTOM ENCODING (parameterized)
# ═══════════════════════════════════════════════════════════════════════

def encode_iris_custom(
    normalized: np.ndarray,
    norm_mask: np.ndarray,
    wavelengths: list,
    bandwidth: float,
    code_rows: Tuple[int, int],
    clahe_clip: float = 2.0,
    blur_sigma: float = 1.0,
    frag_threshold_factor: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom iris encoding with tunable parameters.
    """
    h, w = normalized.shape
    row_start, row_end = code_rows
    row_start = max(0, row_start)
    row_end = min(h, row_end)
    selected_rows = range(row_start, row_end)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 16))
    enhanced = clahe.apply(normalized.astype(np.uint8))
    
    if blur_sigma > 0:
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=blur_sigma)

    code_bits = []
    mask_bits = []

    for wavelength in wavelengths:
        log_gabor = _build_log_gabor_1d(w, wavelength, bandwidth)

        for row_idx in selected_rows:
            row = enhanced[row_idx, :].astype(np.float64)
            row_mask = norm_mask[row_idx, :]

            row_fft = np.fft.fft(row)
            filtered = np.fft.ifft(row_fft * log_gabor)

            re_part = np.real(filtered)
            im_part = np.imag(filtered)

            bit_re = (re_part > 0).astype(np.uint8)
            bit_im = (im_part > 0).astype(np.uint8)

            amplitude = np.sqrt(re_part**2 + im_part**2)
            amp_med = np.median(amplitude[amplitude > 0]) if np.any(amplitude > 0) else 1.0
            frag_thr = frag_threshold_factor * amp_med
            re_ok = (np.abs(re_part) > frag_thr).astype(np.uint8)
            im_ok = (np.abs(im_part) > frag_thr).astype(np.uint8)

            code_bits.append(bit_re)
            code_bits.append(bit_im)
            mask_bits.append(row_mask & re_ok)
            mask_bits.append(row_mask & im_ok)

    iris_code = np.concatenate(code_bits)
    code_mask = np.concatenate(mask_bits)

    return iris_code, code_mask


# ═══════════════════════════════════════════════════════════════════════
# SEGMENTATION WITH PLUGGABLE EYELID MASKING
# ═══════════════════════════════════════════════════════════════════════

def segment_with_custom_mask(image: np.ndarray, eyelid_fn) -> Optional[SegmentationResult]:
    """
    Run segmentation (pupil + iris detection is always the same)
    but use a custom eyelid masking function.
    """
    if image is None or image.size == 0:
        return None
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = image.shape
    
    pupil = _find_pupil(image)
    if pupil is None:
        return None
    px, py, pr = pupil
    
    iris = _find_iris(image, px, py, pr)
    if iris is None:
        return None
    ix, iy, ir = iris
    
    if ir <= pr + 10:
        return None
    if ir > min(h, w) // 2:
        return None
    
    # Build base mask (pupil + iris boundary + reflections)
    mask = np.ones((h, w), dtype=np.uint8)
    iris_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_mask, (ix, iy), ir, 1, -1)
    mask = mask & iris_mask
    cv2.circle(mask, (px, py), pr + 2, 0, -1)
    
    _, reflections = cv2.threshold(image, 210, 1, cv2.THRESH_BINARY)
    reflections = cv2.dilate(reflections, np.ones((7, 7), np.uint8))
    mask = mask & (1 - reflections)
    
    # Apply custom eyelid masking
    eyelid_fn(image, mask, ix, iy, ir)
    
    return SegmentationResult(
        pupil_center=(px, py),
        pupil_radius=pr,
        iris_center=(ix, iy),
        iris_radius=ir,
        noise_mask=mask,
        success=True
    )


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE: image → (code, mask) with all parameters
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Configuration for a single pipeline variant. Fully pickle-safe."""
    name: str
    eyelid_fn_name: str  # key into EYELID_FN_REGISTRY
    wavelengths: list = field(default_factory=lambda: [18, 36])
    bandwidth: float = 0.5
    code_rows: Tuple[int, int] = (20, 44)  # 24 rows (current)
    clahe_clip: float = 2.0
    blur_sigma: float = 1.0
    frag_threshold_factor: float = 0.25

    @property
    def eyelid_fn(self):
        return EYELID_FN_REGISTRY[self.eyelid_fn_name]


def run_pipeline_cached(cached_img: CachedImage, config: PipelineConfig) -> Optional[dict]:
    """
    Run encoding pipeline on a pre-segmented image.
    Only computes: eyelid masking → normalization → encoding.
    """
    image = cached_img.image
    h, w = image.shape
    
    # Apply eyelid masking to a COPY of the base mask
    mask = cached_img.base_mask.copy()
    config.eyelid_fn(image, mask, cached_img.ix, cached_img.iy, cached_img.ir)
    
    # Normalize
    normalized, norm_mask = normalize_iris(
        image,
        (cached_img.px, cached_img.py), cached_img.pr,
        (cached_img.ix, cached_img.iy), cached_img.ir,
        mask
    )
    
    # Encode
    code, code_mask = encode_iris_custom(
        normalized, norm_mask,
        wavelengths=config.wavelengths,
        bandwidth=config.bandwidth,
        code_rows=config.code_rows,
        clahe_clip=config.clahe_clip,
        blur_sigma=config.blur_sigma,
        frag_threshold_factor=config.frag_threshold_factor,
    )
    
    mask_frac = code_mask.sum() / len(code_mask) if len(code_mask) > 0 else 0.0
    
    return {
        'code': code,
        'mask': code_mask,
        'mask_fraction': mask_frac,
    }


# ═══════════════════════════════════════════════════════════════════════
# PARALLEL WORKER FUNCTIONS (must be top-level for pickle)
# ═══════════════════════════════════════════════════════════════════════

def _worker_encode(args):
    """Worker: encode a single pre-segmented image with given config."""
    cached_img, config = args
    result = run_pipeline_cached(cached_img, config)
    if result is not None:
        return (result['code'], result['mask'], result['mask_fraction'])
    return None


def _worker_fhd(args):
    """Worker: compute FHD for a single pair."""
    c1, m1, c2, m2, max_shift = args
    return fractional_hamming_distance(c1, m1, c2, m2, max_shift=max_shift)


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION: compute d' for a given config (with multiprocessing)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_config(
    cached_subjects: Dict[str, List[CachedImage]],
    config: PipelineConfig,
    n_impostor_pairs: int = N_IMPOSTOR_PAIRS,
    max_shift: int = MAX_SHIFT,
    verbose: bool = True,
    use_mp: bool = True,
) -> dict:
    """
    Evaluate a pipeline configuration across all pre-segmented subjects.
    Uses multiprocessing for encoding and FHD computation.
    """
    # ── Stage 1: Encode all images (parallel) ──
    all_tasks = []  # [(cached_img, config), ...]
    task_index = []  # [(subj_id, img_idx), ...]
    for subj_id, cached_images in cached_subjects.items():
        for img_idx, cached_img in enumerate(cached_images):
            all_tasks.append((cached_img, config))
            task_index.append((subj_id, img_idx))

    if use_mp and len(all_tasks) > 50:
        with mp.Pool(N_WORKERS) as pool:
            results_list = pool.map(_worker_encode, all_tasks, chunksize=16)
    else:
        results_list = [_worker_encode(t) for t in all_tasks]

    # Organize results by subject
    codes = {}
    n_success = 0
    n_fail = 0
    mask_coverages = []
    for (subj_id, _), result in zip(task_index, results_list):
        if result is not None:
            code, mask, mask_frac = result
            codes.setdefault(subj_id, []).append((code, mask))
            mask_coverages.append(mask_frac)
            n_success += 1
        else:
            n_fail += 1

    if verbose:
        print(f"    Extracted: {n_success} success, {n_fail} fail, {len(codes)} subjects")

    # ── Stage 2: Compute genuine FHD pairs (parallel) ──
    genuine_pairs = []
    for subj_id, subj_codes in codes.items():
        for i in range(len(subj_codes)):
            for j in range(i + 1, len(subj_codes)):
                c1, m1 = subj_codes[i]
                c2, m2 = subj_codes[j]
                if len(c1) == len(c2):
                    genuine_pairs.append((c1, m1, c2, m2, max_shift))

    # ── Stage 3: Build impostor FHD pairs ──
    rng = np.random.RandomState(SEED)
    subj_list = list(codes.keys())
    impostor_pairs = []
    attempts = 0
    while len(impostor_pairs) < n_impostor_pairs and attempts < n_impostor_pairs * 3:
        attempts += 1
        s1, s2 = rng.choice(len(subj_list), 2, replace=False)
        s1_id, s2_id = subj_list[s1], subj_list[s2]
        i1 = rng.randint(len(codes[s1_id]))
        i2 = rng.randint(len(codes[s2_id]))
        c1, m1 = codes[s1_id][i1]
        c2, m2 = codes[s2_id][i2]
        if len(c1) == len(c2):
            impostor_pairs.append((c1, m1, c2, m2, max_shift))

    # ── Stage 4: Compute all FHDs (sequential — avoids OOM from pickling large code arrays) ──
    all_fhd_pairs = genuine_pairs + impostor_pairs
    all_fhds = [_worker_fhd(p) for p in all_fhd_pairs]

    genuine_fhds = all_fhds[:len(genuine_pairs)]
    impostor_fhds = all_fhds[len(genuine_pairs):]
    
    if not genuine_fhds or not impostor_fhds:
        return {
            'd_prime': 0.0, 'genuine_mean': 1.0, 'genuine_std': 0.0,
            'impostor_mean': 0.5, 'impostor_std': 0.0,
            'mask_coverage_mean': 0.0, 'n_genuine': 0, 'n_impostor': 0,
            'n_success': n_success, 'n_fail': n_fail, 'n_subjects': len(codes),
        }
    
    gen_mean = np.mean(genuine_fhds)
    gen_std = np.std(genuine_fhds)
    imp_mean = np.mean(impostor_fhds)
    imp_std = np.std(impostor_fhds)
    
    d_prime = (imp_mean - gen_mean) / np.sqrt(0.5 * (gen_std**2 + imp_std**2)) if (gen_std + imp_std) > 0 else 0.0
    
    return {
        'd_prime': d_prime,
        'genuine_mean': gen_mean,
        'genuine_std': gen_std,
        'impostor_mean': imp_mean,
        'impostor_std': imp_std,
        'mask_coverage_mean': np.mean(mask_coverages),
        'n_genuine': len(genuine_fhds),
        'n_impostor': len(impostor_fhds),
        'n_success': n_success,
        'n_fail': n_fail,
        'n_subjects': len(codes),
    }


def print_result(name: str, r: dict, elapsed: float):
    """Pretty-print evaluation results."""
    print(f"  {name:>35s} | d'={r['d_prime']:5.2f} | "
          f"gen={r['genuine_mean']:.4f}±{r['genuine_std']:.4f} | "
          f"imp={r['impostor_mean']:.4f}±{r['impostor_std']:.4f} | "
          f"mask={r['mask_coverage_mean']:.3f} | "
          f"{r['n_subjects']}subj {r['n_success']}ok {r['n_fail']}fail | "
          f"{elapsed:.0f}s")


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

def get_masking_experiments() -> List[PipelineConfig]:
    """Experiment 1: Compare eyelid masking strategies (encoding fixed)."""
    base_wl = [18, 36]
    base_rows = (20, 44)
    
    return [
        PipelineConfig("mask_v1_original", 'v1_original', base_wl, code_rows=base_rows),
        PipelineConfig("mask_v2_intensity", 'v2_intensity', base_wl, code_rows=base_rows),
        PipelineConfig("mask_v3_gradient", 'v3_gradient', base_wl, code_rows=base_rows),
        PipelineConfig("mask_v4_ransac_improved", 'v4_ransac_improved', base_wl, code_rows=base_rows),
        PipelineConfig("mask_v5_union", 'v5_union', base_wl, code_rows=base_rows),
        PipelineConfig("mask_v6_flat25", 'v6_flat25', base_wl, code_rows=base_rows),
        PipelineConfig("mask_v7_none", 'v7_none', base_wl, code_rows=base_rows),
        PipelineConfig("mask_v8_adaptive_thresh", 'v8_adaptive_thresh', base_wl, code_rows=base_rows),
    ]


def get_wavelength_experiments(best_mask_fn_name: str) -> List[PipelineConfig]:
    """Experiment 2: Compare wavelength configurations (best masking fixed)."""
    base_rows = (20, 44)
    
    return [
        PipelineConfig("wl_2_[18,36]", best_mask_fn_name, [18, 36], code_rows=base_rows),
        PipelineConfig("wl_3_[12,18,36]", best_mask_fn_name, [12, 18, 36], code_rows=base_rows),
        PipelineConfig("wl_3_[9,18,36]", best_mask_fn_name, [9, 18, 36], code_rows=base_rows),
        PipelineConfig("wl_4_[9,12,18,36]", best_mask_fn_name, [9, 12, 18, 36], code_rows=base_rows),
        PipelineConfig("wl_5_[9,12,18,24,36]", best_mask_fn_name, [9, 12, 18, 24, 36], code_rows=base_rows),
        PipelineConfig("wl_4_[12,18,24,36]", best_mask_fn_name, [12, 18, 24, 36], code_rows=base_rows),
        PipelineConfig("wl_3_[18,24,36]", best_mask_fn_name, [18, 24, 36], code_rows=base_rows),
        PipelineConfig("wl_6_[8,12,16,20,28,36]", best_mask_fn_name, [8, 12, 16, 20, 28, 36], code_rows=base_rows),
    ]


def get_row_band_experiments(best_mask_fn_name: str, best_wl) -> List[PipelineConfig]:
    """Experiment 3: Compare row band widths (best masking + wavelengths)."""
    return [
        PipelineConfig("rows_20-44_(24r)", best_mask_fn_name, best_wl, code_rows=(20, 44)),
        PipelineConfig("rows_16-48_(32r)", best_mask_fn_name, best_wl, code_rows=(16, 48)),
        PipelineConfig("rows_12-52_(40r)", best_mask_fn_name, best_wl, code_rows=(12, 52)),
        PipelineConfig("rows_8-56_(48r)", best_mask_fn_name, best_wl, code_rows=(8, 56)),
        PipelineConfig("rows_4-60_(56r)", best_mask_fn_name, best_wl, code_rows=(4, 60)),
        PipelineConfig("rows_0-64_(64r)", best_mask_fn_name, best_wl, code_rows=(0, 64)),
        PipelineConfig("rows_24-40_(16r)", best_mask_fn_name, best_wl, code_rows=(24, 40)),
        PipelineConfig("rows_16-52_(36r)", best_mask_fn_name, best_wl, code_rows=(16, 52)),
    ]


def get_preprocessing_experiments(best_mask_fn_name: str, best_wl, best_rows) -> List[PipelineConfig]:
    """Experiment 4: Compare pre-processing parameters."""
    return [
        PipelineConfig("clahe_2.0_blur_1.0_frag_0.25", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=2.0, blur_sigma=1.0, frag_threshold_factor=0.25),
        PipelineConfig("clahe_1.5_blur_1.0_frag_0.25", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=1.5, blur_sigma=1.0, frag_threshold_factor=0.25),
        PipelineConfig("clahe_3.0_blur_1.0_frag_0.25", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=3.0, blur_sigma=1.0, frag_threshold_factor=0.25),
        PipelineConfig("clahe_2.0_blur_0.5_frag_0.25", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=2.0, blur_sigma=0.5, frag_threshold_factor=0.25),
        PipelineConfig("clahe_2.0_blur_0.0_frag_0.25", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=2.0, blur_sigma=0.0, frag_threshold_factor=0.25),
        PipelineConfig("clahe_2.0_blur_1.5_frag_0.25", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=2.0, blur_sigma=1.5, frag_threshold_factor=0.25),
        PipelineConfig("clahe_2.0_blur_1.0_frag_0.15", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=2.0, blur_sigma=1.0, frag_threshold_factor=0.15),
        PipelineConfig("clahe_2.0_blur_1.0_frag_0.35", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=2.0, blur_sigma=1.0, frag_threshold_factor=0.35),
        PipelineConfig("clahe_2.0_blur_1.0_frag_0.10", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=2.0, blur_sigma=1.0, frag_threshold_factor=0.10),
        PipelineConfig("clahe_2.0_blur_1.0_frag_0.00", best_mask_fn_name, best_wl, code_rows=best_rows,
                       clahe_clip=2.0, blur_sigma=1.0, frag_threshold_factor=0.00),
    ]


def get_bandwidth_experiments(best_mask_fn_name: str, best_wl, best_rows, best_clahe, best_blur, best_frag) -> List[PipelineConfig]:
    """Experiment 5: Compare Log-Gabor bandwidth parameter."""
    return [
        PipelineConfig("bw_0.35", best_mask_fn_name, best_wl, bandwidth=0.35, code_rows=best_rows,
                       clahe_clip=best_clahe, blur_sigma=best_blur, frag_threshold_factor=best_frag),
        PipelineConfig("bw_0.40", best_mask_fn_name, best_wl, bandwidth=0.40, code_rows=best_rows,
                       clahe_clip=best_clahe, blur_sigma=best_blur, frag_threshold_factor=best_frag),
        PipelineConfig("bw_0.45", best_mask_fn_name, best_wl, bandwidth=0.45, code_rows=best_rows,
                       clahe_clip=best_clahe, blur_sigma=best_blur, frag_threshold_factor=best_frag),
        PipelineConfig("bw_0.50", best_mask_fn_name, best_wl, bandwidth=0.50, code_rows=best_rows,
                       clahe_clip=best_clahe, blur_sigma=best_blur, frag_threshold_factor=best_frag),
        PipelineConfig("bw_0.55", best_mask_fn_name, best_wl, bandwidth=0.55, code_rows=best_rows,
                       clahe_clip=best_clahe, blur_sigma=best_blur, frag_threshold_factor=best_frag),
        PipelineConfig("bw_0.60", best_mask_fn_name, best_wl, bandwidth=0.60, code_rows=best_rows,
                       clahe_clip=best_clahe, blur_sigma=best_blur, frag_threshold_factor=best_frag),
        PipelineConfig("bw_0.65", best_mask_fn_name, best_wl, bandwidth=0.65, code_rows=best_rows,
                       clahe_clip=best_clahe, blur_sigma=best_blur, frag_threshold_factor=best_frag),
    ]


# ═══════════════════════════════════════════════════════════════════════
# MAIN: Run all experiments sequentially
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print(f"SYSTEMATIC IRIS PIPELINE EXPERIMENT — {datetime.now().isoformat()}")
    print("=" * 100)
    
    t_start = time.time()
    
    # Load all images and pre-segment ONCE
    print("\n[1/7] Loading CASIA-Iris-Interval images and pre-segmenting...")
    raw_subjects = load_all_images()
    n_images = sum(len(v) for v in raw_subjects.values())
    print(f"  Loaded {len(raw_subjects)} subjects, {n_images} images")
    
    t_seg = time.time()
    cached_subjects = presegment_all(raw_subjects)
    print(f"  Pre-segmentation took {time.time()-t_seg:.0f}s")
    del raw_subjects  # free memory
    
    results_all = {}
    
    # ─── Experiment 1: Eyelid Masking ─────────────────────────────────
    print("\n" + "=" * 100)
    print("[2/7] EXPERIMENT 1: EYELID MASKING STRATEGIES")
    print("  (encoding fixed: wavelengths=[18,36], rows=20-44, bw=0.5)")
    print("=" * 100)
    
    masking_configs = get_masking_experiments()
    masking_results = {}
    
    for cfg in masking_configs:
        t0 = time.time()
        r = evaluate_config(cached_subjects, cfg, verbose=True)
        elapsed = time.time() - t0
        masking_results[cfg.name] = r
        print_result(cfg.name, r, elapsed)
    
    # Find best masking
    best_mask_name = max(masking_results, key=lambda k: masking_results[k]['d_prime'])
    best_mask_dprime = masking_results[best_mask_name]['d_prime']
    print(f"\n  >>> BEST MASKING: {best_mask_name} (d'={best_mask_dprime:.3f})")
    
    # Map config name back to eyelid_fn_name string
    mask_fn_name_map = {cfg.name: cfg.eyelid_fn_name for cfg in masking_configs}
    best_mask_fn_name = mask_fn_name_map[best_mask_name]
    results_all['masking'] = masking_results
    gc.collect()
    
    # ─── Experiment 2: Wavelengths ────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"[3/7] EXPERIMENT 2: WAVELENGTH CONFIGURATIONS (masking={best_mask_name})")
    print("=" * 100)
    
    wl_configs = get_wavelength_experiments(best_mask_fn_name)
    wl_results = {}
    
    for cfg in wl_configs:
        t0 = time.time()
        r = evaluate_config(cached_subjects, cfg, verbose=True)
        elapsed = time.time() - t0
        wl_results[cfg.name] = r
        print_result(cfg.name, r, elapsed)
    
    best_wl_name = max(wl_results, key=lambda k: wl_results[k]['d_prime'])
    best_wl_dprime = wl_results[best_wl_name]['d_prime']
    # Extract wavelength list from the config
    best_wl = [cfg.wavelengths for cfg in wl_configs if cfg.name == best_wl_name][0]
    print(f"\n  >>> BEST WAVELENGTHS: {best_wl_name} = {best_wl} (d'={best_wl_dprime:.3f})")
    results_all['wavelengths'] = wl_results
    gc.collect()
    
    # ─── Experiment 3: Row Band ───────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"[4/7] EXPERIMENT 3: ROW BAND WIDTH (masking={best_mask_name}, wl={best_wl})")
    print("=" * 100)
    
    row_configs = get_row_band_experiments(best_mask_fn_name, best_wl)
    row_results = {}
    
    for cfg in row_configs:
        t0 = time.time()
        r = evaluate_config(cached_subjects, cfg, verbose=True)
        elapsed = time.time() - t0
        row_results[cfg.name] = r
        print_result(cfg.name, r, elapsed)
    
    best_row_name = max(row_results, key=lambda k: row_results[k]['d_prime'])
    best_row_dprime = row_results[best_row_name]['d_prime']
    best_rows = [cfg.code_rows for cfg in row_configs if cfg.name == best_row_name][0]
    print(f"\n  >>> BEST ROWS: {best_row_name} = {best_rows} (d'={best_row_dprime:.3f})")
    results_all['row_band'] = row_results
    gc.collect()
    
    # ─── Experiment 4: Pre-processing ─────────────────────────────────
    print("\n" + "=" * 100)
    print(f"[5/7] EXPERIMENT 4: PRE-PROCESSING PARAMS (mask={best_mask_name}, wl={best_wl}, rows={best_rows})")
    print("=" * 100)
    
    pp_configs = get_preprocessing_experiments(best_mask_fn_name, best_wl, best_rows)
    pp_results = {}
    
    for cfg in pp_configs:
        t0 = time.time()
        r = evaluate_config(cached_subjects, cfg, verbose=True)
        elapsed = time.time() - t0
        pp_results[cfg.name] = r
        print_result(cfg.name, r, elapsed)
    
    best_pp_name = max(pp_results, key=lambda k: pp_results[k]['d_prime'])
    best_pp = [cfg for cfg in pp_configs if cfg.name == best_pp_name][0]
    print(f"\n  >>> BEST PRE-PROCESSING: {best_pp_name} (d'={pp_results[best_pp_name]['d_prime']:.3f})")
    results_all['preprocessing'] = pp_results
    gc.collect()
    
    # ─── Experiment 5: Bandwidth ──────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"[6/7] EXPERIMENT 5: LOG-GABOR BANDWIDTH")
    print("=" * 100)
    
    bw_configs = get_bandwidth_experiments(
        best_mask_fn_name, best_wl, best_rows,
        best_pp.clahe_clip, best_pp.blur_sigma, best_pp.frag_threshold_factor
    )
    bw_results = {}
    
    for cfg in bw_configs:
        t0 = time.time()
        r = evaluate_config(cached_subjects, cfg, verbose=True)
        elapsed = time.time() - t0
        bw_results[cfg.name] = r
        print_result(cfg.name, r, elapsed)
    
    best_bw_name = max(bw_results, key=lambda k: bw_results[k]['d_prime'])
    best_bw = [cfg.bandwidth for cfg in bw_configs if cfg.name == best_bw_name][0]
    print(f"\n  >>> BEST BANDWIDTH: {best_bw_name} = {best_bw} (d'={bw_results[best_bw_name]['d_prime']:.3f})")
    results_all['bandwidth'] = bw_results
    
    # ─── Final: Best Combined Config ──────────────────────────────────
    print("\n" + "=" * 100)
    print("[7/7] FINAL: BEST COMBINED CONFIGURATION")
    print("=" * 100)
    
    final_config = PipelineConfig(
        name="BEST_COMBINED",
        eyelid_fn_name=best_mask_fn_name,
        wavelengths=best_wl,
        bandwidth=best_bw,
        code_rows=best_rows,
        clahe_clip=best_pp.clahe_clip,
        blur_sigma=best_pp.blur_sigma,
        frag_threshold_factor=best_pp.frag_threshold_factor,
    )
    
    print(f"  Masking:     {best_mask_name}")
    print(f"  Wavelengths: {best_wl}")
    print(f"  Bandwidth:   {best_bw}")
    print(f"  Row band:    {best_rows}")
    print(f"  CLAHE clip:  {best_pp.clahe_clip}")
    print(f"  Blur sigma:  {best_pp.blur_sigma}")
    print(f"  Frag thresh: {best_pp.frag_threshold_factor}")
    
    t0 = time.time()
    final_result = evaluate_config(cached_subjects, final_config, verbose=True)
    elapsed = time.time() - t0
    print_result("BEST_COMBINED", final_result, elapsed)
    results_all['final'] = final_result
    
    # ─── Baseline comparison ──────────────────────────────────────────
    baseline_config = PipelineConfig(
        name="BASELINE",
        eyelid_fn_name='v1_original',
        wavelengths=[18, 36],
        bandwidth=0.5,
        code_rows=(20, 44),
    )
    t0 = time.time()
    baseline_result = evaluate_config(cached_subjects, baseline_config, verbose=True)
    elapsed = time.time() - t0
    print_result("BASELINE", baseline_result, elapsed)
    results_all['baseline'] = baseline_result
    
    # ─── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Baseline d':            {baseline_result['d_prime']:.3f}")
    print(f"  Best combined d':       {final_result['d_prime']:.3f}")
    improvement = final_result['d_prime'] - baseline_result['d_prime']
    pct = (improvement / baseline_result['d_prime'] * 100) if baseline_result['d_prime'] > 0 else 0
    print(f"  Improvement:            +{improvement:.3f} ({pct:.1f}%)")
    print(f"  Genuine FHD:            {baseline_result['genuine_mean']:.4f} → {final_result['genuine_mean']:.4f}")
    print(f"  Impostor FHD:           {baseline_result['impostor_mean']:.4f} → {final_result['impostor_mean']:.4f}")
    print(f"  Mask coverage:          {baseline_result['mask_coverage_mean']:.3f} → {final_result['mask_coverage_mean']:.3f}")
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    
    # Save results
    save_path = Path(__file__).parent / "results" / "iris_pipeline_experiments.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        return obj
    
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'best_config': {
            'masking': best_mask_name,
            'wavelengths': best_wl,
            'bandwidth': best_bw,
            'code_rows': list(best_rows),
            'clahe_clip': best_pp.clahe_clip,
            'blur_sigma': best_pp.blur_sigma,
            'frag_threshold_factor': best_pp.frag_threshold_factor,
        },
        'results': to_native(results_all),
    }
    
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {save_path}")
    
    print("\n" + "=" * 100)
    print(f"ALL DONE in {total_time/60:.1f} minutes")
    print("=" * 100)


if __name__ == "__main__":
    main()
