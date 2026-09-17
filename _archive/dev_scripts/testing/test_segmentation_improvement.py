"""
Test segmentation improvement: compare d' before (old codes) vs after (new codes).

Re-extracts iris codes for 40 subjects using the improved parabolic eyelid
detection, then computes FHD statistics and d' for both old and new codes.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path
from itertools import combinations

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_extraction.segmentation import segment_iris
from iris_extraction.normalization import normalize_iris
from iris_extraction.encoding import encode_iris
from iris_extraction.matching import fractional_hamming_distance

# Paths
CASIA_ROOT = Path(__file__).resolve().parent.parent / "data" / "Iris" / "CASIA-Iris-Interval"
OLD_CODES_DIR = Path(__file__).resolve().parent.parent / "data" / "Iris" / "iris_codes"
EYE = "L"
N_SUBJECTS = 40  # test on 40 subjects


def load_old_codes(subjects):
    """Load pre-existing iris codes."""
    codes = {}  # {subj_id: [(code, mask), ...]}
    for subj_id in subjects:
        codes[subj_id] = []
        for f in sorted(OLD_CODES_DIR.glob(f"S{subj_id}_{EYE}_*.npz")):
            data = np.load(f)
            codes[subj_id].append((data["code"], data["mask"]))
    return codes


def extract_new_codes(subjects):
    """Re-extract iris codes using improved segmentation."""
    codes = {}
    mask_coverages = []
    failures = 0
    total = 0

    for subj_id in subjects:
        codes[subj_id] = []
        eye_dir = CASIA_ROOT / subj_id / EYE
        if not eye_dir.exists():
            continue

        images = sorted([
            f for f in eye_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.bmp', '.png')
        ])

        for img_path in images:
            total += 1
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                failures += 1
                continue

            seg = segment_iris(image)
            if not seg.success:
                failures += 1
                continue

            normalized, norm_mask = normalize_iris(
                image,
                seg.pupil_center, seg.pupil_radius,
                seg.iris_center, seg.iris_radius,
                seg.noise_mask
            )

            iris_code, code_mask = encode_iris(normalized, norm_mask)
            codes[subj_id].append((iris_code, code_mask))

            # Track mask coverage
            coverage = np.mean(code_mask)
            mask_coverages.append(coverage)

    print(f"  Re-extracted: {total - failures}/{total} success")
    if mask_coverages:
        print(f"  Mask coverage: mean={np.mean(mask_coverages):.3f}, "
              f"min={np.min(mask_coverages):.3f}, "
              f"max={np.max(mask_coverages):.3f}, "
              f"std={np.std(mask_coverages):.3f}")
        print(f"  Coverage >= 0.40: {np.mean(np.array(mask_coverages) >= 0.40)*100:.1f}%")

    return codes


def compute_fhd_stats(codes, label):
    """Compute genuine and impostor FHD distributions."""
    subjects = sorted([s for s in codes if len(codes[s]) >= 2])

    genuine_fhds = []
    impostor_fhds = []
    mask_overlaps = []

    # Genuine: same subject, all pairs
    for subj_id in subjects:
        subj_codes = codes[subj_id]
        for i in range(len(subj_codes)):
            for j in range(i + 1, len(subj_codes)):
                c1, m1 = subj_codes[i]
                c2, m2 = subj_codes[j]
                if len(c1) != len(c2):
                    continue
                fhd = fractional_hamming_distance(c1, m1, c2, m2, max_shift=16)
                genuine_fhds.append(fhd)

                # Mask overlap (fraction of bits valid in both)
                n_seg = len(c1) // 512
                m1_2d = m1.reshape(n_seg, 512)
                m2_2d = m2.reshape(n_seg, 512)
                overlap = np.mean(m1_2d & m2_2d)
                mask_overlaps.append(overlap)

    # Impostor: different subjects, first image only (keep manageable)
    subj_first = {s: codes[s][0] for s in subjects}
    subj_list = sorted(subj_first.keys())
    n_imp = 0
    for i in range(len(subj_list)):
        for j in range(i + 1, len(subj_list)):
            c1, m1 = subj_first[subj_list[i]]
            c2, m2 = subj_first[subj_list[j]]
            if len(c1) != len(c2):
                continue
            fhd = fractional_hamming_distance(c1, m1, c2, m2, max_shift=16)
            impostor_fhds.append(fhd)
            n_imp += 1
            if n_imp >= 2000:  # cap at 2000 impostor pairs
                break
        if n_imp >= 2000:
            break

    genuine_fhds = np.array(genuine_fhds)
    impostor_fhds = np.array(impostor_fhds)
    mask_overlaps = np.array(mask_overlaps)

    g_mean = np.mean(genuine_fhds)
    g_std = np.std(genuine_fhds)
    i_mean = np.mean(impostor_fhds)
    i_std = np.std(impostor_fhds)
    dprime = (i_mean - g_mean) / np.sqrt((g_std**2 + i_std**2) / 2)

    print(f"\n=== {label} ===")
    print(f"  Genuine pairs:  {len(genuine_fhds)}")
    print(f"  Impostor pairs: {len(impostor_fhds)}")
    print(f"  Genuine FHD:  mean={g_mean:.4f}, std={g_std:.4f}, "
          f"min={np.min(genuine_fhds):.4f}, max={np.max(genuine_fhds):.4f}")
    print(f"  Impostor FHD: mean={i_mean:.4f}, std={i_std:.4f}, "
          f"min={np.min(impostor_fhds):.4f}, max={np.max(impostor_fhds):.4f}")
    print(f"  d' = {dprime:.3f}")
    print(f"  FHD >= 0.40 in genuine: {np.mean(genuine_fhds >= 0.40)*100:.1f}%")
    if len(mask_overlaps) > 0:
        print(f"  Mask overlap: mean={np.mean(mask_overlaps):.3f}, "
              f"min={np.min(mask_overlaps):.3f}")

    # FRR at various thresholds
    for tau in [0.30, 0.35, 0.40, 0.45]:
        frr = np.mean(genuine_fhds >= tau) * 100
        far = np.mean(impostor_fhds < tau) * 100
        print(f"  tau={tau:.2f}: FAR={far:.2f}%, FRR={frr:.2f}%")

    return {
        "dprime": dprime,
        "genuine_mean": g_mean,
        "genuine_std": g_std,
        "impostor_mean": i_mean,
        "impostor_std": i_std,
        "genuine_fhds": genuine_fhds,
        "impostor_fhds": impostor_fhds,
    }


if __name__ == "__main__":
    # Get subject list
    all_subjects = sorted([
        d.name for d in CASIA_ROOT.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])[:N_SUBJECTS]

    print(f"Testing on {len(all_subjects)} subjects: {all_subjects[0]}–{all_subjects[-1]}")
    print()

    # --- Old codes ---
    print("Loading old iris codes...")
    old_codes = load_old_codes(all_subjects)
    n_old = sum(len(v) for v in old_codes.values())
    print(f"  Loaded {n_old} old codes")
    old_stats = compute_fhd_stats(old_codes, "OLD SEGMENTATION")

    # --- New codes ---
    print("\nRe-extracting with improved segmentation...")
    new_codes = extract_new_codes(all_subjects)
    n_new = sum(len(v) for v in new_codes.values())
    print(f"  Extracted {n_new} new codes")
    new_stats = compute_fhd_stats(new_codes, "NEW SEGMENTATION (Parabolic Eyelid)")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  d' improvement:  {old_stats['dprime']:.3f} → {new_stats['dprime']:.3f} "
          f"(Δ = {new_stats['dprime'] - old_stats['dprime']:+.3f})")
    print(f"  Genuine FHD:     {old_stats['genuine_mean']:.4f} → {new_stats['genuine_mean']:.4f}")
    print(f"  Impostor FHD:    {old_stats['impostor_mean']:.4f} → {new_stats['impostor_mean']:.4f}")
    print(f"  Genuine std:     {old_stats['genuine_std']:.4f} → {new_stats['genuine_std']:.4f}")

    if new_stats['dprime'] > old_stats['dprime']:
        print("\n  ✓ IMPROVEMENT: d' increased!")
        if new_stats['dprime'] >= 3.0:
            print("  ✓ d' >= 3.0: GOOD — proceed with full re-extraction")
        elif new_stats['dprime'] >= 2.5:
            print("  ~ d' in [2.5, 3.0): MODERATE — some improvement expected")
        else:
            print("  ✗ d' < 2.5: MARGINAL — need deeper changes than segmentation")
    else:
        print("\n  ✗ NO IMPROVEMENT: d' did not increase")
        print("  → Bottleneck is NOT eyelid masking — investigate other factors")
