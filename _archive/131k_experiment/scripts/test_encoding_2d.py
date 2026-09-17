#!/usr/bin/env python3
"""
Test the enhanced 2D iris encoding on a few images and compare FHD
distributions with the original 1D encoding.
"""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_extraction.segmentation import segment_iris
from iris_extraction.normalization import normalize_iris
from iris_extraction.encoding import encode_iris, encode_iris_2d, get_code_length, get_code_length_2d
from iris_extraction.matching import fractional_hamming_distance
import cv2

CASIA = Path(__file__).resolve().parent.parent / "data" / "Iris" / "CASIA-Iris-Interval"

def extract_both(img_path):
    """Extract both 1D and 2D iris codes from a single image."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    seg = segment_iris(img)
    if not seg.success:
        return None
    norm, nmask = normalize_iris(img, seg.pupil_center, seg.pupil_radius,
                                  seg.iris_center, seg.iris_radius, seg.noise_mask)
    code_1d, mask_1d = encode_iris(norm, nmask)
    code_2d, mask_2d = encode_iris_2d(norm, nmask)
    return {
        "code_1d": code_1d, "mask_1d": mask_1d,
        "code_2d": code_2d, "mask_2d": mask_2d,
    }


def main():
    print(f"1D code length: {get_code_length()} bits")
    print(f"2D code length: {get_code_length_2d()} bits")
    print()

    # Collect codes for a few subjects
    subjects_dir = sorted(CASIA.glob("*"))[:20]  # First 20 subjects
    all_codes = {}

    t0 = time.time()
    for subj_dir in subjects_dir:
        subj_id = subj_dir.name
        eye_dir = subj_dir / "L"
        if not eye_dir.is_dir():
            continue
        images = sorted(eye_dir.glob("*.jpg"))
        codes = []
        for img_path in images[:5]:  # First 5 impressions
            result = extract_both(img_path)
            if result:
                codes.append(result)
        if len(codes) >= 2:
            all_codes[subj_id] = codes
    elapsed = time.time() - t0
    print(f"Extracted {sum(len(v) for v in all_codes.values())} codes "
          f"from {len(all_codes)} subjects in {elapsed:.1f}s")
    print()

    # Compute genuine and impostor FHD for both 1D and 2D
    genuine_1d, genuine_2d = [], []
    impostor_1d, impostor_2d = [], []

    subjects = list(all_codes.keys())
    for i, subj in enumerate(subjects):
        codes = all_codes[subj]
        # Genuine pairs (within subject)
        for a in range(len(codes)):
            for b in range(a + 1, len(codes)):
                fhd_1d = fractional_hamming_distance(
                    codes[a]["code_1d"], codes[a]["mask_1d"],
                    codes[b]["code_1d"], codes[b]["mask_1d"])
                genuine_1d.append(fhd_1d)

                fhd_2d = fractional_hamming_distance(
                    codes[a]["code_2d"], codes[a]["mask_2d"],
                    codes[b]["code_2d"], codes[b]["mask_2d"])
                genuine_2d.append(fhd_2d)

    # Impostor pairs (between subjects, sample 200)
    rng = np.random.RandomState(42)
    for _ in range(200):
        i, j = rng.choice(len(subjects), 2, replace=False)
        a_idx = rng.randint(len(all_codes[subjects[i]]))
        b_idx = rng.randint(len(all_codes[subjects[j]]))
        ca = all_codes[subjects[i]][a_idx]
        cb = all_codes[subjects[j]][b_idx]

        fhd_1d = fractional_hamming_distance(
            ca["code_1d"], ca["mask_1d"], cb["code_1d"], cb["mask_1d"])
        impostor_1d.append(fhd_1d)

        fhd_2d = fractional_hamming_distance(
            ca["code_2d"], ca["mask_2d"], cb["code_2d"], cb["mask_2d"])
        impostor_2d.append(fhd_2d)

    # Report
    def stats(name, gen, imp):
        gen = np.array(gen)
        imp = np.array(imp)
        d_prime = abs(np.mean(imp) - np.mean(gen)) / np.sqrt(
            0.5 * (np.std(gen)**2 + np.std(imp)**2))
        mask_cov_msg = ""
        print(f"\n=== {name} ===")
        print(f"Genuine:  mean={np.mean(gen):.4f}  std={np.std(gen):.4f}  "
              f"median={np.median(gen):.4f}  n={len(gen)}")
        print(f"Impostor: mean={np.mean(imp):.4f}  std={np.std(imp):.4f}  "
              f"median={np.median(imp):.4f}  n={len(imp)}")
        print(f"d' = {d_prime:.4f}")
        print(f"Genuine FHD > 0.40: {(gen > 0.40).sum()}/{len(gen)} "
              f"({(gen > 0.40).mean()*100:.1f}%)")
        print(f"Impostor FHD < 0.30: {(imp < 0.30).sum()}/{len(imp)} "
              f"({(imp < 0.30).mean()*100:.1f}%)")

    stats("1D Log-Gabor (original)", genuine_1d, impostor_1d)
    stats("2D Log-Gabor (enhanced)", genuine_2d, impostor_2d)

    # Mask coverage comparison
    print("\n=== Mask Coverage ===")
    all_1d_cov = []
    all_2d_cov = []
    for subj in all_codes.values():
        for c in subj:
            all_1d_cov.append(c["mask_1d"].mean())
            all_2d_cov.append(c["mask_2d"].mean())
    print(f"1D mask coverage: {np.mean(all_1d_cov)*100:.1f}% ± {np.std(all_1d_cov)*100:.1f}%")
    print(f"2D mask coverage: {np.mean(all_2d_cov)*100:.1f}% ± {np.std(all_2d_cov)*100:.1f}%")


if __name__ == "__main__":
    main()
