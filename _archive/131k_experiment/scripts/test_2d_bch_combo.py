#!/usr/bin/env python3
"""
Test 2D encoding + BCH concatenated code stabilizer together.

1. Extract 2D codes for all subjects (or load if cached)
2. Test GAR across block sizes and BCH t values
3. Compare with 1D+BCH and 1D-original baselines

Run from: /Users/suryanallaparaju/Desktop/Surya's Project/code/
"""

import sys, os, time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "iris_extraction"))

from iris_extraction.encoding import encode_iris, encode_iris_2d
from iris_extraction.normalization import normalize_iris
from iris_extraction.segmentation import segment_iris
from iris_stabilizer import (
    IrisStabilizer, IrisStabilizerBCH,
    create_iris_commitment, recover_iris_key,
    create_iris_commitment_bch, recover_iris_key_bch,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "Iris" / "CASIA-Iris-Interval"
CODE_1D_DIR = Path(__file__).parent.parent / "data" / "Iris" / "iris_codes"
CODE_2D_DIR = Path(__file__).parent.parent / "data" / "Iris" / "iris_codes_2d"
MAX_SUBJECTS = 0  # 0 = all
MAX_SHIFT = 8


def extract_2d_codes():
    """Extract 2D iris codes for all images (caches to disk)."""
    CODE_2D_DIR.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_paths = sorted(DATA_DIR.rglob("*.jpg"))
    if not image_paths:
        image_paths = sorted(DATA_DIR.rglob("*.bmp"))
    print(f"Found {len(image_paths)} images")

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for img_path in image_paths:
        # Derive output name: S001/L/S1001L01.jpg → S001_L_S1001L01.npz
        parts = img_path.parts
        # Find the subject folder and eye folder
        subj_dir = img_path.parent.parent.name  # e.g. "001"
        eye_dir = img_path.parent.name           # e.g. "L"
        stem = img_path.stem                      # e.g. "S1001L01"
        out_name = f"S{subj_dir}_{eye_dir}_{stem}.npz"
        out_path = CODE_2D_DIR / out_name

        if out_path.exists():
            n_skip += 1
            continue

        try:
            import cv2
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                n_fail += 1
                continue

            seg = segment_iris(image)
            if seg is None or not seg.success:
                n_fail += 1
                continue

            norm, norm_mask = normalize_iris(
                image,
                seg.pupil_center, seg.pupil_radius,
                seg.iris_center, seg.iris_radius,
                seg.noise_mask,
            )

            code, mask = encode_iris_2d(norm, norm_mask)
            np.savez_compressed(str(out_path), code=code, mask=mask)
            n_ok += 1

        except Exception as e:
            n_fail += 1

    print(f"Extracted: {n_ok} new, {n_skip} cached, {n_fail} failed")


def load_codes(code_dir, max_subjects=0, eye="L"):
    """Load iris codes grouped by subject (single eye only)."""
    subjects = defaultdict(list)
    for f in sorted(code_dir.glob("*.npz")):
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[1] != eye:
            continue
        data = np.load(f)
        subj = parts[0]
        subjects[subj].append((data["code"], data["mask"], f.stem))
    subj_list = sorted(subjects.keys())
    if max_subjects > 0:
        subj_list = subj_list[:max_subjects]
    return {s: subjects[s] for s in subj_list}


def eval_gar_orig(subjects, bs):
    """Evaluate GAR with original stabilizer."""
    n_gen = n_ok = n_enrolled = 0
    for subj, samples in subjects.items():
        if len(samples) < 2:
            continue
        c_e, m_e, _ = samples[0]
        stab = IrisStabilizer(block_size=bs)
        comm = stab.enroll(c_e, m_e, seed=42)
        n_enrolled += 1
        for c_q, m_q, _ in samples[1:]:
            r = stab.recover(c_q, m_q, comm, max_shift=MAX_SHIFT)
            n_gen += 1
            if r.success:
                n_ok += 1
    gar = n_ok / n_gen * 100 if n_gen > 0 else 0
    return gar, n_ok, n_gen, n_enrolled


def eval_gar_bch(subjects, bs, bch_t, min_bv=5):
    """Evaluate GAR with BCH stabilizer."""
    n_gen = n_ok = n_enrolled = 0
    key_bits = 0
    for subj, samples in subjects.items():
        if len(samples) < 2:
            continue
        c_e, m_e, _ = samples[0]
        try:
            stab = IrisStabilizerBCH(block_size=bs, bch_t=bch_t, min_block_valid=min_bv)
            comm = stab.enroll(c_e, m_e, seed=42)
            key_bits = comm.key_length_bits
        except (ValueError, RuntimeError):
            continue
        n_enrolled += 1
        for c_q, m_q, _ in samples[1:]:
            r = stab.recover(c_q, m_q, comm, max_shift=MAX_SHIFT)
            n_gen += 1
            if r.success:
                n_ok += 1
    gar = n_ok / n_gen * 100 if n_gen > 0 else 0
    return gar, n_ok, n_gen, n_enrolled, key_bits


def fhd_stats(subjects, n_impostor=500):
    """Compute genuine/impostor FHD stats."""
    gen_fhds = []
    imp_fhds = []
    subj_list = sorted(subjects.keys())

    # Genuine
    for subj in subj_list:
        samples = subjects[subj]
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                c1, m1, _ = samples[i]
                c2, m2, _ = samples[j]
                combined = m1 & m2
                n_valid = combined.sum()
                if n_valid < 100:
                    continue
                fhd = float(((c1 ^ c2) & combined).sum()) / n_valid
                gen_fhds.append(fhd)

    # Impostor (random pairs)
    rng = np.random.RandomState(42)
    for _ in range(n_impostor):
        s1, s2 = rng.choice(subj_list, 2, replace=False)
        idx1 = rng.randint(len(subjects[s1]))
        idx2 = rng.randint(len(subjects[s2]))
        c1, m1, _ = subjects[s1][idx1]
        c2, m2, _ = subjects[s2][idx2]
        combined = m1 & m2
        n_valid = combined.sum()
        if n_valid < 100:
            continue
        fhd = float(((c1 ^ c2) & combined).sum()) / n_valid
        imp_fhds.append(fhd)

    gen = np.array(gen_fhds)
    imp = np.array(imp_fhds)
    pooled_std = np.sqrt((gen.std()**2 + imp.std()**2) / 2)
    dprime = (imp.mean() - gen.mean()) / pooled_std if pooled_std > 0 else 0
    return gen.mean(), gen.std(), imp.mean(), imp.std(), dprime, gen.mean() * 100


def main():
    print("=" * 70)
    print("2D Encoding + BCH Concatenated Code — Full Pipeline Test")
    print("=" * 70)

    # Step 1: Extract 2D codes
    print("\n--- Step 1: Extract 2D codes ---")
    t0 = time.time()
    extract_2d_codes()
    print(f"  Extraction time: {time.time() - t0:.1f}s")

    # Step 2: Load codes
    print("\n--- Step 2: Load codes ---")
    codes_1d = load_codes(CODE_1D_DIR, MAX_SUBJECTS)
    codes_2d = load_codes(CODE_2D_DIR, MAX_SUBJECTS)
    n1 = sum(len(v) for v in codes_1d.values())
    n2 = sum(len(v) for v in codes_2d.values())
    print(f"  1D: {n1} codes from {len(codes_1d)} subjects")
    print(f"  2D: {n2} codes from {len(codes_2d)} subjects")

    # Code lengths
    if codes_1d:
        c1 = next(iter(codes_1d.values()))[0][0]
        print(f"  1D code length: {len(c1)} bits")
    if codes_2d:
        c2 = next(iter(codes_2d.values()))[0][0]
        print(f"  2D code length: {len(c2)} bits")

    # Step 3: FHD stats
    print("\n--- Step 3: FHD Statistics ---")
    g1, gs1, i1, is1, dp1, gen_pct1 = fhd_stats(codes_1d)
    print(f"  1D: gen_FHD={g1:.4f}±{gs1:.4f}, imp_FHD={i1:.4f}±{is1:.4f}, d'={dp1:.3f}")
    g2, gs2, i2, is2, dp2, gen_pct2 = fhd_stats(codes_2d)
    print(f"  2D: gen_FHD={g2:.4f}±{gs2:.4f}, imp_FHD={i2:.4f}±{is2:.4f}, d'={dp2:.3f}")

    # Step 4: GAR sweep
    print("\n--- Step 4: GAR Results ---")
    print(f"{'Enc':>3s} {'Bs':>5s} {'Type':>12s} | {'GAR':>7s} {'ok/gen':>10s} {'key_bits':>8s}")
    print("-" * 60)

    for label, codes in [("1D", codes_1d), ("2D", codes_2d)]:
        code_len = len(next(iter(codes.values()))[0][0])
        # Use larger block sizes for 2D since it has 3x more bits
        block_sizes = [1023, 2047] if code_len < 200000 else [1023, 2047, 4095]
        for bs in block_sizes:
            # Original
            gar, ok, gen, enrolled = eval_gar_orig(codes, bs)
            n_blocks = len(next(iter(codes.values()))[0][0]) // bs
            print(f"{label:>3s} {bs:5d} {'Original':>12s} | {gar:6.2f}% {ok:4d}/{gen:<4d}  {n_blocks:>5d}")

            # BCH t=3 and t=5
            for t in [3, 5]:
                gar, ok, gen, enrolled, kbits = eval_gar_bch(codes, bs, t)
                print(f"{label:>3s} {bs:5d} {'BCH(t='+str(t)+')':>12s} | {gar:6.2f}% {ok:4d}/{gen:<4d}  {kbits:>5d}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
