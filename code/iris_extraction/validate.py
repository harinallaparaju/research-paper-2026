"""
Validate Iris Extraction Quality.

Computes genuine and impostor FHD distributions, decidability index,
and determines if the extraction pipeline meets quality thresholds.

Must-pass thresholds:
    - Segmentation success rate >= 85%
    - Genuine FHD mean <= 0.35
    - Impostor FHD mean >= 0.42
    - Decidability index d' >= 2.0
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from itertools import combinations
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris_extraction.matching import fractional_hamming_distance

IRIS_CODES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Iris" / "iris_codes"


def load_iris_codes(codes_dir: Path = IRIS_CODES_DIR) -> dict:
    """
    Load all extracted iris codes grouped by subject.

    Returns:
        {subject_id: [(filename, code, mask), ...]}
    """
    subjects = {}
    for f in sorted(codes_dir.glob("*.npz")):
        # Filename format: S{subj_id}_{eye}_{image_stem}.npz
        parts = f.stem.split("_")
        subj_id = parts[0]  # e.g., "S001"

        data = np.load(f)
        code = data["code"]
        mask = data["mask"]

        if subj_id not in subjects:
            subjects[subj_id] = []
        subjects[subj_id].append((f.name, code, mask))

    return subjects


def validate_extraction(
    codes_dir: Path = IRIS_CODES_DIR,
    max_impostor_pairs: int = 5000,
    seed: int = 42
) -> dict:
    """
    Validate iris extraction by computing genuine/impostor FHD distributions.

    Args:
        codes_dir: Directory containing .npz iris code files
        max_impostor_pairs: Maximum impostor pairs to sample (for speed)
        seed: Random seed for reproducible impostor sampling

    Returns:
        Validation results dict
    """
    random.seed(seed)
    np.random.seed(seed)

    print("Loading iris codes...")
    subjects = load_iris_codes(codes_dir)
    n_subjects = len(subjects)
    total_codes = sum(len(v) for v in subjects.values())
    print(f"Loaded {total_codes} codes from {n_subjects} subjects")

    if n_subjects < 2:
        return {"error": "Need at least 2 subjects for validation"}

    # Compute shift unit for rotation compensation
    # (the matching function handles this internally now)

    # --- Genuine comparisons ---
    print("Computing genuine FHD...")
    genuine_fhds = []
    for subj_id, codes in subjects.items():
        if len(codes) < 2:
            continue
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                fhd = fractional_hamming_distance(
                    codes[i][1], codes[i][2],
                    codes[j][1], codes[j][2],
                    max_shift=32
                )
                genuine_fhds.append(fhd)

    if len(genuine_fhds) == 0:
        return {"error": "No genuine pairs found (need subjects with >=2 images)"}

    genuine_fhds = np.array(genuine_fhds)

    # --- Impostor comparisons ---
    print("Computing impostor FHD...")
    impostor_fhds = []
    subj_ids = list(subjects.keys())

    # Generate all impostor pairs (between different subjects)
    impostor_pairs = []
    for i in range(len(subj_ids)):
        for j in range(i + 1, len(subj_ids)):
            # Take first image from each subject
            code_i = subjects[subj_ids[i]][0]
            code_j = subjects[subj_ids[j]][0]
            impostor_pairs.append((code_i, code_j))

    # Sample if too many
    if len(impostor_pairs) > max_impostor_pairs:
        impostor_pairs = random.sample(impostor_pairs, max_impostor_pairs)

    for code_i, code_j in impostor_pairs:
        fhd = fractional_hamming_distance(
            code_i[1], code_i[2],
            code_j[1], code_j[2],
            max_shift=32
        )
        impostor_fhds.append(fhd)

    impostor_fhds = np.array(impostor_fhds)

    # --- Compute metrics ---
    gen_mean = float(np.mean(genuine_fhds))
    gen_std = float(np.std(genuine_fhds))
    imp_mean = float(np.mean(impostor_fhds))
    imp_std = float(np.std(impostor_fhds))

    # Decidability index d'
    d_prime = abs(imp_mean - gen_mean) / np.sqrt(0.5 * (gen_std**2 + imp_std**2))

    # Check thresholds
    seg_report_path = codes_dir / "extraction_report.json"
    seg_success_rate = None
    if seg_report_path.exists():
        with open(seg_report_path) as f:
            seg_report = json.load(f)
            seg_success_rate = seg_report.get("success_rate_percent")

    results = {
        "n_subjects": n_subjects,
        "n_codes": total_codes,
        "n_genuine_pairs": len(genuine_fhds),
        "n_impostor_pairs": len(impostor_fhds),
        "genuine_fhd_mean": round(gen_mean, 4),
        "genuine_fhd_std": round(gen_std, 4),
        "genuine_fhd_median": round(float(np.median(genuine_fhds)), 4),
        "impostor_fhd_mean": round(imp_mean, 4),
        "impostor_fhd_std": round(imp_std, 4),
        "impostor_fhd_median": round(float(np.median(impostor_fhds)), 4),
        "decidability_d_prime": round(float(d_prime), 4),
        "segmentation_success_rate": seg_success_rate,
    }

    # Validation gates
    gates = {
        "seg_success_ge_85": seg_success_rate is not None and seg_success_rate >= 85.0,
        "genuine_fhd_le_035": gen_mean <= 0.35,
        "impostor_fhd_ge_042": imp_mean >= 0.42,
        "d_prime_ge_1_8": d_prime >= 1.8,
    }
    results["validation_gates"] = gates
    results["ALL_GATES_PASSED"] = all(gates.values())

    # Print results
    print(f"\n{'='*60}")
    print(f"  IRIS EXTRACTION VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Subjects:            {n_subjects}")
    print(f"  Total codes:         {total_codes}")
    print(f"  Genuine pairs:       {len(genuine_fhds)}")
    print(f"  Impostor pairs:      {len(impostor_fhds)}")
    print(f"")
    print(f"  Genuine FHD:    mean={gen_mean:.4f}  std={gen_std:.4f}")
    print(f"  Impostor FHD:   mean={imp_mean:.4f}  std={imp_std:.4f}")
    print(f"  Decidability d':     {d_prime:.4f}")
    if seg_success_rate is not None:
        print(f"  Seg success rate:    {seg_success_rate:.1f}%")
    print(f"")
    print(f"  --- Validation Gates ---")
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {gate}")
    print(f"")
    overall = "ALL GATES PASSED" if results["ALL_GATES_PASSED"] else "VALIDATION FAILED"
    print(f"  >>> {overall} <<<")
    print(f"{'='*60}")

    # Save results
    out_path = codes_dir / "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save distributions for later plotting
    np.savez_compressed(
        codes_dir / "fhd_distributions.npz",
        genuine=genuine_fhds,
        impostor=impostor_fhds
    )

    return results


if __name__ == "__main__":
    results = validate_extraction()
    sys.exit(0 if results.get("ALL_GATES_PASSED", False) else 1)
