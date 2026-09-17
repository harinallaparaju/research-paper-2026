#!/usr/bin/env python3
"""
IBFV Enhanced Evaluation: 2D Encoding + BCH Concatenated Code.

Compares old IBFV (1D + rep-only) vs enhanced IBFV (1D/2D + BCH outer code)
on all 4 FVC databases, showing the impact of the improved iris pipeline.

Configurations tested:
  - Baseline:  1D + Bs=1023 + rep-only  (paper original)
  - Enhanced:  1D + Bs=2047 + BCH(t=5)  (best 1D with BCH)
  - Enhanced:  2D + Bs=4095 + BCH(t=5)  (best overall)
  - Enhanced:  2D + Bs=4095 + BCH(t=3)  (more key bits)
"""

import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file,
    quantize_minutiae,
    create_vault,
    unlock_vault,
)
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv

# Paths
CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"
IRIS_1D_DIR = Path(__file__).resolve().parent.parent / "data" / "Iris" / "iris_codes"
IRIS_2D_DIR = Path(__file__).resolve().parent.parent / "data" / "Iris" / "iris_codes_2d"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ibfv_enhanced"

# Database configs: (db_name, best_factor, best_degree) from senior's paper
DB_CONFIGS = [
    ("fvc2002_1", [(14, 5), (18, 7), (24, 10)]),
    ("fvc2002_2", [(14, 5), (18, 7)]),
    ("fvc2002_3", [(14, 5), (18, 5)]),
    ("fvc2004_1", [(14, 5), (18, 5)]),
]

# Iris configurations to test
# SECURITY NOTE: At Bs≥2047, impostor block error rate ≈10%, allowing BCH to
# "correct" impostor iris codes → FAR explosion. Only Bs=1023 (18% imp block
# error, 128 blocks) is safe with BCH.
IRIS_CONFIGS = [
    # (label,        iris_dir,  block_size, bch_t, n_bonus_list)
    ("1D_Bs1023",    "1d",      1023,       0,     [2, 4, 6]),   # Original paper
    ("1D_BCH3_1023", "1d",      1023,       3,     [2, 4, 6]),   # BCH t=3
    ("1D_BCH5_1023", "1d",      1023,       5,     [2, 4, 6]),   # BCH t=5
    ("2D_Bs1023",    "2d",      1023,       0,     [2, 4, 6]),   # 2D rep-only
    ("2D_BCH3_1023", "2d",      1023,       3,     [2, 4, 6]),   # 2D + BCH t=3
    ("2D_BCH5_1023", "2d",      1023,       5,     [2, 4, 6]),   # 2D + BCH t=5
]


def load_chimeric_data(fp_db: str, iris_dir: str = "1d", min_iris: int = 2) -> List[dict]:
    """Load chimeric mapping with 1D or 2D iris codes."""
    mapping_path = CHIMERIC_DIR / f"chimeric_{fp_db}_min{min_iris}.json"
    with open(mapping_path) as f:
        mapping = json.load(f)

    code_dir = IRIS_1D_DIR if iris_dir == "1d" else IRIS_2D_DIR

    subjects = []
    for entry in mapping["mapping"]:
        subj = {
            "chimeric_id": entry["chimeric_id"],
            "minutiae": {},
            "iris": {},
        }

        for pair in entry["pairs"]:
            imp = pair["impression"] - 1  # 0-indexed
            subj["minutiae"][imp] = read_minutiae_file(pair["fp_file"])

            # Load iris codes from the specified directory
            iris_path = Path(pair["iris_file"])
            if iris_dir == "2d":
                # Replace iris_codes with iris_codes_2d in path
                iris_path = Path(str(iris_path).replace("/iris_codes/", "/iris_codes_2d/"))

            if iris_path.exists():
                iris_data = np.load(str(iris_path))
                subj["iris"][imp] = {
                    "code": iris_data["code"],
                    "mask": iris_data["mask"],
                }

        subjects.append(subj)

    return subjects


def run_unimodal(subjects, factor, degree):
    """Run unimodal FP-only evaluation (baseline)."""
    lock_imp, test_imp = 0, 1

    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        if not mins:
            continue
        quantized = quantize_minutiae(mins, factor)
        if len(quantized) < degree + 1:
            continue
        vaults[cid] = create_vault(quantized, degree, seed=cid)

    ga, ia, gacc, iacc = 0, 0, 0, 0
    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        test_mins = test_subj["minutiae"].get(test_imp, [])
        if not test_mins:
            continue
        test_quantized = quantize_minutiae(test_mins, factor)
        for vault_cid, vault in vaults.items():
            is_genuine = (vault_cid == test_cid)
            success = unlock_vault(test_quantized, vault)
            if is_genuine:
                ga += 1
                if success: gacc += 1
            else:
                ia += 1
                if success: iacc += 1

    far = (iacc / ia * 100) if ia > 0 else 0
    frr = ((ga - gacc) / ga * 100) if ga > 0 else 0
    return {"arch": "Unimodal", "iris_cfg": "-", "factor": factor, "degree": degree,
            "bonus": 0, "block_size": 0, "bch_t": 0,
            "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": round(far, 6), "frr": round(frr, 6)}


def run_ibfv(subjects, factor, degree, n_bonus, block_size, bch_t, iris_cfg_label):
    """Run IBFV evaluation with specified iris config."""
    lock_imp, test_imp = 0, 1

    vaults = {}
    n_enroll_fail = 0
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        iris = subj["iris"].get(lock_imp)
        if not mins or iris is None:
            continue
        try:
            vault = lock_vault_ibfv(
                mins, iris["code"], iris["mask"],
                factor, degree,
                n_bonus=n_bonus, block_size=block_size,
                bch_t=bch_t, seed=cid
            )
        except Exception:
            n_enroll_fail += 1
            continue
        if vault is not None:
            vaults[cid] = vault

    ga, ia, gacc, iacc = 0, 0, 0, 0
    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        test_mins = test_subj["minutiae"].get(test_imp, [])
        test_iris = test_subj["iris"].get(test_imp)
        if not test_mins or test_iris is None:
            continue
        for vault_cid, vault in vaults.items():
            is_genuine = (vault_cid == test_cid)
            success = unlock_vault_ibfv(
                test_mins, test_iris["code"], test_iris["mask"], vault
            )
            if is_genuine:
                ga += 1
                if success: gacc += 1
            else:
                ia += 1
                if success: iacc += 1

    far = (iacc / ia * 100) if ia > 0 else 0
    frr = ((ga - gacc) / ga * 100) if ga > 0 else 0
    return {"arch": "IBFV", "iris_cfg": iris_cfg_label,
            "factor": factor, "degree": degree,
            "bonus": n_bonus, "block_size": block_size, "bch_t": bch_t,
            "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": round(far, 6), "frr": round(frr, 6),
            "n_enrolled": len(vaults), "n_enroll_fail": n_enroll_fail}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    summary_lines = []

    for db_name, param_combos in DB_CONFIGS:
        print(f"\n{'='*70}")
        print(f"DATABASE: {db_name}")
        print(f"{'='*70}")

        # Pre-load data for each iris type needed
        data_cache = {}
        for cfg_label, iris_dir, bs, bch_t, bonus_list in IRIS_CONFIGS:
            if iris_dir not in data_cache:
                data_cache[iris_dir] = load_chimeric_data(db_name, iris_dir)
                n_subj = len(data_cache[iris_dir])
                n_iris = sum(1 for s in data_cache[iris_dir] for _ in s["iris"])
                print(f"  Loaded {n_subj} subjects ({n_iris} iris codes) for {iris_dir}")

        for factor, degree in param_combos:
            print(f"\n--- f={factor}, k={degree} ---")

            # 1. Unimodal baseline (same for all iris configs)
            t0 = time.time()
            uni = run_unimodal(data_cache["1d"], factor, degree)
            elapsed = time.time() - t0
            print(f"  Unimodal:  FAR={uni['far']:.4f}%  FRR={uni['frr']:.4f}%  ({elapsed:.1f}s)")
            all_results.append({**uni, "db": db_name})

            # 2. IBFV with each iris config
            best_per_cfg = {}
            for cfg_label, iris_dir, bs, bch_t, bonus_list in IRIS_CONFIGS:
                subjects = data_cache[iris_dir]
                best_this_cfg = None

                for n_bonus in bonus_list:
                    t0 = time.time()
                    res = run_ibfv(subjects, factor, degree, n_bonus, bs, bch_t, cfg_label)
                    elapsed = time.time() - t0
                    print(f"  {cfg_label} bonus={n_bonus}: "
                          f"FAR={res['far']:.4f}%  FRR={res['frr']:.4f}%  "
                          f"enrolled={res['n_enrolled']}  ({elapsed:.1f}s)")
                    all_results.append({**res, "db": db_name})

                    if best_this_cfg is None or (res['far'] + res['frr']) < (best_this_cfg['far'] + best_this_cfg['frr']):
                        best_this_cfg = res

                best_per_cfg[cfg_label] = best_this_cfg

            # Summary
            parts = [f"Unimodal: FAR={uni['far']:.2f}% FRR={uni['frr']:.2f}%"]
            for cfg_label in [c[0] for c in IRIS_CONFIGS]:
                r = best_per_cfg.get(cfg_label)
                if r:
                    parts.append(f"{cfg_label}: FAR={r['far']:.2f}% FRR={r['frr']:.2f}% (bonus={r['bonus']})")
            summary = f"{db_name} f={factor} k={degree}: " + " | ".join(parts)
            summary_lines.append(summary)
            print(f"\n  >>> {summary}")

    # Save results
    csv_path = RESULTS_DIR / "ibfv_enhanced_results.csv"
    fieldnames = ["db", "arch", "iris_cfg", "factor", "degree", "bonus",
                  "block_size", "bch_t", "ga", "ia", "gacc", "iacc", "far", "frr",
                  "n_enrolled", "n_enroll_fail"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\n{'='*70}")
    print("COMPLETE SUMMARY")
    print(f"{'='*70}")
    for line in summary_lines:
        print(f"  {line}")
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
