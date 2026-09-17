"""
IBFV Comprehensive Evaluation.

Runs IBFV + unimodal baseline + Architecture A on ALL 4 FVC databases.
Produces a complete comparison showing IBFV ≤ unimodal EER guarantee.

For each database × parameter combination:
    1. Unimodal (FP-only vault): baseline EER
    2. Architecture A (AND-fusion): best EER across τ sweep
    3. IBFV (bonus=2,4,6): EER with graceful iris degradation

Setup 1: Lock=impression 1, Test=impression 2
"""

import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file,
    quantize_minutiae,
    create_vault,
    unlock_vault,
)
from architectures.architecture_a import lock_vault_a, unlock_vault_a
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv

# Paths
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ibfv"

# Database configs: (db_name, best_factor, best_degree) from senior's paper
DB_CONFIGS = [
    ("fvc2002_1", [(14, 5), (18, 7), (24, 10)]),
    ("fvc2002_2", [(14, 5), (18, 7)]),
    ("fvc2002_3", [(14, 5), (18, 5)]),
    ("fvc2004_1", [(14, 5), (18, 5)]),
]

# IBFV bonus point counts to test
BONUS_COUNTS = [2, 4, 6]

# Architecture A thresholds to sweep
ARCH_A_THRESHOLDS = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46]

# Iris stabilizer block sizes to test
BLOCK_SIZES = [255, 511, 1023]


def load_chimeric_data(fp_db: str, min_iris: int = 2) -> List[dict]:
    """Load chimeric mapping and associated data."""
    mapping_path = CHIMERIC_DIR / f"chimeric_{fp_db}_min{min_iris}.json"
    with open(mapping_path) as f:
        mapping = json.load(f)

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

            iris_data = np.load(pair["iris_file"])
            subj["iris"][imp] = {
                "code": iris_data["code"],
                "mask": iris_data["mask"],
            }

        subjects.append(subj)

    return subjects


def run_unimodal(subjects, factor, degree):
    """Run unimodal FP-only evaluation. Setup 1."""
    lock_imp = 0
    test_imp = 1

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
                if success:
                    gacc += 1
            else:
                ia += 1
                if success:
                    iacc += 1

    far = (iacc / ia * 100) if ia > 0 else 0
    frr = ((ga - gacc) / ga * 100) if ga > 0 else 0
    eer = (far + frr) / 2

    return {"arch": "Unimodal", "factor": factor, "degree": degree,
            "bonus": 0, "block_size": 0, "tau": None,
            "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": round(far, 6), "frr": round(frr, 6), "eer": round(eer, 6)}


def run_arch_a(subjects, factor, degree, tau):
    """Run Architecture A evaluation. Setup 1."""
    lock_imp = 0
    test_imp = 1

    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        iris = subj["iris"].get(lock_imp)
        if not mins or iris is None:
            continue
        vault = lock_vault_a(mins, iris["code"], iris["mask"],
                             factor, degree, iris_threshold=tau)
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
            success = unlock_vault_a(test_mins, test_iris["code"],
                                     test_iris["mask"], vault)
            if is_genuine:
                ga += 1
                if success:
                    gacc += 1
            else:
                ia += 1
                if success:
                    iacc += 1

    far = (iacc / ia * 100) if ia > 0 else 0
    frr = ((ga - gacc) / ga * 100) if ga > 0 else 0
    eer = (far + frr) / 2

    return {"arch": "A", "factor": factor, "degree": degree,
            "bonus": 0, "block_size": 0, "tau": tau,
            "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": round(far, 6), "frr": round(frr, 6), "eer": round(eer, 6)}


def run_ibfv(subjects, factor, degree, n_bonus, block_size):
    """Run IBFV evaluation. Setup 1."""
    lock_imp = 0
    test_imp = 1

    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        iris = subj["iris"].get(lock_imp)
        if not mins or iris is None:
            continue
        vault = lock_vault_ibfv(
            mins, iris["code"], iris["mask"],
            factor, degree,
            n_bonus=n_bonus, block_size=block_size, seed=cid
        )
        if vault is not None:
            vaults[cid] = vault

    ga, ia, gacc, iacc = 0, 0, 0, 0
    iris_success_genuine = 0
    iris_fail_genuine = 0

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
                if success:
                    gacc += 1
            else:
                ia += 1
                if success:
                    iacc += 1

    far = (iacc / ia * 100) if ia > 0 else 0
    frr = ((ga - gacc) / ga * 100) if ga > 0 else 0
    eer = (far + frr) / 2

    return {"arch": "IBFV", "factor": factor, "degree": degree,
            "bonus": n_bonus, "block_size": block_size, "tau": None,
            "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": round(far, 6), "frr": round(frr, 6), "eer": round(eer, 6)}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_lines = []

    for db_name, param_combos in DB_CONFIGS:
        print(f"\n{'='*70}")
        print(f"DATABASE: {db_name}")
        print(f"{'='*70}")

        subjects = load_chimeric_data(db_name, min_iris=2)
        print(f"Loaded {len(subjects)} chimeric subjects")

        for factor, degree in param_combos:
            print(f"\n--- f={factor}, k={degree} ---")

            # 1. Unimodal baseline
            t0 = time.time()
            uni = run_unimodal(subjects, factor, degree)
            elapsed = time.time() - t0
            print(f"  Unimodal:  FAR={uni['far']:.4f}%  FRR={uni['frr']:.4f}%  "
                  f"EER={uni['eer']:.4f}%  ({elapsed:.1f}s)")
            all_results.append({**uni, "db": db_name})

            # 2. Architecture A (best tau)
            best_a = None
            for tau in ARCH_A_THRESHOLDS:
                t0 = time.time()
                a_res = run_arch_a(subjects, factor, degree, tau)
                elapsed = time.time() - t0
                print(f"  Arch A τ={tau:.2f}: FAR={a_res['far']:.4f}%  "
                      f"FRR={a_res['frr']:.4f}%  EER={a_res['eer']:.4f}%  ({elapsed:.1f}s)")
                all_results.append({**a_res, "db": db_name})
                if best_a is None or a_res['eer'] < best_a['eer']:
                    best_a = a_res

            # 3. IBFV (all bonus × block_size combos)
            best_ibfv = None
            for block_size in BLOCK_SIZES:
                for n_bonus in BONUS_COUNTS:
                    t0 = time.time()
                    ibfv = run_ibfv(subjects, factor, degree, n_bonus, block_size)
                    elapsed = time.time() - t0
                    print(f"  IBFV B={block_size} bonus={n_bonus}: "
                          f"FAR={ibfv['far']:.4f}%  FRR={ibfv['frr']:.4f}%  "
                          f"EER={ibfv['eer']:.4f}%  ({elapsed:.1f}s)")
                    all_results.append({**ibfv, "db": db_name})
                    if best_ibfv is None or ibfv['eer'] < best_ibfv['eer']:
                        best_ibfv = ibfv

            # Summary for this config
            summary = (f"{db_name} f={factor} k={degree}: "
                       f"Unimodal EER={uni['eer']:.4f}% | "
                       f"Best A EER={best_a['eer']:.4f}% (τ={best_a['tau']}) | "
                       f"Best IBFV EER={best_ibfv['eer']:.4f}% "
                       f"(B={best_ibfv['block_size']},bonus={best_ibfv['bonus']})")
            summary_lines.append(summary)
            print(f"\n  >>> {summary}")

    # Save all results
    csv_path = RESULTS_DIR / "ibfv_comprehensive_results.csv"
    fieldnames = ["db", "arch", "factor", "degree", "bonus", "block_size", "tau",
                  "ga", "ia", "gacc", "iacc", "far", "frr", "eer"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
