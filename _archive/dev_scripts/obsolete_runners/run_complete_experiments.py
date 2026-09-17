"""
Complete Experiment Suite — Replicating Senior's Paper + IBFV Evaluation.

Runs BOTH Setup 1 and Setup 2 across all 4 databases with full parameter sweep.
Matches the senior's JISAA 2025 methodology exactly.

Setup 1: Lock=impression 1, Test=impression 2 only
    - Uses chimeric_*_min2.json (100 subjects for DB1/DB2/DB4, 98 for DB3)
    - GA = N_subj, IA = N_subj × (N_subj - 1)

Setup 2: Lock=impression 1, Test=impressions 2-8 (all other)
    - Uses chimeric_*_min8.json (48 subjects, all with 8 impressions)
    - GA = N_subj × N_test_imps, IA = GA × (N_subj - 1)

Parameters (matching senior's paper):
    Factors: [14, 16, 18, 20, 22, 24, 26, 28, 30]
    Degrees: [5, 7, 9, 11]

Architectures:
    1. Unimodal (fingerprint-only fuzzy vault)
    2. Architecture A (Decision AND-fusion, best τ)
    3. IBFV (Iris-Boosted Fuzzy Vault, B=63, bonus=4)

Output: CSV files with all metrics per (database, setup, architecture, factor, degree)
"""

import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
from itertools import combinations
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file, quantize_minutiae, create_vault, unlock_vault,
)
from architectures.architecture_a import lock_vault_a, unlock_vault_a
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv

CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "complete_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters matching senior's paper
FACTORS = [14, 16, 18, 20, 22, 24, 26, 28, 30]
DEGREES = [5, 7, 9, 11]
DATABASES = ["fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"]

# IBFV config (B=63 is secure sweet spot)
IBFV_BLOCK_SIZE = 63
IBFV_N_BONUS = 4

# Architecture A thresholds to sweep
ARCH_A_THRESHOLDS = [0.32, 0.36, 0.40, 0.44]


def load_chimeric_data(fp_db, min_iris):
    """Load chimeric mapping with all subject data."""
    mapping_path = CHIMERIC_DIR / f"chimeric_{fp_db}_min{min_iris}.json"
    with open(mapping_path) as f:
        mapping = json.load(f)
    subjects = []
    for entry in mapping["mapping"]:
        subj = {"chimeric_id": entry["chimeric_id"], "minutiae": {}, "iris": {}}
        for pair in entry["pairs"]:
            imp = pair["impression"] - 1  # 0-indexed
            subj["minutiae"][imp] = read_minutiae_file(pair["fp_file"])
            iris_data = np.load(pair["iris_file"])
            subj["iris"][imp] = {"code": iris_data["code"], "mask": iris_data["mask"]}
        subjects.append(subj)
    return subjects


def get_test_impressions(subjects, setup):
    """Get test impression indices based on setup."""
    if setup == 1:
        return [1]  # Only impression 2
    else:
        # All impressions except 0 (lock impression)
        all_imps = set()
        for subj in subjects:
            all_imps.update(subj["minutiae"].keys())
        return sorted(imp for imp in all_imps if imp != 0)


def run_unimodal(subjects, setup, factor, degree):
    """Run unimodal fingerprint-only vault evaluation."""
    lock_imp = 0
    test_imps = get_test_impressions(subjects, setup)

    # Create vaults
    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        q = quantize_minutiae(mins, factor)
        if len(q) >= degree + 1:
            vaults[cid] = create_vault(q, degree, seed=cid)

    ga, ia, gacc, iacc = 0, 0, 0, 0
    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        for test_imp in test_imps:
            test_mins = test_subj["minutiae"].get(test_imp, [])
            if not test_mins:
                continue
            test_q = quantize_minutiae(test_mins, factor)
            for vault_cid, vault in vaults.items():
                success = unlock_vault(test_q, vault)
                if vault_cid == test_cid:
                    ga += 1
                    gacc += int(success)
                else:
                    ia += 1
                    iacc += int(success)

    far = iacc / ia * 100 if ia else 0
    frr = (ga - gacc) / ga * 100 if ga else 0
    eer = (far + frr) / 2
    return {"ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": far, "frr": frr, "eer": eer, "n_vaults": len(vaults)}


def run_arch_a(subjects, setup, factor, degree, tau):
    """Run Architecture A (AND-fusion) evaluation."""
    lock_imp = 0
    test_imps = get_test_impressions(subjects, setup)

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
        for test_imp in test_imps:
            test_mins = test_subj["minutiae"].get(test_imp, [])
            test_iris = test_subj["iris"].get(test_imp)
            if not test_mins or test_iris is None:
                continue
            for vault_cid, vault in vaults.items():
                success = unlock_vault_a(test_mins, test_iris["code"],
                                         test_iris["mask"], vault)
                if vault_cid == test_cid:
                    ga += 1
                    gacc += int(success)
                else:
                    ia += 1
                    iacc += int(success)

    far = iacc / ia * 100 if ia else 0
    frr = (ga - gacc) / ga * 100 if ga else 0
    eer = (far + frr) / 2
    return {"ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": far, "frr": frr, "eer": eer, "n_vaults": len(vaults)}


def run_ibfv(subjects, setup, factor, degree, n_bonus=IBFV_N_BONUS,
             block_size=IBFV_BLOCK_SIZE):
    """Run IBFV (Iris-Boosted Fuzzy Vault) evaluation."""
    lock_imp = 0
    test_imps = get_test_impressions(subjects, setup)

    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        iris = subj["iris"].get(lock_imp)
        if not mins or iris is None:
            continue
        vault = lock_vault_ibfv(
            mins, iris["code"], iris["mask"],
            factor, degree, n_bonus=n_bonus,
            block_size=block_size, seed=cid
        )
        if vault is not None:
            vaults[cid] = vault

    ga, ia, gacc, iacc = 0, 0, 0, 0
    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        for test_imp in test_imps:
            test_mins = test_subj["minutiae"].get(test_imp, [])
            test_iris = test_subj["iris"].get(test_imp)
            if not test_mins or test_iris is None:
                continue
            for vault_cid, vault in vaults.items():
                success = unlock_vault_ibfv(
                    test_mins, test_iris["code"], test_iris["mask"], vault
                )
                if vault_cid == test_cid:
                    ga += 1
                    gacc += int(success)
                else:
                    ia += 1
                    iacc += int(success)

    far = iacc / ia * 100 if ia else 0
    frr = (ga - gacc) / ga * 100 if ga else 0
    eer = (far + frr) / 2
    return {"ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": far, "frr": frr, "eer": eer, "n_vaults": len(vaults)}


def run_all_experiments():
    """Run complete experiment suite."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"complete_results_{timestamp}.csv"

    fieldnames = [
        "database", "setup", "architecture", "factor", "degree",
        "iris_threshold", "block_size", "n_bonus",
        "n_subjects", "n_vaults",
        "ga", "ia", "gacc", "iacc",
        "far_pct", "frr_pct", "eer_pct",
    ]

    all_rows = []
    total_start = time.time()

    for db_name in DATABASES:
        for setup in [1, 2]:
            min_iris = 2 if setup == 1 else 8
            print(f"\n{'='*80}")
            print(f"DATABASE: {db_name} | SETUP {setup} (min_iris={min_iris})")
            print(f"{'='*80}")

            subjects = load_chimeric_data(db_name, min_iris)
            n_subj = len(subjects)
            print(f"Loaded {n_subj} chimeric subjects")

            for factor in FACTORS:
                for degree in DEGREES:
                    # --- Unimodal ---
                    t0 = time.time()
                    uni = run_unimodal(subjects, setup, factor, degree)
                    elapsed = time.time() - t0
                    print(f"  f={factor:2d} k={degree:2d} Unimodal:     "
                          f"FAR={uni['far']:8.4f}% FRR={uni['frr']:8.4f}% "
                          f"EER={uni['eer']:8.4f}%  "
                          f"GA={uni['ga']} IA={uni['ia']} "
                          f"({elapsed:.1f}s)")
                    all_rows.append({
                        "database": db_name, "setup": setup,
                        "architecture": "Unimodal",
                        "factor": factor, "degree": degree,
                        "iris_threshold": "", "block_size": "", "n_bonus": "",
                        "n_subjects": n_subj, "n_vaults": uni["n_vaults"],
                        **{k: uni[k] for k in ["ga", "ia", "gacc", "iacc"]},
                        "far_pct": round(uni["far"], 6),
                        "frr_pct": round(uni["frr"], 6),
                        "eer_pct": round(uni["eer"], 6),
                    })

                    # --- Architecture A (best threshold) ---
                    best_a = None
                    for tau in ARCH_A_THRESHOLDS:
                        t0 = time.time()
                        a_res = run_arch_a(subjects, setup, factor, degree, tau)
                        elapsed = time.time() - t0
                        print(f"  f={factor:2d} k={degree:2d} ArchA τ={tau}: "
                              f"FAR={a_res['far']:8.4f}% FRR={a_res['frr']:8.4f}% "
                              f"EER={a_res['eer']:8.4f}%  ({elapsed:.1f}s)")
                        if best_a is None or a_res["eer"] < best_a["eer"]:
                            best_a = a_res
                            best_a["tau"] = tau

                    all_rows.append({
                        "database": db_name, "setup": setup,
                        "architecture": "A",
                        "factor": factor, "degree": degree,
                        "iris_threshold": best_a["tau"],
                        "block_size": "", "n_bonus": "",
                        "n_subjects": n_subj, "n_vaults": best_a["n_vaults"],
                        **{k: best_a[k] for k in ["ga", "ia", "gacc", "iacc"]},
                        "far_pct": round(best_a["far"], 6),
                        "frr_pct": round(best_a["frr"], 6),
                        "eer_pct": round(best_a["eer"], 6),
                    })

                    # --- IBFV ---
                    t0 = time.time()
                    ibfv = run_ibfv(subjects, setup, factor, degree)
                    elapsed = time.time() - t0
                    delta = ibfv["eer"] - uni["eer"]
                    print(f"  f={factor:2d} k={degree:2d} IBFV B=63:    "
                          f"FAR={ibfv['far']:8.4f}% FRR={ibfv['frr']:8.4f}% "
                          f"EER={ibfv['eer']:8.4f}%  "
                          f"ΔEER={delta:+.4f}%  ({elapsed:.1f}s)")
                    all_rows.append({
                        "database": db_name, "setup": setup,
                        "architecture": "IBFV",
                        "factor": factor, "degree": degree,
                        "iris_threshold": "",
                        "block_size": IBFV_BLOCK_SIZE,
                        "n_bonus": IBFV_N_BONUS,
                        "n_subjects": n_subj, "n_vaults": ibfv["n_vaults"],
                        **{k: ibfv[k] for k in ["ga", "ia", "gacc", "iacc"]},
                        "far_pct": round(ibfv["far"], 6),
                        "frr_pct": round(ibfv["frr"], 6),
                        "eer_pct": round(ibfv["eer"], 6),
                    })

                    # Write incrementally after each (factor, degree)
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(all_rows)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE — {total_elapsed:.0f}s total")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*80}")

    # Print summary table
    print_summary(all_rows)


def print_summary(rows):
    """Print summary comparing Unimodal vs IBFV for each setup/database."""
    print(f"\n{'='*80}")
    print("SUMMARY: Best EER per Database/Setup")
    print(f"{'='*80}")

    for setup in [1, 2]:
        print(f"\n--- Setup {setup} ---")
        print(f"{'Database':<12} {'f':>3} {'k':>3}  "
              f"{'Unimodal':>10} {'ArchA':>10} {'IBFV':>10} {'ΔEER':>8}")
        print("-" * 65)

        for db in DATABASES:
            # Find best unimodal EER for this DB/Setup
            uni_rows = [r for r in rows if r["database"] == db
                        and r["setup"] == setup and r["architecture"] == "Unimodal"]
            if not uni_rows:
                continue
            best_uni = min(uni_rows, key=lambda r: r["eer_pct"])
            f_best, k_best = best_uni["factor"], best_uni["degree"]

            # Find matching ArchA and IBFV
            a_rows = [r for r in rows if r["database"] == db
                      and r["setup"] == setup and r["architecture"] == "A"
                      and r["factor"] == f_best and r["degree"] == k_best]
            ibfv_rows = [r for r in rows if r["database"] == db
                         and r["setup"] == setup and r["architecture"] == "IBFV"
                         and r["factor"] == f_best and r["degree"] == k_best]

            a_eer = a_rows[0]["eer_pct"] if a_rows else float("nan")
            ibfv_eer = ibfv_rows[0]["eer_pct"] if ibfv_rows else float("nan")
            delta = ibfv_eer - best_uni["eer_pct"]

            print(f"{db:<12} {f_best:3d} {k_best:3d}  "
                  f"{best_uni['eer_pct']:10.4f} {a_eer:10.4f} "
                  f"{ibfv_eer:10.4f} {delta:+8.4f}")

            # Also show best IBFV across all params (might be different f,k)
            ibfv_all = [r for r in rows if r["database"] == db
                        and r["setup"] == setup and r["architecture"] == "IBFV"]
            if ibfv_all:
                best_ibfv = min(ibfv_all, key=lambda r: r["eer_pct"])
                if best_ibfv["factor"] != f_best or best_ibfv["degree"] != k_best:
                    print(f"  (Best IBFV @ f={best_ibfv['factor']},k={best_ibfv['degree']}: "
                          f"EER={best_ibfv['eer_pct']:.4f}%)")


if __name__ == "__main__":
    run_all_experiments()
