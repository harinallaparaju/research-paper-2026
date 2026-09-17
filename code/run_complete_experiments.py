#!/usr/bin/env python3
"""
Complete Experiment Suite — Rerun with optimized iris codes.

Optimizations over v2:
  1. Iris FHD caching in ArchA (compute once per pair, test all thresholds)
  2. Multiprocessing across databases (4 parallel workers)
  3. Incremental CSV save per-setup completion

Run with: caffeinate -i python3 -u run_complete_experiments.py
"""

import sys
import csv
import gc
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file, quantize_minutiae, create_vault, unlock_vault,
)
from architectures.architecture_a import lock_vault_a, unlock_vault_a
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv
from iris_extraction.matching import fractional_hamming_distance

CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "complete_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATABASES = ["fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"]

FACTORS = [14, 16, 18, 20, 22, 24, 26, 28, 30]
DEGREES = [5, 7, 9, 11]

IBFV_BLOCK_SIZE = 255
IBFV_N_BONUS = 4

ARCH_A_THRESHOLDS = [0.32, 0.36, 0.40, 0.44]

# Parallel config — 1 worker per DB, safe on 8GB
N_DB_WORKERS = 4


def load_chimeric_data(fp_db, min_iris):
    mapping_path = CHIMERIC_DIR / f"chimeric_{fp_db}_min{min_iris}.json"
    with open(mapping_path) as f:
        mapping = json.load(f)
    subjects = []
    for entry in mapping["mapping"]:
        subj = {"chimeric_id": entry["chimeric_id"], "minutiae": {}, "iris": {}}
        for pair in entry["pairs"]:
            imp = pair["impression"] - 1
            subj["minutiae"][imp] = read_minutiae_file(pair["fp_file"])
            iris_data = np.load(pair["iris_file"])
            subj["iris"][imp] = {"code": iris_data["code"], "mask": iris_data["mask"]}
        subjects.append(subj)
    return subjects


def get_test_impressions(subjects, setup):
    if setup == 1:
        return [1]
    else:
        all_imps = set()
        for subj in subjects:
            all_imps.update(subj["minutiae"].keys())
        return sorted(imp for imp in all_imps if imp != 0)


def run_unimodal(subjects, setup, factor, degree):
    lock_imp = 0
    test_imps = get_test_impressions(subjects, setup)

    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        q = quantize_minutiae(subj["minutiae"].get(lock_imp, []), factor)
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
                    ga += 1; gacc += int(success)
                else:
                    ia += 1; iacc += int(success)

    far = iacc / ia * 100 if ia else 0
    frr = (ga - gacc) / ga * 100 if ga else 0
    return {"ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": far, "frr": frr, "eer": (far + frr) / 2,
            "n_vaults": len(vaults)}


def run_arch_a_cached(subjects, setup, factor, degree):
    """Run ArchA with FHD caching — compute iris FHD once per pair, test all thresholds."""
    lock_imp = 0
    test_imps = get_test_impressions(subjects, setup)

    # Step 1: Lock vaults for each threshold
    vaults_by_tau = {}
    for tau in ARCH_A_THRESHOLDS:
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
        vaults_by_tau[tau] = vaults

    # Step 2: Compute iris FHD once per (test_subj, test_imp, vault_subj) pair
    fhd_cache = {}
    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        for test_imp in test_imps:
            test_iris = test_subj["iris"].get(test_imp)
            if test_iris is None:
                continue
            for vault_subj in subjects:
                vault_cid = vault_subj["chimeric_id"]
                vault_iris = vault_subj["iris"].get(lock_imp)
                if vault_iris is None:
                    continue
                fhd = fractional_hamming_distance(
                    vault_iris["code"], vault_iris["mask"],
                    test_iris["code"], test_iris["mask"],
                    max_shift=32
                )
                fhd_cache[(test_cid, test_imp, vault_cid)] = fhd

    # Step 3: Evaluate each threshold using cached FHD
    best = None
    for tau in ARCH_A_THRESHOLDS:
        vaults = vaults_by_tau[tau]
        ga, ia, gacc, iacc = 0, 0, 0, 0

        for test_subj in subjects:
            test_cid = test_subj["chimeric_id"]
            for test_imp in test_imps:
                test_mins = test_subj["minutiae"].get(test_imp, [])
                if not test_mins:
                    continue
                for vault_cid, vault in vaults.items():
                    fhd = fhd_cache.get((test_cid, test_imp, vault_cid))
                    if fhd is None:
                        continue

                    if fhd >= tau:
                        success = False
                    else:
                        test_q = quantize_minutiae(test_mins, vault.factor)
                        success = unlock_vault(test_q, vault.fp_vault)

                    if vault_cid == test_cid:
                        ga += 1; gacc += int(success)
                    else:
                        ia += 1; iacc += int(success)

        far = iacc / ia * 100 if ia else 0
        frr = (ga - gacc) / ga * 100 if ga else 0
        eer = (far + frr) / 2

        result = {"ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
                  "far": far, "frr": frr, "eer": eer,
                  "tau": tau, "n_vaults": len(vaults)}

        if best is None or eer < best["eer"]:
            best = result

    return best


def run_ibfv(subjects, setup, factor, degree):
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
            factor, degree, n_bonus=IBFV_N_BONUS,
            block_size=IBFV_BLOCK_SIZE, seed=cid
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
                    ga += 1; gacc += int(success)
                else:
                    ia += 1; iacc += int(success)

    far = iacc / ia * 100 if ia else 0
    frr = (ga - gacc) / ga * 100 if ga else 0
    return {"ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
            "far": far, "frr": frr, "eer": (far + frr) / 2,
            "n_vaults": len(vaults)}


FIELDNAMES = [
    "database", "setup", "architecture", "factor", "degree",
    "iris_threshold", "block_size", "n_bonus",
    "n_subjects", "n_vaults",
    "ga", "ia", "gacc", "iacc",
    "far_pct", "frr_pct", "eer_pct",
]


def run_one_db_setup(args):
    """Run all experiments for one (database, setup). Designed for multiprocessing."""
    db_name, setup = args
    min_iris = 2 if setup == 1 else 8

    print(f"\n{'='*80}", flush=True)
    print(f"DATABASE: {db_name} | SETUP {setup} (min_iris={min_iris})", flush=True)
    print(f"{'='*80}", flush=True)

    subjects = load_chimeric_data(db_name, min_iris)
    n_subj = len(subjects)
    test_imps = get_test_impressions(subjects, setup)
    print(f"  Loaded {n_subj} subjects, test impressions: {test_imps}", flush=True)

    rows = []

    for factor in FACTORS:
        for degree in DEGREES:
            tag = f"S{setup} {db_name} f={factor:2d} k={degree:2d}"

            # --- Unimodal ---
            t0 = time.time()
            uni = run_unimodal(subjects, setup, factor, degree)
            t_uni = time.time() - t0
            print(f"  {tag} Uni:  EER={uni['eer']:8.4f}%  "
                  f"FAR={uni['far']:8.4f}% FRR={uni['frr']:8.4f}%  "
                  f"GA={uni['ga']} IA={uni['ia']}  ({t_uni:.1f}s)", flush=True)
            rows.append({
                "database": db_name, "setup": setup,
                "architecture": "Unimodal",
                "factor": factor, "degree": degree,
                "iris_threshold": "", "block_size": "", "n_bonus": "",
                "n_subjects": n_subj, "n_vaults": uni["n_vaults"],
                "ga": uni["ga"], "ia": uni["ia"],
                "gacc": uni["gacc"], "iacc": uni["iacc"],
                "far_pct": round(uni["far"], 6),
                "frr_pct": round(uni["frr"], 6),
                "eer_pct": round(uni["eer"], 6),
            })

            # --- Architecture A (cached FHD) ---
            t0 = time.time()
            a_res = run_arch_a_cached(subjects, setup, factor, degree)
            t_a = time.time() - t0
            print(f"  {tag} ArchA: EER={a_res['eer']:8.4f}%  "
                  f"FAR={a_res['far']:8.4f}% FRR={a_res['frr']:8.4f}%  "
                  f"τ={a_res['tau']}  ({t_a:.1f}s)", flush=True)
            rows.append({
                "database": db_name, "setup": setup,
                "architecture": "A",
                "factor": factor, "degree": degree,
                "iris_threshold": a_res["tau"], "block_size": "", "n_bonus": "",
                "n_subjects": n_subj, "n_vaults": a_res["n_vaults"],
                "ga": a_res["ga"], "ia": a_res["ia"],
                "gacc": a_res["gacc"], "iacc": a_res["iacc"],
                "far_pct": round(a_res["far"], 6),
                "frr_pct": round(a_res["frr"], 6),
                "eer_pct": round(a_res["eer"], 6),
            })

            # --- IBFV ---
            t0 = time.time()
            ibfv = run_ibfv(subjects, setup, factor, degree)
            t_ibfv = time.time() - t0
            delta = ibfv["eer"] - uni["eer"]
            marker = " ★" if delta < -0.01 else ""
            print(f"  {tag} IBFV:  EER={ibfv['eer']:8.4f}%  "
                  f"FAR={ibfv['far']:8.4f}% FRR={ibfv['frr']:8.4f}%  "
                  f"ΔEER={delta:+.4f}%{marker}  ({t_ibfv:.1f}s)", flush=True)
            rows.append({
                "database": db_name, "setup": setup,
                "architecture": "IBFV",
                "factor": factor, "degree": degree,
                "iris_threshold": "",
                "block_size": IBFV_BLOCK_SIZE, "n_bonus": IBFV_N_BONUS,
                "n_subjects": n_subj, "n_vaults": ibfv["n_vaults"],
                "ga": ibfv["ga"], "ia": ibfv["ia"],
                "gacc": ibfv["gacc"], "iacc": ibfv["iacc"],
                "far_pct": round(ibfv["far"], 6),
                "frr_pct": round(ibfv["frr"], 6),
                "eer_pct": round(ibfv["eer"], 6),
            })

            gc.collect()

    print(f"\n  {db_name} Setup {setup} DONE — {len(rows)} rows", flush=True)
    return rows


def save_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print(f"\n{'='*80}")
    print("COMPLETE RESULTS SUMMARY")
    print(f"{'='*80}")

    for setup in [1, 2]:
        setup_rows = [r for r in rows if r["setup"] == setup]
        if not setup_rows:
            continue

        print(f"\n--- SETUP {setup} ---")
        print(f"{'DB':<12} {'f':>3} {'k':>3}  "
              f"{'Uni EER%':>10} {'ArchA EER%':>12} {'IBFV EER%':>11} {'ΔEER%':>8}")
        print("-" * 70)

        for db in DATABASES:
            db_rows = [r for r in setup_rows if r["database"] == db]
            if not db_rows:
                continue
            for factor in FACTORS:
                for degree in DEGREES:
                    uni_r = [r for r in db_rows if r["architecture"] == "Unimodal"
                             and r["factor"] == factor and r["degree"] == degree]
                    a_r = [r for r in db_rows if r["architecture"] == "A"
                            and r["factor"] == factor and r["degree"] == degree]
                    ibfv_r = [r for r in db_rows if r["architecture"] == "IBFV"
                              and r["factor"] == factor and r["degree"] == degree]
                    if not uni_r:
                        continue
                    u_eer = uni_r[0]["eer_pct"]
                    a_eer = a_r[0]["eer_pct"] if a_r else float("nan")
                    i_eer = ibfv_r[0]["eer_pct"] if ibfv_r else float("nan")
                    delta = i_eer - u_eer
                    marker = " ★" if delta < -0.01 else ""
                    print(f"{db:<12} {factor:3d} {degree:3d}  "
                          f"{u_eer:10.4f} {a_eer:12.4f} {i_eer:11.4f} "
                          f"{delta:+8.4f}{marker}")

        print(f"\n  BEST EER per database (Setup {setup}):")
        for db in DATABASES:
            uni_rows = [r for r in setup_rows if r["database"] == db
                        and r["architecture"] == "Unimodal"]
            ibfv_rows = [r for r in setup_rows if r["database"] == db
                         and r["architecture"] == "IBFV"]
            if not uni_rows:
                continue
            best_uni = min(uni_rows, key=lambda r: r["eer_pct"])
            best_ibfv = min(ibfv_rows, key=lambda r: r["eer_pct"]) if ibfv_rows else None
            if best_ibfv:
                print(f"    {db}: Uni={best_uni['eer_pct']:.4f}% "
                      f"(f={best_uni['factor']},k={best_uni['degree']}) | "
                      f"IBFV={best_ibfv['eer_pct']:.4f}% "
                      f"(f={best_ibfv['factor']},k={best_ibfv['degree']}) | "
                      f"Δ={best_ibfv['eer_pct']-best_uni['eer_pct']:+.4f}%")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows = []
    total_start = time.time()

    for setup in [1, 2]:
        setup_start = time.time()
        csv_path = RESULTS_DIR / f"setup{setup}_results_{timestamp}.csv"

        # Build job list for this setup
        jobs = [(db, setup) for db in DATABASES]

        print(f"\n{'#'*80}")
        print(f"# SETUP {setup} — Running {len(jobs)} databases in parallel "
              f"({N_DB_WORKERS} workers)")
        print(f"{'#'*80}", flush=True)

        # Run databases in parallel
        with Pool(N_DB_WORKERS) as pool:
            results = pool.map(run_one_db_setup, jobs)

        # Collect all rows
        setup_rows = []
        for db_rows in results:
            setup_rows.extend(db_rows)

        # Save per-setup CSV
        save_csv(setup_rows, csv_path)
        all_rows.extend(setup_rows)

        setup_elapsed = time.time() - setup_start
        print(f"\nSetup {setup} complete: {len(setup_rows)} rows in "
              f"{setup_elapsed:.0f}s ({setup_elapsed/3600:.1f}h)")
        print(f"Saved to: {csv_path}", flush=True)

        gc.collect()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE — {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"Total rows: {len(all_rows)}")
    print(f"{'='*80}")

    print_summary(all_rows)


if __name__ == "__main__":
    main()
