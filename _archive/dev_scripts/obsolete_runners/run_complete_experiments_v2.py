"""
Complete Experiment Suite — Senior's Paper Replication + IBFV.

Runs BOTH Setup 1 and Setup 2 across all 4 databases.

Strategy: Run targeted key (factor, degree) combos that cover:
  1. Senior's published best results
  2. High-EER regime where IBFV shows improvement
  3. Multiple operating points for complete characterization

Uses multiprocessing to parallelize across databases.
"""

import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
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

DATABASES = ["fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"]

# Full parameter sweep matching senior's paper
FACTORS = [14, 16, 18, 20, 22, 24, 26, 28, 30]
DEGREES = [5, 7, 9, 11]

# IBFV config
IBFV_BLOCK_SIZE = 63
IBFV_N_BONUS = 4

# Architecture A thresholds
ARCH_A_THRESHOLDS = [0.32, 0.36, 0.40, 0.44]


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


def run_arch_a_best(subjects, setup, factor, degree):
    """Run ArchA with multiple thresholds, return best."""
    lock_imp = 0
    test_imps = get_test_impressions(subjects, setup)

    best = None
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


def run_one_db_setup(db_name, setup, all_rows, csv_path, fieldnames):
    """Run all experiments for one database and setup."""
    min_iris = 2 if setup == 1 else 8
    print(f"\n{'='*80}")
    print(f"DATABASE: {db_name} | SETUP {setup} (min_iris={min_iris})")
    print(f"{'='*80}")

    subjects = load_chimeric_data(db_name, min_iris)
    n_subj = len(subjects)
    test_imps = get_test_impressions(subjects, setup)
    print(f"Loaded {n_subj} subjects, test impressions: {test_imps}")

    for factor in FACTORS:
        for degree in DEGREES:
            tag = f"S{setup} {db_name} f={factor:2d} k={degree:2d}"

            # --- Unimodal ---
            t0 = time.time()
            uni = run_unimodal(subjects, setup, factor, degree)
            t_uni = time.time() - t0
            print(f"  {tag} Unimodal:  EER={uni['eer']:8.4f}%  "
                  f"FAR={uni['far']:8.4f}% FRR={uni['frr']:8.4f}%  "
                  f"GA={uni['ga']} IA={uni['ia']}  ({t_uni:.1f}s)")
            all_rows.append({
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

            # --- Architecture A (best threshold) ---
            t0 = time.time()
            a_res = run_arch_a_best(subjects, setup, factor, degree)
            t_a = time.time() - t0
            print(f"  {tag} ArchA:     EER={a_res['eer']:8.4f}%  "
                  f"FAR={a_res['far']:8.4f}% FRR={a_res['frr']:8.4f}%  "
                  f"τ={a_res['tau']}  ({t_a:.1f}s)")
            all_rows.append({
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
            print(f"  {tag} IBFV:      EER={ibfv['eer']:8.4f}%  "
                  f"FAR={ibfv['far']:8.4f}% FRR={ibfv['frr']:8.4f}%  "
                  f"ΔEER={delta:+.4f}%  ({t_ibfv:.1f}s){marker}")
            all_rows.append({
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

            # Save incrementally
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)


def print_summary(rows):
    """Print result summary tables."""
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

        # Best results per DB
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
            f1, k1 = best_uni["factor"], best_uni["degree"]
            eer1 = best_uni["eer_pct"]
            if best_ibfv:
                f2, k2 = best_ibfv["factor"], best_ibfv["degree"]
                eer2 = best_ibfv["eer_pct"]
                print(f"    {db}: Unimodal={eer1:.4f}% (f={f1},k={k1}) | "
                      f"IBFV={eer2:.4f}% (f={f2},k={k2}) | Δ={eer2-eer1:+.4f}%")
            else:
                print(f"    {db}: Unimodal={eer1:.4f}% (f={f1},k={k1})")


def main():
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

    # Run Setup 1 first (faster), then Setup 2
    for setup in [1, 2]:
        for db_name in DATABASES:
            run_one_db_setup(db_name, setup, all_rows, csv_path, fieldnames)

    total_elapsed = time.time() - total_start
    hours = total_elapsed / 3600
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE — {total_elapsed:.0f}s ({hours:.1f}h)")
    print(f"Results saved to: {csv_path}")
    print(f"Total rows: {len(all_rows)}")
    print(f"{'='*80}")

    print_summary(all_rows)


if __name__ == "__main__":
    main()
