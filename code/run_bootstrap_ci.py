#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for IBFV Paper.

Runs key experiments with per-trial outcome storage, then performs
subject-level bootstrap resampling for correct 95% CIs on GAR, FAR, and EER.

Subject-level bootstrap is essential because trials from the same subject
are correlated (same biometric template). Trial-level bootstrap would
underestimate CI width.

Usage:
    caffeinate -i python3 -u run_bootstrap_ci.py
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file, quantize_minutiae, create_vault, unlock_vault,
)
from architectures.architecture_a import lock_vault_a, unlock_vault_a
from architectures.architecture_b import lock_vault_b, unlock_vault_b
from architectures.architecture_c import lock_vault_c, unlock_vault_c
from architectures.architecture_d import lock_vault_d, unlock_vault_d
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv
from iris_stabilizer import IrisStabilizer

CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "paper_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATABASES = ["fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"]
DB_NAMES = {"fvc2002_1": "DB1", "fvc2002_2": "DB2", "fvc2002_3": "DB3", "fvc2004_1": "DB4"}

IBFV_BLOCK_SIZE = 255  # Match main experiments: Bs=255 for 131K codes → 514 key bits
IBFV_N_BONUS = 4

# Best configs per DB (same as main experiments)
BEST_CONFIGS = {
    "fvc2002_1": [(22, 7), (18, 5)],
    "fvc2002_2": [(22, 7), (18, 5)],
    "fvc2002_3": [(16, 5), (18, 5)],
    "fvc2004_1": [(22, 5), (22, 7)],
}

N_BOOTSTRAP = 2000
CI_ALPHA = 0.05  # 95% CI


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


def run_trials_per_subject(subjects, lock_fn, unlock_fn, factor, degree, lock_kwargs,
                           is_unimodal=False):
    """
    Run all genuine + impostor trials and return per-subject outcomes.

    Returns:
        per_subject: dict mapping chimeric_id -> {
            "genuine_success": bool (did genuine unlock succeed?),
            "impostor_accepts": int (how many impostors opened this vault),
            "n_impostors": int (total impostor attempts on this vault)
        }
    """
    n_subj = len(subjects)
    per_subject = {}

    # Enrollment: impression 0 locks, impression 1 tests
    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(0)
        iris = subj["iris"].get(0)
        if mins is None or iris is None or len(mins) == 0:
            continue
        try:
            if is_unimodal:
                q_mins = quantize_minutiae(mins, factor)
                vault = create_vault(q_mins, degree, seed=cid)
            else:
                vault = lock_fn(mins, iris["code"], iris["mask"], factor, degree, seed=cid, **lock_kwargs)
            vaults[cid] = vault
        except Exception:
            continue

    # Run all trials
    for subj in subjects:
        test_cid = subj["chimeric_id"]
        test_mins = subj["minutiae"].get(1)
        test_iris = subj["iris"].get(1)
        if test_mins is None or test_iris is None or len(test_mins) == 0:
            continue

        # For each vault, test against this subject
        genuine_result = None
        impostor_accepts = 0
        n_impostors = 0

        for vault_cid, vault in vaults.items():
            try:
                if is_unimodal:
                    q_test = quantize_minutiae(test_mins, factor)
                    success = unlock_vault(q_test, vault)
                else:
                    success = unlock_fn(test_mins, test_iris["code"], test_iris["mask"], vault)
            except Exception:
                success = False

            if vault_cid == test_cid:
                genuine_result = bool(success)
            else:
                n_impostors += 1
                if success:
                    impostor_accepts += 1

        per_subject[test_cid] = {
            "genuine_success": genuine_result,
            "impostor_accepts": impostor_accepts,
            "n_impostors": n_impostors,
        }

    return per_subject


def bootstrap_ci(per_subject_data, n_bootstrap=N_BOOTSTRAP, alpha=CI_ALPHA, seed=42):
    """
    Subject-level bootstrap CIs on GAR and FAR.

    Resamples subjects with replacement. For each bootstrap sample,
    computes GAR and FAR from the resampled subjects' outcomes.

    Returns:
        dict with point estimates and 95% CIs for GAR, FAR, FRR, EER
    """
    rng = np.random.RandomState(seed)
    subject_ids = list(per_subject_data.keys())
    n = len(subject_ids)

    gar_samples = []
    far_samples = []

    for _ in range(n_bootstrap):
        # Resample subjects with replacement
        boot_ids = rng.choice(subject_ids, size=n, replace=True)

        ga_total, ga_accept = 0, 0
        ia_total, ia_accept = 0, 0

        for sid in boot_ids:
            d = per_subject_data[sid]
            if d["genuine_success"] is not None:
                ga_total += 1
                ga_accept += int(d["genuine_success"])
            ia_total += d["n_impostors"]
            ia_accept += d["impostor_accepts"]

        gar = ga_accept / ga_total * 100 if ga_total > 0 else 0
        far = ia_accept / ia_total * 100 if ia_total > 0 else 0
        gar_samples.append(gar)
        far_samples.append(far)

    gar_arr = np.array(gar_samples)
    far_arr = np.array(far_samples)
    frr_arr = 100.0 - gar_arr
    eer_arr = (far_arr + frr_arr) / 2

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    # Point estimates from original data
    ga_t, ga_a, ia_t, ia_a = 0, 0, 0, 0
    for d in per_subject_data.values():
        if d["genuine_success"] is not None:
            ga_t += 1
            ga_a += int(d["genuine_success"])
        ia_t += d["n_impostors"]
        ia_a += d["impostor_accepts"]

    gar_pt = ga_a / ga_t * 100 if ga_t > 0 else 0
    far_pt = ia_a / ia_t * 100 if ia_t > 0 else 0
    frr_pt = 100 - gar_pt
    eer_pt = (far_pt + frr_pt) / 2

    return {
        "gar": gar_pt, "gar_ci_lo": np.percentile(gar_arr, lo), "gar_ci_hi": np.percentile(gar_arr, hi),
        "far": far_pt, "far_ci_lo": np.percentile(far_arr, lo), "far_ci_hi": np.percentile(far_arr, hi),
        "frr": frr_pt, "frr_ci_lo": np.percentile(frr_arr, lo), "frr_ci_hi": np.percentile(frr_arr, hi),
        "eer": eer_pt, "eer_ci_lo": np.percentile(eer_arr, lo), "eer_ci_hi": np.percentile(eer_arr, hi),
        "ga_total": ga_t, "ga_accept": ga_a, "ia_total": ia_t, "ia_accept": ia_a,
    }


def main():
    print("=" * 80)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  N_BOOTSTRAP = {N_BOOTSTRAP}, CI_ALPHA = {CI_ALPHA}")
    print(f"  Databases: {list(DB_NAMES.values())}")
    print("=" * 80)

    architectures = {
        "Unimodal": {
            "lock": None,
            "unlock": None,
            "kwargs": {},
            "is_unimodal": True,
        },
        "A": {
            "lock": lock_vault_a,
            "unlock": unlock_vault_a,
            "kwargs": {"iris_threshold": 0.44},
            "is_unimodal": False,
        },
        "B": {
            "lock": lock_vault_b,
            "unlock": unlock_vault_b,
            "kwargs": {"block_size": IBFV_BLOCK_SIZE},
            "is_unimodal": False,
        },
        "C": {
            "lock": lock_vault_c,
            "unlock": unlock_vault_c,
            "kwargs": {"block_size": IBFV_BLOCK_SIZE},
            "is_unimodal": False,
        },
        "D": {
            "lock": lock_vault_d,
            "unlock": unlock_vault_d,
            "kwargs": {"block_size": IBFV_BLOCK_SIZE},
            "is_unimodal": False,
        },
        "IBFV": {
            "lock": lock_vault_ibfv,
            "unlock": unlock_vault_ibfv,
            "kwargs": {"block_size": IBFV_BLOCK_SIZE, "n_bonus": IBFV_N_BONUS},
            "is_unimodal": False,
        },
    }

    all_results = []

    for db_name in DATABASES:
        db_label = DB_NAMES[db_name]
        subjects = load_chimeric_data(db_name, min_iris=2)
        configs = BEST_CONFIGS[db_name]

        for factor, degree in configs:
            print(f"\n  {db_label} f={factor} k={degree}:")

            for arch_name, arch in architectures.items():
                per_subj = run_trials_per_subject(
                    subjects, arch["lock"], arch["unlock"],
                    factor, degree, arch["kwargs"],
                    is_unimodal=arch["is_unimodal"],
                )

                ci = bootstrap_ci(per_subj)
                row = {
                    "database": db_label,
                    "factor": factor,
                    "degree": degree,
                    "architecture": arch_name,
                    **ci,
                }
                all_results.append(row)

                print(f"    {arch_name:10s}: GAR={ci['gar']:6.2f}% [{ci['gar_ci_lo']:6.2f},{ci['gar_ci_hi']:6.2f}]  "
                      f"FAR={ci['far']:6.4f}% [{ci['far_ci_lo']:6.4f},{ci['far_ci_hi']:6.4f}]  "
                      f"EER={ci['eer']:6.2f}% [{ci['eer_ci_lo']:6.2f},{ci['eer_ci_hi']:6.2f}]")

    # Save results
    out_path = RESULTS_DIR / "bootstrap_ci_results.csv"
    fieldnames = [
        "database", "factor", "degree", "architecture",
        "gar", "gar_ci_lo", "gar_ci_hi",
        "far", "far_ci_lo", "far_ci_hi",
        "frr", "frr_ci_lo", "frr_ci_hi",
        "eer", "eer_ci_lo", "eer_ci_hi",
        "ga_total", "ga_accept", "ia_total", "ia_accept",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\nSaved: {out_path}")

    # Print summary comparison: IBFV vs each competitor
    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISONS: IBFV vs EACH ARCHITECTURE")
    print("=" * 80)
    for db_name in DATABASES:
        db_label = DB_NAMES[db_name]
        db_results = [r for r in all_results if r["database"] == db_label]
        for factor, degree in BEST_CONFIGS[db_name]:
            cfg_results = {
                r["architecture"]: r
                for r in db_results
                if r["factor"] == factor and r["degree"] == degree
            }
            ibfv = cfg_results.get("IBFV")
            if not ibfv:
                continue
            print(f"\n  {db_label} f={factor} k={degree}:")
            for name in ["Unimodal", "A", "B", "C", "D"]:
                comp = cfg_results.get(name)
                if not comp:
                    continue
                eer_diff = comp["eer"] - ibfv["eer"]
                # Check if CIs overlap (rough non-overlap test)
                sig = "***" if ibfv["eer_ci_hi"] < comp["eer_ci_lo"] else (
                    "**" if ibfv["eer"] < comp["eer_ci_lo"] else "ns")
                print(f"    IBFV vs {name:10s}: ΔEER = {eer_diff:+.2f}%  {sig}")


if __name__ == "__main__":
    main()
