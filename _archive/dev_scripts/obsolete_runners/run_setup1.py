"""
Setup 1 experiments — comprehensive parameter sweep.
Lock=impression 1, Test=impression 2, min_iris=2.
All 4 databases, 9 factors, 4 degrees.
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
FACTORS = [14, 16, 18, 20, 22, 24, 26, 28, 30]
DEGREES = [5, 7, 9, 11]

IBFV_BLOCK_SIZE = 63
IBFV_N_BONUS = 4
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


def run_unimodal(subjects, factor, degree):
    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        q = quantize_minutiae(subj["minutiae"].get(0, []), factor)
        if len(q) >= degree + 1:
            vaults[cid] = create_vault(q, degree, seed=cid)

    ga, ia, gacc, iacc = 0, 0, 0, 0
    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        test_mins = test_subj["minutiae"].get(1, [])
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
            "far": far, "frr": frr, "eer": (far + frr) / 2, "n_vaults": len(vaults)}


def run_arch_a_best(subjects, factor, degree):
    best = None
    for tau in ARCH_A_THRESHOLDS:
        vaults = {}
        for subj in subjects:
            cid = subj["chimeric_id"]
            mins = subj["minutiae"].get(0, [])
            iris = subj["iris"].get(0)
            if not mins or iris is None:
                continue
            vault = lock_vault_a(mins, iris["code"], iris["mask"],
                                 factor, degree, iris_threshold=tau)
            if vault is not None:
                vaults[cid] = vault

        ga, ia, gacc, iacc = 0, 0, 0, 0
        for test_subj in subjects:
            test_cid = test_subj["chimeric_id"]
            test_mins = test_subj["minutiae"].get(1, [])
            test_iris = test_subj["iris"].get(1)
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
                  "far": far, "frr": frr, "eer": eer, "tau": tau, "n_vaults": len(vaults)}
        if best is None or eer < best["eer"]:
            best = result
    return best


def run_ibfv(subjects, factor, degree):
    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(0, [])
        iris = subj["iris"].get(0)
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
        test_mins = test_subj["minutiae"].get(1, [])
        test_iris = test_subj["iris"].get(1)
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
            "far": far, "frr": frr, "eer": (far + frr) / 2, "n_vaults": len(vaults)}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"setup1_results_{timestamp}.csv"

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
        print(f"\n{'='*80}")
        print(f"DATABASE: {db_name} | SETUP 1")
        print(f"{'='*80}")

        subjects = load_chimeric_data(db_name, min_iris=2)
        n_subj = len(subjects)
        print(f"Loaded {n_subj} subjects")

        for factor in FACTORS:
            for degree in DEGREES:
                tag = f"  f={factor:2d} k={degree:2d}"

                # Unimodal
                t0 = time.time()
                uni = run_unimodal(subjects, factor, degree)
                print(f"{tag} Uni:  EER={uni['eer']:8.4f}%  "
                      f"FAR={uni['far']:8.4f}% FRR={uni['frr']:8.4f}%  "
                      f"GA={uni['ga']} IA={uni['ia']}  ({time.time()-t0:.1f}s)")

                all_rows.append({
                    "database": db_name, "setup": 1,
                    "architecture": "Unimodal",
                    "factor": factor, "degree": degree,
                    "iris_threshold": "", "block_size": "", "n_bonus": "",
                    "n_subjects": n_subj, "n_vaults": uni["n_vaults"],
                    "ga": uni["ga"], "ia": uni["ia"],
                    "gacc": uni["gacc"], "iacc": uni["iacc"],
                    "far_pct": round(uni["far"], 6), "frr_pct": round(uni["frr"], 6),
                    "eer_pct": round(uni["eer"], 6),
                })

                # Architecture A (best τ)
                t0 = time.time()
                a = run_arch_a_best(subjects, factor, degree)
                print(f"{tag} ArchA: EER={a['eer']:8.4f}%  "
                      f"FAR={a['far']:8.4f}% FRR={a['frr']:8.4f}%  "
                      f"τ={a['tau']}  ({time.time()-t0:.1f}s)")

                all_rows.append({
                    "database": db_name, "setup": 1,
                    "architecture": "A",
                    "factor": factor, "degree": degree,
                    "iris_threshold": a["tau"], "block_size": "", "n_bonus": "",
                    "n_subjects": n_subj, "n_vaults": a["n_vaults"],
                    "ga": a["ga"], "ia": a["ia"],
                    "gacc": a["gacc"], "iacc": a["iacc"],
                    "far_pct": round(a["far"], 6), "frr_pct": round(a["frr"], 6),
                    "eer_pct": round(a["eer"], 6),
                })

                # IBFV
                t0 = time.time()
                ibfv = run_ibfv(subjects, factor, degree)
                delta = ibfv["eer"] - uni["eer"]
                marker = " ★" if delta < -0.01 else ""
                print(f"{tag} IBFV:  EER={ibfv['eer']:8.4f}%  "
                      f"FAR={ibfv['far']:8.4f}% FRR={ibfv['frr']:8.4f}%  "
                      f"ΔEER={delta:+.4f}%{marker}  ({time.time()-t0:.1f}s)")

                all_rows.append({
                    "database": db_name, "setup": 1,
                    "architecture": "IBFV",
                    "factor": factor, "degree": degree,
                    "iris_threshold": "", "block_size": IBFV_BLOCK_SIZE,
                    "n_bonus": IBFV_N_BONUS,
                    "n_subjects": n_subj, "n_vaults": ibfv["n_vaults"],
                    "ga": ibfv["ga"], "ia": ibfv["ia"],
                    "gacc": ibfv["gacc"], "iacc": ibfv["iacc"],
                    "far_pct": round(ibfv["far"], 6), "frr_pct": round(ibfv["frr"], 6),
                    "eer_pct": round(ibfv["eer"], 6),
                })

                # Save incrementally
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_rows)

    total = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"SETUP 1 COMPLETE — {total:.0f}s ({total/3600:.1f}h)")
    print(f"Results: {csv_path}")
    print(f"{'='*80}")

    # Print summary
    print(f"\nBEST EER per database:")
    for db in DATABASES:
        uni_rows = [r for r in all_rows if r["database"] == db and r["architecture"] == "Unimodal"]
        ibfv_rows = [r for r in all_rows if r["database"] == db and r["architecture"] == "IBFV"]
        if not uni_rows:
            continue
        bu = min(uni_rows, key=lambda r: r["eer_pct"])
        bi = min(ibfv_rows, key=lambda r: r["eer_pct"]) if ibfv_rows else None
        print(f"  {db}: Uni={bu['eer_pct']:.4f}% (f={bu['factor']},k={bu['degree']}) | "
              f"IBFV={bi['eer_pct']:.4f}% (f={bi['factor']},k={bi['degree']}) | "
              f"Δ={bi['eer_pct']-bu['eer_pct']:+.4f}%" if bi else "")


if __name__ == "__main__":
    main()
