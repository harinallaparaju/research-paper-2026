"""
Resume Setup 2 — second resume. DB4 remaining only.
Done through f=22,k=9. Remaining: f=22 k=11, f=24-30 all degrees.
"""
import sys
import csv
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file, quantize_minutiae, create_vault, unlock_vault,
)
from architectures.architecture_a import lock_vault_a, unlock_vault_a
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv

CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"
CSV_PATH = Path(__file__).resolve().parent / "results" / "complete_experiments" / "setup2_results_20260406_005116.csv"

IBFV_BLOCK_SIZE = 63
IBFV_N_BONUS = 4
ARCH_A_THRESHOLDS = [0.32, 0.36, 0.40, 0.44]

REMAINING = [
    ("fvc2004_1", 22, [11]),
    ("fvc2004_1", 24, [5, 7, 9, 11]),
    ("fvc2004_1", 26, [5, 7, 9, 11]),
    ("fvc2004_1", 28, [5, 7, 9, 11]),
    ("fvc2004_1", 30, [5, 7, 9, 11]),
]

FIELDNAMES = [
    "database", "setup", "architecture", "factor", "degree",
    "iris_threshold", "block_size", "n_bonus",
    "n_subjects", "n_vaults",
    "ga", "ia", "gacc", "iacc",
    "far_pct", "frr_pct", "eer_pct",
]


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


def get_test_impressions(subjects):
    all_imps = set()
    for subj in subjects:
        all_imps.update(subj["minutiae"].keys())
    return sorted(imp for imp in all_imps if imp != 0)


def run_unimodal(subjects, test_imps, factor, degree):
    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        q = quantize_minutiae(subj["minutiae"].get(0, []), factor)
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
            "far": far, "frr": frr, "eer": (far + frr) / 2, "n_vaults": len(vaults)}


def run_arch_a_best(subjects, test_imps, factor, degree):
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
                  "far": far, "frr": frr, "eer": eer, "tau": tau, "n_vaults": len(vaults)}
        if best is None or eer < best["eer"]:
            best = result
    return best


def run_ibfv(subjects, test_imps, factor, degree):
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
            "far": far, "frr": frr, "eer": (far + frr) / 2, "n_vaults": len(vaults)}


def main():
    total_start = time.time()
    total_remaining = sum(len(degrees) for _, _, degrees in REMAINING)
    done = 0

    subjects = load_chimeric_data("fvc2004_1", min_iris=8)
    n_subj = len(subjects)
    test_imps = get_test_impressions(subjects)
    print(f"Loaded {n_subj} subjects, test imps: {test_imps}")
    print(f"Remaining: {total_remaining} configs")

    for db_name, factor, degrees in REMAINING:
        for degree in degrees:
            done += 1
            tag = f"  [{done}/{total_remaining}] f={factor:2d} k={degree:2d}"

            t0 = time.time()
            uni = run_unimodal(subjects, test_imps, factor, degree)
            print(f"{tag} Uni:  EER={uni['eer']:8.4f}%  ({time.time()-t0:.1f}s)")

            row_uni = {
                "database": db_name, "setup": 2, "architecture": "Unimodal",
                "factor": factor, "degree": degree,
                "iris_threshold": "", "block_size": "", "n_bonus": "",
                "n_subjects": n_subj, "n_vaults": uni["n_vaults"],
                "ga": uni["ga"], "ia": uni["ia"],
                "gacc": uni["gacc"], "iacc": uni["iacc"],
                "far_pct": round(uni["far"], 6), "frr_pct": round(uni["frr"], 6),
                "eer_pct": round(uni["eer"], 6),
            }

            t0 = time.time()
            a = run_arch_a_best(subjects, test_imps, factor, degree)
            print(f"{tag} ArchA: EER={a['eer']:8.4f}%  τ={a['tau']}  ({time.time()-t0:.1f}s)")

            row_a = {
                "database": db_name, "setup": 2, "architecture": "A",
                "factor": factor, "degree": degree,
                "iris_threshold": a["tau"], "block_size": "", "n_bonus": "",
                "n_subjects": n_subj, "n_vaults": a["n_vaults"],
                "ga": a["ga"], "ia": a["ia"],
                "gacc": a["gacc"], "iacc": a["iacc"],
                "far_pct": round(a["far"], 6), "frr_pct": round(a["frr"], 6),
                "eer_pct": round(a["eer"], 6),
            }

            t0 = time.time()
            ibfv = run_ibfv(subjects, test_imps, factor, degree)
            delta = ibfv["eer"] - uni["eer"]
            marker = " ★" if delta < -0.01 else ""
            print(f"{tag} IBFV:  EER={ibfv['eer']:8.4f}%  ΔEER={delta:+.4f}%{marker}  ({time.time()-t0:.1f}s)")

            row_ibfv = {
                "database": db_name, "setup": 2, "architecture": "IBFV",
                "factor": factor, "degree": degree,
                "iris_threshold": "", "block_size": IBFV_BLOCK_SIZE,
                "n_bonus": IBFV_N_BONUS,
                "n_subjects": n_subj, "n_vaults": ibfv["n_vaults"],
                "ga": ibfv["ga"], "ia": ibfv["ia"],
                "gacc": ibfv["gacc"], "iacc": ibfv["iacc"],
                "far_pct": round(ibfv["far"], 6), "frr_pct": round(ibfv["frr"], 6),
                "eer_pct": round(ibfv["eer"], 6),
            }

            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(row_uni)
                writer.writerow(row_a)
                writer.writerow(row_ibfv)

            sys.stdout.flush()

    total = time.time() - total_start
    final_lines = sum(1 for _ in open(CSV_PATH)) - 1
    print(f"\nRESUME COMPLETE — {total:.0f}s ({total/3600:.1f}h)")
    print(f"Total rows: {final_lines}/432")


if __name__ == "__main__":
    main()
