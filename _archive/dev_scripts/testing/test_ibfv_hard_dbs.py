"""
IBFV on hard databases (DB3, DB4) where unimodal has non-zero EER.
This is where IBFV should show real improvement.

Block sizes: B=63 (secure, 54% genuine iris recovery),
             B=31 (most secure, 27% genuine iris recovery)
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file, quantize_minutiae, create_vault, unlock_vault,
)
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv
from architectures.architecture_a import lock_vault_a, unlock_vault_a

CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"


def load_chimeric_data(fp_db, min_iris=2):
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
    lock_imp, test_imp = 0, 1
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
    return far, frr, (far + frr) / 2, ga, gacc, ia, iacc


def run_arch_a(subjects, factor, degree, tau):
    lock_imp, test_imp = 0, 1
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
            success = unlock_vault_a(test_mins, test_iris["code"],
                                     test_iris["mask"], vault)
            if vault_cid == test_cid:
                ga += 1; gacc += int(success)
            else:
                ia += 1; iacc += int(success)

    far = iacc / ia * 100 if ia else 0
    frr = (ga - gacc) / ga * 100 if ga else 0
    return far, frr, (far + frr) / 2


def run_ibfv(subjects, factor, degree, n_bonus, block_size):
    lock_imp, test_imp = 0, 1
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
    # Track iris recovery for genuine
    iris_ok_genuine = 0
    iris_fail_genuine = 0

    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
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
    return far, frr, (far + frr) / 2, ga, gacc, ia, iacc


def main():
    databases = [
        ("fvc2002_1", [(14, 5), (24, 10)]),
        ("fvc2002_2", [(14, 5)]),
        ("fvc2002_3", [(14, 5), (18, 5)]),
        ("fvc2004_1", [(14, 5), (18, 5)]),
    ]

    all_summaries = []

    for db_name, param_combos in databases:
        print(f"\n{'='*70}")
        print(f"DATABASE: {db_name}")
        print(f"{'='*70}")

        subjects = load_chimeric_data(db_name, min_iris=2)
        print(f"Loaded {len(subjects)} chimeric subjects")

        for factor, degree in param_combos:
            print(f"\n  --- f={factor}, k={degree} ---")

            # Unimodal
            t0 = time.time()
            uni_far, uni_frr, uni_eer, ga, gacc, ia, iacc = run_unimodal(
                subjects, factor, degree
            )
            print(f"  Unimodal:     FAR={uni_far:.4f}%  FRR={uni_frr:.4f}%  "
                  f"EER={uni_eer:.4f}%  GA={ga} GAcc={gacc}  ({time.time()-t0:.1f}s)")

            # Best Architecture A
            best_a_eer = 999
            best_a_tau = 0
            for tau in [0.32, 0.36, 0.40, 0.44, 0.46]:
                t0 = time.time()
                a_far, a_frr, a_eer = run_arch_a(subjects, factor, degree, tau)
                elapsed = time.time() - t0
                print(f"  Arch A τ={tau:.2f}: FAR={a_far:.4f}%  FRR={a_frr:.4f}%  "
                      f"EER={a_eer:.4f}%  ({elapsed:.1f}s)")
                if a_eer < best_a_eer:
                    best_a_eer = a_eer
                    best_a_tau = tau

            # IBFV with secure block sizes
            best_ibfv_eer = 999
            best_ibfv_cfg = ""
            for bs in [63]:  # B=63 is the sweet spot
                for n_bonus in [2, 4, 6, 8]:
                    t0 = time.time()
                    far, frr, eer, ga, gacc, ia, iacc = run_ibfv(
                        subjects, factor, degree, n_bonus, bs
                    )
                    elapsed = time.time() - t0
                    delta = eer - uni_eer
                    print(f"  IBFV B={bs} bonus={n_bonus}: "
                          f"FAR={far:.4f}%  FRR={frr:.4f}%  EER={eer:.4f}%  "
                          f"ΔEER={delta:+.4f}%  GA={ga} GAcc={gacc}  ({elapsed:.1f}s)")
                    if eer < best_ibfv_eer:
                        best_ibfv_eer = eer
                        best_ibfv_cfg = f"B={bs},bonus={n_bonus}"

            summary = (f"{db_name} f={factor} k={degree}: "
                       f"Unimodal={uni_eer:.4f}% | "
                       f"BestA={best_a_eer:.4f}% (τ={best_a_tau}) | "
                       f"BestIBFV={best_ibfv_eer:.4f}% ({best_ibfv_cfg})")
            all_summaries.append(summary)
            print(f"\n  >>> {summary}")

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for s in all_summaries:
        print(f"  {s}")


if __name__ == "__main__":
    main()
