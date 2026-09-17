"""Quick smoke test for IBFV on one database, one parameter combo."""

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

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"


def load_chimeric_data(fp_db="fvc2002_1", min_iris=2):
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


def main():
    print("Loading FVC2002_1 chimeric data...")
    subjects = load_chimeric_data("fvc2002_1", min_iris=2)
    print(f"Loaded {len(subjects)} subjects")

    factor, degree = 14, 5
    lock_imp = 0
    test_imp = 1

    # --- Unimodal baseline ---
    print(f"\n--- Unimodal f={factor} k={degree} ---")
    t0 = time.time()
    uni_vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        q = quantize_minutiae(mins, factor)
        if len(q) >= degree + 1:
            uni_vaults[cid] = create_vault(q, degree, seed=cid)

    ga, ia, gacc, iacc = 0, 0, 0, 0
    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        test_mins = test_subj["minutiae"].get(test_imp, [])
        if not test_mins:
            continue
        test_q = quantize_minutiae(test_mins, factor)
        for vault_cid, vault in uni_vaults.items():
            is_genuine = (vault_cid == test_cid)
            success = unlock_vault(test_q, vault)
            if is_genuine:
                ga += 1; gacc += int(success)
            else:
                ia += 1; iacc += int(success)

    far = iacc / ia * 100 if ia else 0
    frr = (ga - gacc) / ga * 100 if ga else 0
    print(f"  Unimodal: FAR={far:.4f}% FRR={frr:.4f}% EER={(far+frr)/2:.4f}%  ({time.time()-t0:.1f}s)")
    print(f"  GA={ga} IA={ia} GAcc={gacc} IAcc={iacc}")

    # --- IBFV ---
    for block_size in [255, 1023]:
        for n_bonus in [2, 4]:
            print(f"\n--- IBFV B={block_size} bonus={n_bonus} f={factor} k={degree} ---")
            t0 = time.time()

            ibfv_vaults = {}
            enroll_fails = 0
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
                    ibfv_vaults[cid] = vault
                else:
                    enroll_fails += 1

            print(f"  Enrolled: {len(ibfv_vaults)} vaults, {enroll_fails} fails")

            ga, ia, gacc, iacc = 0, 0, 0, 0
            for test_subj in subjects:
                test_cid = test_subj["chimeric_id"]
                test_mins = test_subj["minutiae"].get(test_imp, [])
                test_iris = test_subj["iris"].get(test_imp)
                if not test_mins or test_iris is None:
                    continue

                for vault_cid, vault in ibfv_vaults.items():
                    is_genuine = (vault_cid == test_cid)
                    success = unlock_vault_ibfv(
                        test_mins, test_iris["code"], test_iris["mask"], vault
                    )
                    if is_genuine:
                        ga += 1; gacc += int(success)
                    else:
                        ia += 1; iacc += int(success)

            far = iacc / ia * 100 if ia else 0
            frr = (ga - gacc) / ga * 100 if ga else 0
            elapsed = time.time() - t0
            print(f"  IBFV: FAR={far:.4f}% FRR={frr:.4f}% EER={(far+frr)/2:.4f}%  ({elapsed:.1f}s)")
            print(f"  GA={ga} IA={ia} GAcc={gacc} IAcc={iacc}")


if __name__ == "__main__":
    main()
