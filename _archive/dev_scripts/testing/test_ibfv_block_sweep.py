"""
IBFV block size sweep: find the sweet spot where impostor iris key recovery = 0
while genuine recovery remains high enough to help.
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
from iris_stabilizer import IrisStabilizer

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


def test_iris_stabilizer_standalone(subjects, block_size):
    """Test iris stabilizer FAR/FRR standalone for a given block size."""
    lock_imp = 0
    test_imp = 1

    stabilizer = IrisStabilizer(block_size=block_size)

    genuine_success = 0
    genuine_total = 0
    impostor_success = 0
    impostor_total = 0

    commitments = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        iris = subj["iris"].get(lock_imp)
        if iris is None:
            continue
        commitments[cid] = stabilizer.enroll(iris["code"], iris["mask"], seed=cid)

    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        test_iris = test_subj["iris"].get(test_imp)
        if test_iris is None:
            continue

        for cid, commit in commitments.items():
            result = stabilizer.recover(
                test_iris["code"], test_iris["mask"], commit, max_shift=16
            )
            if cid == test_cid:
                genuine_total += 1
                if result.success:
                    genuine_success += 1
            else:
                impostor_total += 1
                if result.success:
                    impostor_success += 1

    g_rate = genuine_success / genuine_total * 100 if genuine_total else 0
    i_rate = impostor_success / impostor_total * 100 if impostor_total else 0

    n_blocks = 49152 // block_size
    print(f"  B={block_size:5d} ({n_blocks:4d} blocks, {n_blocks:4d}-bit key): "
          f"Genuine={g_rate:6.2f}% ({genuine_success}/{genuine_total})  "
          f"Impostor={i_rate:6.4f}% ({impostor_success}/{impostor_total})")
    return g_rate, i_rate


def test_ibfv_full(subjects, factor, degree, n_bonus, block_size):
    """Run full IBFV evaluation."""
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
            factor, degree, n_bonus=n_bonus,
            block_size=block_size, seed=cid
        )
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
                ga += 1; gacc += int(success)
            else:
                ia += 1; iacc += int(success)

    far = iacc / ia * 100 if ia else 0
    frr = (ga - gacc) / ga * 100 if ga else 0
    eer = (far + frr) / 2
    return far, frr, eer, ga, gacc, ia, iacc


def main():
    print("Loading FVC2002_1 chimeric data...")
    subjects = load_chimeric_data("fvc2002_1", min_iris=2)
    print(f"Loaded {len(subjects)} subjects\n")

    # Part 1: Iris stabilizer standalone for various block sizes
    print("=" * 70)
    print("PART 1: Iris Stabilizer FAR/FRR by Block Size")
    print("=" * 70)
    block_sizes = [31, 63, 127, 255, 511, 1023]
    iris_results = {}
    for bs in block_sizes:
        g_rate, i_rate = test_iris_stabilizer_standalone(subjects, bs)
        iris_results[bs] = (g_rate, i_rate)

    # Part 2: IBFV with different block sizes and bonus counts
    # Use the "hardest" database config first: f=18, k=5 (where unimodal is imperfect)
    print("\n" + "=" * 70)
    print("PART 2: IBFV Results (FVC2002_1)")
    print("=" * 70)

    for factor, degree in [(14, 5), (18, 7)]:
        # Unimodal baseline
        print(f"\n--- f={factor}, k={degree} ---")
        t0 = time.time()
        uni_vaults = {}
        for subj in subjects:
            cid = subj["chimeric_id"]
            mins = subj["minutiae"].get(0, [])
            q = quantize_minutiae(mins, factor)
            if len(q) >= degree + 1:
                uni_vaults[cid] = create_vault(q, degree, seed=cid)

        ga, ia, gacc, iacc = 0, 0, 0, 0
        for test_subj in subjects:
            test_cid = test_subj["chimeric_id"]
            test_mins = test_subj["minutiae"].get(1, [])
            if not test_mins:
                continue
            test_q = quantize_minutiae(test_mins, factor)
            for vault_cid, vault in uni_vaults.items():
                success = unlock_vault(test_q, vault)
                if vault_cid == test_cid:
                    ga += 1; gacc += int(success)
                else:
                    ia += 1; iacc += int(success)

        uni_far = iacc / ia * 100 if ia else 0
        uni_frr = (ga - gacc) / ga * 100 if ga else 0
        print(f"  Unimodal: FAR={uni_far:.4f}% FRR={uni_frr:.4f}% "
              f"EER={(uni_far+uni_frr)/2:.4f}%  ({time.time()-t0:.1f}s)")

        # IBFV sweep (only block sizes where iris FAR=0%)
        secure_bs = [bs for bs in block_sizes if iris_results[bs][1] == 0.0]
        print(f"  Secure block sizes (impostor FAR=0%): {secure_bs}")

        for bs in secure_bs:
            g_iris_rate = iris_results[bs][0]
            for n_bonus in [2, 4, 6]:
                t0 = time.time()
                far, frr, eer, ga, gacc, ia, iacc = test_ibfv_full(
                    subjects, factor, degree, n_bonus, bs
                )
                elapsed = time.time() - t0
                delta_eer = eer - (uni_far + uni_frr) / 2
                print(f"  IBFV B={bs:4d} bonus={n_bonus}: "
                      f"FAR={far:.4f}% FRR={frr:.4f}% EER={eer:.4f}% "
                      f"(ΔEER={delta_eer:+.4f}%) IrisRecov={g_iris_rate:.1f}% "
                      f"({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
