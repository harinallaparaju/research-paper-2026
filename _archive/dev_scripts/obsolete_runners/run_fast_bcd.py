#!/usr/bin/env python3
"""
FAST Architectures B/C/D experiment — only representative configs.
Gets the essential data for the paper table without running 240 configs.
Tests the best (f,k) per DB + a few extras to find best EER per arch.
"""
import sys
import csv
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_all_remaining_experiments import (
    load_chimeric_data, quantize_minutiae, create_vault, unlock_vault,
    lock_vault_b, unlock_vault_b,
    lock_vault_c, unlock_vault_c,
    lock_vault_d, unlock_vault_d,
    lock_vault_ibfv, unlock_vault_ibfv,
    DATABASES, DB_NAMES, IBFV_BLOCK_SIZE,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "paper_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Representative configs — covers the best unimodal configs per DB
# and a spread to find architecture minimums
TEST_CONFIGS = [
    (14, 5), (16, 5), (18, 5), (18, 7), (22, 5), (22, 7), (26, 7), (30, 5),
]

ARCHITECTURES = [
    ("B", lock_vault_b, unlock_vault_b),
    ("C", lock_vault_c, unlock_vault_c),
    ("D", lock_vault_d, unlock_vault_d),
]

def run_single(subjects, factor, degree, arch_name, lock_fn, unlock_fn):
    """Run a single (DB, f, k, Arch) config and return results dict."""
    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(0, [])
        iris = subj["iris"].get(0)
        if not mins or iris is None:
            continue
        try:
            vault = lock_fn(
                mins, iris["code"], iris["mask"],
                factor, degree,
                block_size=IBFV_BLOCK_SIZE,
                seed=cid,
            )
            if vault is not None:
                vaults[cid] = vault
        except Exception:
            pass

    ga, ia, gacc, iacc = 0, 0, 0, 0
    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        test_mins = test_subj["minutiae"].get(1, [])
        test_iris = test_subj["iris"].get(1)
        if not test_mins or test_iris is None:
            continue
        for vault_cid, vault in vaults.items():
            try:
                success = unlock_fn(
                    test_mins, test_iris["code"],
                    test_iris["mask"], vault,
                )
            except Exception:
                success = False
            if vault_cid == test_cid:
                ga += 1; gacc += int(success)
            else:
                ia += 1; iacc += int(success)

    far = iacc / ia * 100 if ia else 0
    frr = (ga - gacc) / ga * 100 if ga else 0
    eer = (far + frr) / 2
    return {
        "factor": factor, "degree": degree,
        "architecture": arch_name,
        "n_vaults": len(vaults),
        "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
        "far_pct": round(far, 6), "frr_pct": round(frr, 6),
        "eer_pct": round(eer, 6),
    }


def main():
    print("=" * 80)
    print(f"FAST Arch B/C/D — Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Configs: {len(DATABASES)} DBs × {len(TEST_CONFIGS)} configs × {len(ARCHITECTURES)} archs = {len(DATABASES)*len(TEST_CONFIGS)*len(ARCHITECTURES)}")
    print("=" * 80)

    all_rows = []
    fieldnames = [
        "database", "setup", "architecture", "factor", "degree",
        "block_size", "n_subjects", "n_vaults",
        "ga", "ia", "gacc", "iacc",
        "far_pct", "frr_pct", "eer_pct",
    ]

    for db_name in DATABASES:
        print(f"\n  {DB_NAMES[db_name]}:")
        subjects = load_chimeric_data(db_name, min_iris=2)
        n_subj = len(subjects)

        for factor, degree in TEST_CONFIGS:
            for arch_name, lock_fn, unlock_fn in ARCHITECTURES:
                t0 = time.time()
                res = run_single(subjects, factor, degree, arch_name, lock_fn, unlock_fn)
                elapsed = time.time() - t0

                print(f"    f={factor:2d} k={degree:2d} Arch{arch_name}: "
                      f"EER={res['eer_pct']:8.4f}% FAR={res['far_pct']:8.4f}% "
                      f"FRR={res['frr_pct']:8.4f}% ({elapsed:.1f}s)")

                row = {
                    "database": db_name, "setup": 1,
                    "block_size": IBFV_BLOCK_SIZE,
                    "n_subjects": n_subj,
                    **res,
                }
                all_rows.append(row)

        # Print best per arch for this DB
        for arch_name, _, _ in ARCHITECTURES:
            arch_rows = [r for r in all_rows if r["database"] == db_name and r["architecture"] == arch_name]
            if arch_rows:
                best = min(arch_rows, key=lambda r: r["eer_pct"])
                print(f"    >>> Best Arch{arch_name}: EER={best['eer_pct']:.4f}% (f={best['factor']}, k={best['degree']})")

    csv_path = RESULTS_DIR / "arch_bcd_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Saved: {csv_path}")
    print(f"\n>>> DONE: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
