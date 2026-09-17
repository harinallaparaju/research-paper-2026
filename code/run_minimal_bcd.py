#!/usr/bin/env python3
"""
MINIMAL Arch B/C/D: Just 1 config per DB per arch = 12 runs.
Picks the fastest config (f=16, k=5) for speed.
"""
import sys, csv, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all_remaining_experiments import (
    load_chimeric_data, quantize_minutiae, create_vault, unlock_vault,
    lock_vault_b, unlock_vault_b,
    lock_vault_c, unlock_vault_c,
    lock_vault_d, unlock_vault_d,
    DATABASES, DB_NAMES, IBFV_BLOCK_SIZE,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "paper_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Best configs per DB (from main experiments — configs that give best unimodal EER)
BEST_CONFIGS = {
    "fvc2002_1": [(22, 7), (18, 5)],
    "fvc2002_2": [(22, 7), (18, 5)],
    "fvc2002_3": [(16, 5), (18, 5)],
    "fvc2004_1": [(22, 5), (22, 7)],
}

ARCHITECTURES = [
    ("B", lock_vault_b, unlock_vault_b),
    ("C", lock_vault_c, unlock_vault_c),
    ("D", lock_vault_d, unlock_vault_d),
]

def main():
    print(f"MINIMAL B/C/D — {datetime.now():%H:%M:%S}")
    all_rows = []

    for db_name in DATABASES:
        print(f"\n  {DB_NAMES[db_name]}:")
        subjects = load_chimeric_data(db_name, min_iris=2)
        n_subj = len(subjects)

        for factor, degree in BEST_CONFIGS[db_name]:
            for arch_name, lock_fn, unlock_fn in ARCHITECTURES:
                t0 = time.time()
                vaults = {}
                for subj in subjects:
                    cid = subj["chimeric_id"]
                    mins = subj["minutiae"].get(0, [])
                    iris = subj["iris"].get(0)
                    if not mins or iris is None:
                        continue
                    try:
                        vault = lock_fn(mins, iris["code"], iris["mask"],
                                        factor, degree, block_size=IBFV_BLOCK_SIZE, seed=cid)
                        if vault is not None:
                            vaults[cid] = vault
                    except Exception:
                        pass

                ga, ia, gacc, iacc = 0, 0, 0, 0
                for ts in subjects:
                    tcid = ts["chimeric_id"]
                    tm = ts["minutiae"].get(1, [])
                    ti = ts["iris"].get(1)
                    if not tm or ti is None:
                        continue
                    for vcid, vault in vaults.items():
                        try:
                            ok = unlock_fn(tm, ti["code"], ti["mask"], vault)
                        except Exception:
                            ok = False
                        if vcid == tcid:
                            ga += 1; gacc += int(ok)
                        else:
                            ia += 1; iacc += int(ok)

                far = iacc / ia * 100 if ia else 0
                frr = (ga - gacc) / ga * 100 if ga else 0
                eer = (far + frr) / 2
                elapsed = time.time() - t0
                print(f"    f={factor:2d} k={degree:2d} Arch{arch_name}: EER={eer:8.4f}% FAR={far:8.4f}% FRR={frr:8.4f}% ({elapsed:.1f}s)")

                all_rows.append({
                    "database": db_name, "setup": 1, "architecture": arch_name,
                    "factor": factor, "degree": degree, "block_size": IBFV_BLOCK_SIZE,
                    "n_subjects": n_subj, "n_vaults": len(vaults),
                    "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
                    "far_pct": round(far, 6), "frr_pct": round(frr, 6),
                    "eer_pct": round(eer, 6),
                })

        # Print best per arch
        for arch_name, _, _ in ARCHITECTURES:
            arch_rows = [r for r in all_rows if r["database"] == db_name and r["architecture"] == arch_name]
            if arch_rows:
                best = min(arch_rows, key=lambda r: r["eer_pct"])
                print(f"    >>> Best Arch{arch_name}: EER={best['eer_pct']:.4f}% (f={best['factor']}, k={best['degree']})")

    csv_path = RESULTS_DIR / "arch_bcd_minimal.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["database","setup","architecture","factor","degree","block_size",
                      "n_subjects","n_vaults","ga","ia","gacc","iacc","far_pct","frr_pct","eer_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Saved: {csv_path}")
    print(f"  DONE: {datetime.now():%H:%M:%S}")

if __name__ == "__main__":
    main()
