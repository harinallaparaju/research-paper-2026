#!/usr/bin/env python3
"""
COMPREHENSIVE SWEEP: B_s and FHD-threshold analysis.

Experiments:
  1. Arch B/C/D at block_size ∈ {31, 63, 127, 255, 511}
  2. Arch A at FHD threshold ∈ {0.25, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45}
  3. IBFV at block_size ∈ {31, 63, 127, 255, 511}
  4. Iris GAR at each block_size (needed for interpretation)

Uses 2 best factor/degree configs per DB + Setup 1 only (enrollment=imp0, test=imp1).
"""
import sys, csv, time, json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import quantize_minutiae, create_vault, unlock_vault, read_minutiae_file
from architectures.architecture_a import lock_vault_a, unlock_vault_a
from architectures.architecture_b import lock_vault_b, unlock_vault_b
from architectures.architecture_c import lock_vault_c, unlock_vault_c
from architectures.architecture_d import lock_vault_d, unlock_vault_d
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv
from iris_stabilizer import IrisStabilizer
from iris_extraction.matching import fractional_hamming_distance

# === PATHS ===
CODE_DIR = Path(__file__).resolve().parent
CHIMERIC_DIR = CODE_DIR / "chimeric_db"
RESULTS_DIR = CODE_DIR / "results" / "paper_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === CONSTANTS ===
DATABASES = ["fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"]
DB_NAMES = {"fvc2002_1": "DB1", "fvc2002_2": "DB2", "fvc2002_3": "DB3", "fvc2004_1": "DB4"}

# Best 2 configs per DB (known from main experiments)
BEST_CONFIGS = {
    "fvc2002_1": [(22, 7), (18, 5)],
    "fvc2002_2": [(22, 7), (18, 5)],
    "fvc2002_3": [(16, 5), (18, 5)],
    "fvc2004_1": [(22, 5), (22, 7)],
}

BLOCK_SIZES = [31, 63, 127, 255, 511]
FHD_THRESHOLDS = [0.25, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45]
IBFV_N_BONUS = 4


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


# =========================================================================
# EXPERIMENT 1: Iris GAR at each block_size
# =========================================================================
def experiment_iris_gar_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: IRIS GAR SWEEP ACROSS BLOCK SIZES")
    print("=" * 70)

    rows = []
    for db_name in DATABASES:
        subjects = load_chimeric_data(db_name, min_iris=2)
        for bs in BLOCK_SIZES:
            stab = IrisStabilizer(block_size=bs)
            commitments = {}
            for subj in subjects:
                cid = subj["chimeric_id"]
                iris0 = subj["iris"].get(0)
                if iris0 is None:
                    continue
                try:
                    comm = stab.enroll(iris0["code"], iris0["mask"], seed=cid)
                    commitments[cid] = comm
                except Exception:
                    pass

            # Genuine trials
            gen_ok, gen_total = 0, 0
            for subj in subjects:
                cid = subj["chimeric_id"]
                iris1 = subj["iris"].get(1)
                if iris1 is None or cid not in commitments:
                    continue
                gen_total += 1
                try:
                    result = stab.recover(iris1["code"], iris1["mask"],
                                          commitments[cid], max_shift=32)
                    if result.success:
                        gen_ok += 1
                except Exception:
                    pass

            # Impostor trials (sample: each test against 10 random other commitments)
            imp_ok, imp_total = 0, 0
            cid_list = list(commitments.keys())
            for subj in subjects:
                cid = subj["chimeric_id"]
                iris1 = subj["iris"].get(1)
                if iris1 is None:
                    continue
                for other_cid in cid_list:
                    if other_cid == cid:
                        continue
                    imp_total += 1
                    try:
                        result = stab.recover(iris1["code"], iris1["mask"],
                                              commitments[other_cid], max_shift=32)
                        if result.success:
                            imp_ok += 1
                    except Exception:
                        pass

            gar = gen_ok / gen_total * 100 if gen_total else 0
            far = imp_ok / imp_total * 100 if imp_total else 0
            n_key_bits = len(subjects[0]["iris"][0]["code"]) // bs if subjects else 0
            print(f"  {DB_NAMES[db_name]} Bs={bs:4d}: GAR={gar:6.2f}% FAR={far:6.4f}% "
                  f"key_bits={n_key_bits} ({gen_ok}/{gen_total} gen, {imp_ok}/{imp_total} imp)")
            rows.append({
                "database": db_name, "block_size": bs,
                "n_key_bits": n_key_bits,
                "gen_total": gen_total, "gen_ok": gen_ok, "gar_pct": round(gar, 4),
                "imp_total": imp_total, "imp_ok": imp_ok, "far_pct": round(far, 6),
            })

    csv_path = RESULTS_DIR / "iris_gar_bs_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")
    return rows


# =========================================================================
# EXPERIMENT 2: Arch B/C/D at all block_sizes
# =========================================================================
def experiment_bcd_bs_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ARCH B/C/D SWEEP ACROSS BLOCK SIZES")
    print("=" * 70)

    ARCHS = [
        ("B", lock_vault_b, unlock_vault_b),
        ("C", lock_vault_c, unlock_vault_c),
        ("D", lock_vault_d, unlock_vault_d),
    ]

    rows = []
    for db_name in DATABASES:
        print(f"\n  {DB_NAMES[db_name]}:")
        subjects = load_chimeric_data(db_name, min_iris=2)
        n_subj = len(subjects)

        for factor, degree in BEST_CONFIGS[db_name]:
            for bs in BLOCK_SIZES:
                for arch_name, lock_fn, unlock_fn in ARCHS:
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
                                            factor, degree, block_size=bs, seed=cid)
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
                    print(f"    f={factor:2d} k={degree:2d} Bs={bs:4d} Arch{arch_name}: "
                          f"EER={eer:8.4f}% FAR={far:8.4f}% FRR={frr:8.4f}% ({elapsed:.1f}s)")

                    rows.append({
                        "database": db_name, "setup": 1, "architecture": arch_name,
                        "factor": factor, "degree": degree, "block_size": bs,
                        "n_subjects": n_subj, "n_vaults": len(vaults),
                        "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
                        "far_pct": round(far, 6), "frr_pct": round(frr, 6),
                        "eer_pct": round(eer, 6),
                    })

    csv_path = RESULTS_DIR / "arch_bcd_bs_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")
    return rows


# =========================================================================
# EXPERIMENT 3: Arch A at all FHD thresholds
# =========================================================================
def experiment_arch_a_fhd_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: ARCH A SWEEP ACROSS FHD THRESHOLDS")
    print("=" * 70)

    rows = []
    for db_name in DATABASES:
        print(f"\n  {DB_NAMES[db_name]}:")
        subjects = load_chimeric_data(db_name, min_iris=2)
        n_subj = len(subjects)

        for factor, degree in BEST_CONFIGS[db_name]:
            for tau in FHD_THRESHOLDS:
                t0 = time.time()
                vaults = {}
                for subj in subjects:
                    cid = subj["chimeric_id"]
                    mins = subj["minutiae"].get(0, [])
                    iris = subj["iris"].get(0)
                    if not mins or iris is None:
                        continue
                    try:
                        vault = lock_vault_a(mins, iris["code"], iris["mask"],
                                             factor, degree, iris_threshold=tau, seed=cid)
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
                            ok = unlock_vault_a(tm, ti["code"], ti["mask"], vault)
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
                print(f"    f={factor:2d} k={degree:2d} tau={tau:.2f}: "
                      f"EER={eer:8.4f}% FAR={far:8.4f}% FRR={frr:8.4f}% ({elapsed:.1f}s)")

                rows.append({
                    "database": db_name, "setup": 1, "architecture": "A",
                    "factor": factor, "degree": degree, "fhd_threshold": tau,
                    "n_subjects": n_subj, "n_vaults": len(vaults),
                    "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
                    "far_pct": round(far, 6), "frr_pct": round(frr, 6),
                    "eer_pct": round(eer, 6),
                })

    csv_path = RESULTS_DIR / "arch_a_fhd_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")
    return rows


# =========================================================================
# EXPERIMENT 4: IBFV at all block_sizes  +  Unimodal baseline
# =========================================================================
def experiment_ibfv_bs_sweep():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: IBFV + UNIMODAL SWEEP ACROSS BLOCK SIZES")
    print("=" * 70)

    rows = []
    for db_name in DATABASES:
        print(f"\n  {DB_NAMES[db_name]}:")
        subjects = load_chimeric_data(db_name, min_iris=2)
        n_subj = len(subjects)

        for factor, degree in BEST_CONFIGS[db_name]:
            # --- Unimodal baseline (run once per config) ---
            t0 = time.time()
            uni_vaults = {}
            for subj in subjects:
                cid = subj["chimeric_id"]
                mins = subj["minutiae"].get(0, [])
                if not mins:
                    continue
                q = quantize_minutiae(mins, factor)
                vault = create_vault(q, degree, seed=cid)
                if vault is not None:
                    uni_vaults[cid] = vault

            ga, ia, gacc, iacc = 0, 0, 0, 0
            for ts in subjects:
                tcid = ts["chimeric_id"]
                tm = ts["minutiae"].get(1, [])
                if not tm:
                    continue
                q_test = quantize_minutiae(tm, factor)
                for vcid, vault in uni_vaults.items():
                    try:
                        ok = unlock_vault(q_test, vault)
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
            print(f"    f={factor:2d} k={degree:2d} Unimodal:    "
                  f"EER={eer:8.4f}% FAR={far:8.4f}% FRR={frr:8.4f}% ({elapsed:.1f}s)")

            rows.append({
                "database": db_name, "setup": 1, "architecture": "Unimodal",
                "factor": factor, "degree": degree, "block_size": 0,
                "n_subjects": n_subj, "n_vaults": len(uni_vaults),
                "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
                "far_pct": round(far, 6), "frr_pct": round(frr, 6),
                "eer_pct": round(eer, 6),
            })

            # --- IBFV at each block_size ---
            for bs in BLOCK_SIZES:
                t0 = time.time()
                vaults = {}
                for subj in subjects:
                    cid = subj["chimeric_id"]
                    mins = subj["minutiae"].get(0, [])
                    iris = subj["iris"].get(0)
                    if not mins or iris is None:
                        continue
                    try:
                        vault = lock_vault_ibfv(mins, iris["code"], iris["mask"],
                                                factor, degree, n_bonus=IBFV_N_BONUS,
                                                block_size=bs, seed=cid)
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
                            ok = unlock_vault_ibfv(tm, ti["code"], ti["mask"], vault)
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
                print(f"    f={factor:2d} k={degree:2d} Bs={bs:4d} IBFV:    "
                      f"EER={eer:8.4f}% FAR={far:8.4f}% FRR={frr:8.4f}% ({elapsed:.1f}s)")

                rows.append({
                    "database": db_name, "setup": 1, "architecture": "IBFV",
                    "factor": factor, "degree": degree, "block_size": bs,
                    "n_subjects": n_subj, "n_vaults": len(vaults),
                    "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
                    "far_pct": round(far, 6), "frr_pct": round(frr, 6),
                    "eer_pct": round(eer, 6),
                })

    csv_path = RESULTS_DIR / "ibfv_unimodal_bs_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")
    return rows


# =========================================================================
# MAIN
# =========================================================================
def main():
    t_start = time.time()
    print(f"{'=' * 70}")
    print(f"COMPREHENSIVE Bs / FHD SWEEP — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 70}")
    print(f"  Block sizes: {BLOCK_SIZES}")
    print(f"  FHD thresholds: {FHD_THRESHOLDS}")
    print(f"  Databases: {list(DB_NAMES.values())}")
    print(f"  Configs per DB: {[BEST_CONFIGS[d] for d in DATABASES]}")

    experiment_iris_gar_sweep()
    experiment_bcd_bs_sweep()
    experiment_arch_a_fhd_sweep()
    experiment_ibfv_bs_sweep()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ALL DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
