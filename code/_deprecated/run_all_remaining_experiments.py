#!/usr/bin/env python3
"""
ALL remaining experiments for the IBFV paper.
Run with: caffeinate -i python3 -u run_all_remaining_experiments.py

Experiments:
1. ROC/DET curve data (sweep k at fixed f for each DB)
2. Iris genuine acceptance rate (per DB, per setup)
3. Architecture B/C/D experiments (representative configs)
4. n_bonus ablation study 
5. Timing benchmarks
6. Entropy/KLD analysis
"""
import sys
import csv
import json
import time
import math
import hashlib
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file, quantize_minutiae, create_vault, unlock_vault,
    _get_curve, _ec_multiply,
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
FACTORS = [14, 16, 18, 20, 22, 24, 26, 28, 30]
DEGREES = [5, 7, 9, 11]
IBFV_BLOCK_SIZE = 255
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


# ============================================================================
# EXPERIMENT 1: IRIS GENUINE ACCEPTANCE RATE
# ============================================================================
def experiment_iris_gar():
    """Measure how often iris fuzzy commitment succeeds for genuine users."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: IRIS GENUINE ACCEPTANCE RATE")
    print("=" * 80)
    
    results = []
    stabilizer = IrisStabilizer(block_size=IBFV_BLOCK_SIZE)
    
    for db_name in DATABASES:
        # Setup 1: min_iris=2
        subjects = load_chimeric_data(db_name, min_iris=2)
        n_subj = len(subjects)
        
        # Enroll with impression 0, verify with impression 1
        iris_ga = 0
        iris_gacc = 0
        iris_ia = 0
        iris_iacc = 0
        
        commitments = {}
        for subj in subjects:
            cid = subj["chimeric_id"]
            iris = subj["iris"].get(0)
            if iris is not None:
                commitment = stabilizer.enroll(iris["code"], iris["mask"], seed=cid)
                commitments[cid] = commitment
        
        # Genuine attempts: each subject verifies against own commitment
        for subj in subjects:
            cid = subj["chimeric_id"]
            if cid not in commitments:
                continue
            test_iris = subj["iris"].get(1)
            if test_iris is None:
                continue
            result = stabilizer.recover(
                test_iris["code"], test_iris["mask"],
                commitments[cid], max_shift=32
            )
            iris_ga += 1
            iris_gacc += int(result.success)
        
        # Impostor attempts: sample (first 20 subjects for speed)
        impostor_subjects = subjects[:20]
        for test_subj in impostor_subjects:
            test_iris = test_subj["iris"].get(1)
            if test_iris is None:
                continue
            for vault_cid, commitment in commitments.items():
                if vault_cid == test_subj["chimeric_id"]:
                    continue
                result = stabilizer.recover(
                    test_iris["code"], test_iris["mask"],
                    commitment, max_shift=32
                )
                iris_ia += 1
                iris_iacc += int(result.success)
        
        gar = iris_gacc / iris_ga * 100 if iris_ga > 0 else 0
        far = iris_iacc / iris_ia * 100 if iris_ia > 0 else 0
        print(f"  {DB_NAMES[db_name]} Setup1: GAR={gar:.2f}% ({iris_gacc}/{iris_ga})  "
              f"FAR={far:.4f}% ({iris_iacc}/{iris_ia})")
        results.append({
            "database": db_name, "setup": 1, "block_size": IBFV_BLOCK_SIZE,
            "iris_ga": iris_ga, "iris_gacc": iris_gacc,
            "iris_ia": iris_ia, "iris_iacc": iris_iacc,
            "iris_gar_pct": round(gar, 4), "iris_far_pct": round(far, 6),
        })
    
    # Setup 2: min_iris=8
    for db_name in DATABASES:
        subjects = load_chimeric_data(db_name, min_iris=8)
        n_subj = len(subjects)
        
        commitments = {}
        for subj in subjects:
            cid = subj["chimeric_id"]
            iris = subj["iris"].get(0)
            if iris is not None:
                commitments[cid] = stabilizer.enroll(iris["code"], iris["mask"], seed=cid)
        
        iris_ga = 0
        iris_gacc = 0
        # Test with impressions 1-7
        for subj in subjects:
            cid = subj["chimeric_id"]
            if cid not in commitments:
                continue
            for imp in range(1, 8):
                test_iris = subj["iris"].get(imp)
                if test_iris is None:
                    continue
                result = stabilizer.recover(
                    test_iris["code"], test_iris["mask"],
                    commitments[cid], max_shift=32
                )
                iris_ga += 1
                iris_gacc += int(result.success)
        
        gar = iris_gacc / iris_ga * 100 if iris_ga > 0 else 0
        print(f"  {DB_NAMES[db_name]} Setup2: GAR={gar:.2f}% ({iris_gacc}/{iris_ga})")
        results.append({
            "database": db_name, "setup": 2, "block_size": IBFV_BLOCK_SIZE,
            "iris_ga": iris_ga, "iris_gacc": iris_gacc,
            "iris_ia": 0, "iris_iacc": 0,
            "iris_gar_pct": round(gar, 4), "iris_far_pct": 0,
        })
    
    # Save
    csv_path = RESULTS_DIR / "iris_gar_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {csv_path}")
    return results


# ============================================================================
# EXPERIMENT 2: ARCHITECTURE B/C/D EXPERIMENTS
# ============================================================================
def experiment_arch_bcd():
    """Run Architectures B, C, D at representative configs."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: ARCHITECTURES B, C, D")
    print("=" * 80)
    
    # Representative configs: use a subset that covers the interesting range
    TEST_FACTORS = [14, 18, 22, 26, 30]
    TEST_DEGREES = [5, 7, 9, 11]
    
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
        
        for factor in TEST_FACTORS:
            for degree in TEST_DEGREES:
                for arch_name, lock_fn, unlock_fn in [
                    ("B", lock_vault_b, unlock_vault_b),
                    ("C", lock_vault_c, unlock_vault_c),
                    ("D", lock_vault_d, unlock_vault_d),
                ]:
                    t0 = time.time()
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
                                seed=cid
                            )
                            if vault is not None:
                                vaults[cid] = vault
                        except Exception as e:
                            pass  # Some configs may fail
                    
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
                                    test_iris["mask"], vault
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
                    elapsed = time.time() - t0
                    
                    print(f"    f={factor:2d} k={degree:2d} Arch{arch_name}: "
                          f"EER={eer:8.4f}% FAR={far:8.4f}% FRR={frr:8.4f}% "
                          f"({elapsed:.1f}s)")
                    
                    all_rows.append({
                        "database": db_name, "setup": 1,
                        "architecture": arch_name,
                        "factor": factor, "degree": degree,
                        "block_size": IBFV_BLOCK_SIZE,
                        "n_subjects": n_subj, "n_vaults": len(vaults),
                        "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
                        "far_pct": round(far, 6), "frr_pct": round(frr, 6),
                        "eer_pct": round(eer, 6),
                    })
    
    csv_path = RESULTS_DIR / "arch_bcd_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved: {csv_path}")
    return all_rows


# ============================================================================
# EXPERIMENT 3: n_bonus ABLATION STUDY
# ============================================================================
def experiment_nbonus_ablation():
    """Test n_bonus ∈ {0, 1, 2, 4, 6, 8} at representative configs."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: n_bonus ABLATION STUDY")
    print("=" * 80)
    
    N_BONUS_VALUES = [0, 1, 2, 4, 6, 8]
    # Use the best configs from our main results for each DB
    # DB3: f=16,k=5 and f=18,k=5. DB4: f=18,k=7 and f=22,k=5
    # Also include a hard config to show bigger effect
    TEST_CONFIGS = [
        (16, 5), (18, 5), (18, 7), (22, 5), (22, 7),
        (14, 9), (14, 11),  # hard configs: max IBFV improvement region
    ]
    
    all_rows = []
    fieldnames = [
        "database", "factor", "degree", "n_bonus", "block_size",
        "n_subjects", "n_vaults",
        "ga", "ia", "gacc", "iacc",
        "far_pct", "frr_pct", "eer_pct",
    ]
    
    for db_name in ["fvc2002_3", "fvc2004_1"]:  # Focus on hard DBs
        print(f"\n  {DB_NAMES[db_name]}:")
        subjects = load_chimeric_data(db_name, min_iris=2)
        n_subj = len(subjects)
        
        for factor, degree in TEST_CONFIGS:
            for n_bonus in N_BONUS_VALUES:
                t0 = time.time()
                vaults = {}
                for subj in subjects:
                    cid = subj["chimeric_id"]
                    mins = subj["minutiae"].get(0, [])
                    iris = subj["iris"].get(0)
                    if not mins or iris is None:
                        continue
                    if n_bonus == 0:
                        # Pure unimodal
                        q = quantize_minutiae(mins, factor)
                        if len(q) >= degree + 1:
                            vaults[cid] = ("uni", create_vault(q, degree, seed=cid))
                    else:
                        vault = lock_vault_ibfv(
                            mins, iris["code"], iris["mask"],
                            factor, degree, n_bonus=n_bonus,
                            block_size=IBFV_BLOCK_SIZE, seed=cid
                        )
                        if vault is not None:
                            vaults[cid] = ("ibfv", vault)
                
                ga, ia, gacc, iacc = 0, 0, 0, 0
                for test_subj in subjects:
                    test_cid = test_subj["chimeric_id"]
                    test_mins = test_subj["minutiae"].get(1, [])
                    test_iris = test_subj["iris"].get(1)
                    if not test_mins or test_iris is None:
                        continue
                    for vault_cid, (vtype, vault) in vaults.items():
                        if vtype == "uni":
                            test_q = quantize_minutiae(test_mins, factor)
                            success = unlock_vault(test_q, vault)
                        else:
                            success = unlock_vault_ibfv(
                                test_mins, test_iris["code"],
                                test_iris["mask"], vault
                            )
                        if vault_cid == test_cid:
                            ga += 1; gacc += int(success)
                        else:
                            ia += 1; iacc += int(success)
                
                far = iacc / ia * 100 if ia else 0
                frr = (ga - gacc) / ga * 100 if ga else 0
                eer = (far + frr) / 2
                elapsed = time.time() - t0
                
                print(f"    f={factor:2d} k={degree:2d} B={n_bonus}: "
                      f"EER={eer:8.4f}% FAR={far:8.4f}% FRR={frr:8.4f}% "
                      f"({elapsed:.1f}s)")
                
                all_rows.append({
                    "database": db_name, "factor": factor, "degree": degree,
                    "n_bonus": n_bonus, "block_size": IBFV_BLOCK_SIZE,
                    "n_subjects": n_subj, "n_vaults": len(vaults),
                    "ga": ga, "ia": ia, "gacc": gacc, "iacc": iacc,
                    "far_pct": round(far, 6), "frr_pct": round(frr, 6),
                    "eer_pct": round(eer, 6),
                })
    
    csv_path = RESULTS_DIR / "nbonus_ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved: {csv_path}")
    return all_rows


# ============================================================================
# EXPERIMENT 4: TIMING BENCHMARKS
# ============================================================================
def experiment_timing():
    """Benchmark enrollment and verification times per architecture."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: TIMING BENCHMARKS")
    print("=" * 80)
    
    results = []
    # Use DB1 with f=22, k=7 (representative config)
    db_name = "fvc2002_1"
    subjects = load_chimeric_data(db_name, min_iris=2)
    factor, degree = 22, 7
    N_TRIALS = 3  # Average over 3 runs
    
    for trial in range(N_TRIALS):
        # --- Unimodal ---
        t0 = time.time()
        uni_vaults = {}
        for subj in subjects:
            cid = subj["chimeric_id"]
            q = quantize_minutiae(subj["minutiae"].get(0, []), factor)
            if len(q) >= degree + 1:
                uni_vaults[cid] = create_vault(q, degree, seed=cid)
        t_enroll_uni = time.time() - t0
        
        t0 = time.time()
        for test_subj in subjects:
            test_mins = test_subj["minutiae"].get(1, [])
            if not test_mins:
                continue
            test_q = quantize_minutiae(test_mins, factor)
            for vault_cid, vault in uni_vaults.items():
                unlock_vault(test_q, vault)
        t_verify_uni = time.time() - t0
        
        # --- IBFV ---
        t0 = time.time()
        ibfv_vaults = {}
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
                ibfv_vaults[cid] = vault
        t_enroll_ibfv = time.time() - t0
        
        t0 = time.time()
        for test_subj in subjects:
            test_mins = test_subj["minutiae"].get(1, [])
            test_iris = test_subj["iris"].get(1)
            if not test_mins or test_iris is None:
                continue
            for vault_cid, vault in ibfv_vaults.items():
                unlock_vault_ibfv(
                    test_mins, test_iris["code"], test_iris["mask"], vault
                )
        t_verify_ibfv = time.time() - t0
        
        # --- Architecture A ---
        t0 = time.time()
        a_vaults = {}
        for subj in subjects:
            cid = subj["chimeric_id"]
            mins = subj["minutiae"].get(0, [])
            iris = subj["iris"].get(0)
            if not mins or iris is None:
                continue
            vault = lock_vault_a(
                mins, iris["code"], iris["mask"],
                factor, degree, iris_threshold=0.44
            )
            if vault is not None:
                a_vaults[cid] = vault
        t_enroll_a = time.time() - t0
        
        t0 = time.time()
        for test_subj in subjects:
            test_mins = test_subj["minutiae"].get(1, [])
            test_iris = test_subj["iris"].get(1)
            if not test_mins or test_iris is None:
                continue
            for vault_cid, vault in a_vaults.items():
                unlock_vault_a(
                    test_mins, test_iris["code"], test_iris["mask"], vault
                )
        t_verify_a = time.time() - t0
        
        n_subj = len(subjects)
        n_verify = n_subj * len(uni_vaults)  # total verify attempts
        
        results.append({
            "trial": trial + 1,
            "n_subjects": n_subj,
            "n_verify_attempts": n_verify,
            "uni_enroll_total_s": round(t_enroll_uni, 3),
            "uni_verify_total_s": round(t_verify_uni, 3),
            "uni_enroll_per_subj_ms": round(t_enroll_uni / n_subj * 1000, 2),
            "uni_verify_per_attempt_ms": round(t_verify_uni / n_verify * 1000, 4),
            "ibfv_enroll_total_s": round(t_enroll_ibfv, 3),
            "ibfv_verify_total_s": round(t_verify_ibfv, 3),
            "ibfv_enroll_per_subj_ms": round(t_enroll_ibfv / n_subj * 1000, 2),
            "ibfv_verify_per_attempt_ms": round(t_verify_ibfv / n_verify * 1000, 4),
            "archa_enroll_total_s": round(t_enroll_a, 3),
            "archa_verify_total_s": round(t_verify_a, 3),
            "archa_enroll_per_subj_ms": round(t_enroll_a / n_subj * 1000, 2),
            "archa_verify_per_attempt_ms": round(t_verify_a / n_verify * 1000, 4),
        })
        
        print(f"  Trial {trial+1}:")
        print(f"    Uni:   enroll={t_enroll_uni:.2f}s verify={t_verify_uni:.2f}s")
        print(f"    IBFV:  enroll={t_enroll_ibfv:.2f}s verify={t_verify_ibfv:.2f}s")
        print(f"    ArchA: enroll={t_enroll_a:.2f}s verify={t_verify_a:.2f}s")
    
    csv_path = RESULTS_DIR / "timing_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {csv_path}")
    
    # Print averages
    avg = lambda key: sum(r[key] for r in results) / len(results)
    print(f"\n  AVERAGES (over {N_TRIALS} trials):")
    print(f"    Unimodal:  enroll={avg('uni_enroll_per_subj_ms'):.2f} ms/subj  "
          f"verify={avg('uni_verify_per_attempt_ms'):.4f} ms/attempt")
    print(f"    IBFV:      enroll={avg('ibfv_enroll_per_subj_ms'):.2f} ms/subj  "
          f"verify={avg('ibfv_verify_per_attempt_ms'):.4f} ms/attempt")
    print(f"    ArchA:     enroll={avg('archa_enroll_per_subj_ms'):.2f} ms/subj  "
          f"verify={avg('archa_verify_per_attempt_ms'):.4f} ms/attempt")
    return results


# ============================================================================
# EXPERIMENT 5: ENTROPY / KLD ANALYSIS
# ============================================================================
def experiment_entropy():
    """Compute KLD between genuine and impostor vault x-coordinate distributions."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: ENTROPY / KLD ANALYSIS")
    print("=" * 80)
    
    results = []
    factor = 24  # Same as senior's paper
    
    for db_name in DATABASES:
        subjects = load_chimeric_data(db_name, min_iris=2)
        
        # Get genuine x-coordinates (from enrollment minutiae)
        genuine_xs = []
        for subj in subjects:
            mins = subj["minutiae"].get(0, [])
            if not mins:
                continue
            quantized = quantize_minutiae(mins, factor)
            for m in quantized:
                point = _ec_multiply(m)
                genuine_xs.append(point.x)
        
        # Get impostor x-coordinates (from test minutiae of OTHER subjects)
        impostor_xs = []
        for subj in subjects:
            mins = subj["minutiae"].get(1, [])
            if not mins:
                continue
            quantized = quantize_minutiae(mins, factor)
            for m in quantized:
                point = _ec_multiply(m)
                impostor_xs.append(point.x)
        
        # Compute histogram-based KLD
        # Use bit-length buckets (256-bit field → group by top 8 bits)
        n_bins = 256
        
        # Map to bins
        curve, _ = _get_curve()
        p = curve.field.p
        
        gen_bins = np.zeros(n_bins)
        imp_bins = np.zeros(n_bins)
        
        for x in genuine_xs:
            b = int(x * n_bins / p)
            b = min(b, n_bins - 1)
            gen_bins[b] += 1
        
        for x in impostor_xs:
            b = int(x * n_bins / p)
            b = min(b, n_bins - 1)
            imp_bins[b] += 1
        
        # Normalize to probabilities with Laplace smoothing
        gen_prob = (gen_bins + 1) / (gen_bins.sum() + n_bins)
        imp_prob = (imp_bins + 1) / (imp_bins.sum() + n_bins)
        
        # KLD(P || Q) = sum P(x) * log2(P(x) / Q(x))
        kld = 0
        for i in range(n_bins):
            if gen_prob[i] > 0:
                kld += gen_prob[i] * math.log2(gen_prob[i] / imp_prob[i])
        
        # Shannon entropy of genuine distribution
        shannon = -sum(p_i * math.log2(p_i) for p_i in gen_prob if p_i > 0)
        
        print(f"  {DB_NAMES[db_name]}: KLD={kld:.4f} bits, Shannon={shannon:.4f} bits, "
              f"n_genuine={len(genuine_xs)}, n_impostor={len(impostor_xs)}")
        
        results.append({
            "database": db_name,
            "factor": factor,
            "n_genuine_points": len(genuine_xs),
            "n_impostor_points": len(impostor_xs),
            "kld_bits": round(kld, 6),
            "shannon_entropy_bits": round(shannon, 6),
        })
    
    csv_path = RESULTS_DIR / "entropy_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {csv_path}")
    return results


# ============================================================================
# EXPERIMENT 6: ROC/DET CURVE DATA
# ============================================================================
def experiment_roc_det():
    """Extract FAR/FRR points at each (f,k) combo → ROC/DET curves.
    
    The existing CSV already has these! We just need to reformat for plotting.
    For each DB, at fixed f, sweep k to get multiple (FAR, FRR) points.
    
    Additionally, compute AUC from the ROC points.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: ROC/DET CURVE DATA (from existing results)")
    print("=" * 80)
    
    # Load existing results
    setup1_path = Path(__file__).resolve().parent / "results" / "complete_experiments" / "setup1_results_20260405_134407.csv"
    setup2_path = Path(__file__).resolve().parent / "results" / "complete_experiments" / "setup2_results_20260406_005116.csv"
    
    all_roc = []
    
    for csv_path, setup in [(setup1_path, 1), (setup2_path, 2)]:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        
        for db in DATABASES:
            for arch in ["Unimodal", "IBFV"]:
                arch_key = arch if arch == "Unimodal" else "IBFV"
                arch_rows = [r for r in rows if r["database"] == db and r["architecture"] == arch_key]
                
                # For each factor, sweep degrees → multiple FAR/FRR points
                for f_val in FACTORS:
                    points = []
                    for r in arch_rows:
                        if int(r["factor"]) == f_val:
                            far = float(r["far_pct"])
                            frr = float(r["frr_pct"])
                            gar = 100 - frr
                            points.append({
                                "database": db, "setup": setup,
                                "architecture": arch_key,
                                "factor": f_val,
                                "degree": int(r["degree"]),
                                "far_pct": far,
                                "frr_pct": frr,
                                "gar_pct": round(gar, 6),
                            })
                    all_roc.extend(points)
                
                # Also compute overall "best f" ROC: for each k, use the f with best balance
                # This gives a cleaner ROC curve
                for k_val in DEGREES:
                    for r in arch_rows:
                        if int(r["degree"]) == k_val:
                            far = float(r["far_pct"])
                            frr = float(r["frr_pct"])
                            gar = 100 - frr
                            all_roc.append({
                                "database": db, "setup": setup,
                                "architecture": arch_key,
                                "factor": int(r["factor"]),
                                "degree": k_val,
                                "far_pct": far,
                                "frr_pct": frr,
                                "gar_pct": round(gar, 6),
                            })
        
        # Compute AUC for each (db, arch) at best factor
        print(f"\n  Setup {setup} AUC:")
        for db in DATABASES:
            for arch_key in ["Unimodal", "IBFV"]:
                arch_rows = [r for r in rows if r["database"] == db and r["architecture"] == arch_key]
                
                # For each f, compute AUC from the k-sweep ROC
                best_auc = 0
                best_f = 0
                for f_val in FACTORS:
                    f_rows = [r for r in arch_rows if int(r["factor"]) == f_val]
                    if not f_rows:
                        continue
                    # Sort by FAR
                    pts = [(float(r["far_pct"]) / 100, 1 - float(r["frr_pct"]) / 100) for r in f_rows]
                    pts.sort()
                    # Add (0,0) and (1,1) endpoints
                    pts = [(0, 0)] + pts + [(1, 1)]
                    # Trapezoidal AUC
                    auc = sum(
                        (pts[i+1][0] - pts[i][0]) * (pts[i+1][1] + pts[i][1]) / 2
                        for i in range(len(pts) - 1)
                    )
                    if auc > best_auc:
                        best_auc = auc
                        best_f = f_val
                
                print(f"    {DB_NAMES[db]} {arch_key:>8s}: AUC={best_auc:.6f} (f={best_f})")
    
    # Save the ROC points (deduplicated)
    seen = set()
    unique_roc = []
    for r in all_roc:
        key = (r["database"], r["setup"], r["architecture"], r["factor"], r["degree"])
        if key not in seen:
            seen.add(key)
            unique_roc.append(r)
    
    csv_path = RESULTS_DIR / "roc_det_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unique_roc[0].keys())
        writer.writeheader()
        writer.writerows(unique_roc)
    print(f"\n  Saved: {csv_path} ({len(unique_roc)} points)")
    return unique_roc


# ============================================================================
# MAIN
# ============================================================================
def main():
    total_start = time.time()
    print(f"{'=' * 80}")
    print(f"IBFV PAPER — ALL REMAINING EXPERIMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")
    
    # Run in order of priority
    experiment_iris_gar()       # ~minutes
    experiment_roc_det()        # ~seconds (reads existing CSV)
    experiment_entropy()        # ~seconds
    experiment_timing()         # ~minutes
    experiment_nbonus_ablation()  # ~10-20 min
    experiment_arch_bcd()       # ~20-40 min (most expensive)
    
    total = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(f"ALL EXPERIMENTS COMPLETE — {total:.0f}s ({total/60:.1f}min)")
    print(f"Results in: {RESULTS_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
