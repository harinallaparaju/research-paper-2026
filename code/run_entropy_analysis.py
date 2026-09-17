#!/usr/bin/env python3
"""
Comprehensive Vault Entropy Analysis for the IBFV Paper.
Run with: caffeinate -i python3 -u run_entropy_analysis.py

Metrics computed:
1. Per-user KLD: D_KL(P_user || Q_population) averaged across users — matches Maurya et al.
2. Aggregate KLD: D_KL(P_genuine_all || Q_impostor_all) — global distribution comparison
3. Genuine vs. Chaff KLD: D_KL(P_genuine || Q_chaff) inside a real vault — attack-relevant
4. Shannon entropy H of vault x-coordinates
5. Min-entropy H_inf = -log2(max_i p_i) — conservative security bound
6. All metrics across multiple quantization factors
"""
import sys
import csv
import json
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file, quantize_minutiae, create_vault,
    _get_curve, _ec_multiply,
)

CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "paper_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATABASES = ["fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"]
DB_NAMES = {"fvc2002_1": "DB1", "fvc2002_2": "DB2", "fvc2002_3": "DB3", "fvc2004_1": "DB4"}
FACTORS = [14, 16, 18, 20, 22, 24, 26, 28, 30]
N_BINS = 256  # Histogram bins over [0, p), max Shannon = log2(256) = 8 bits


def load_chimeric_data(fp_db, min_iris, load_iris=False):
    mapping_path = CHIMERIC_DIR / f"chimeric_{fp_db}_min{min_iris}.json"
    with open(mapping_path) as f:
        mapping = json.load(f)
    subjects = []
    for entry in mapping["mapping"]:
        subj = {"chimeric_id": entry["chimeric_id"], "minutiae": {}}
        if load_iris:
            subj["iris"] = {}
        for pair in entry["pairs"]:
            imp = pair["impression"] - 1
            subj["minutiae"][imp] = read_minutiae_file(pair["fp_file"])
            if load_iris:
                iris_data = np.load(pair["iris_file"])
                subj["iris"][imp] = {"code": iris_data["code"], "mask": iris_data["mask"]}
        subjects.append(subj)
    return subjects


def histogram(x_coords, p, n_bins=N_BINS):
    """Bin EC x-coordinates into n_bins equally-spaced bins over [0, p)."""
    bins = np.zeros(n_bins)
    for x in x_coords:
        b = min(int(x * n_bins / p), n_bins - 1)
        bins[b] += 1
    return bins


def to_prob(bins, laplace=True):
    """Convert counts to probabilities with optional Laplace smoothing."""
    n_bins = len(bins)
    if laplace:
        return (bins + 1) / (bins.sum() + n_bins)
    else:
        total = bins.sum()
        if total == 0:
            return np.ones(n_bins) / n_bins
        return bins / total


def kl_divergence(p_dist, q_dist):
    """KL divergence D_KL(P || Q) in bits."""
    kld = 0.0
    for i in range(len(p_dist)):
        if p_dist[i] > 0 and q_dist[i] > 0:
            kld += p_dist[i] * math.log2(p_dist[i] / q_dist[i])
    return kld


def shannon_entropy(p_dist):
    """Shannon entropy H(P) in bits."""
    return -sum(p * math.log2(p) for p in p_dist if p > 0)


def min_entropy(p_dist):
    """Min-entropy H_inf(P) = -log2(max p_i) in bits."""
    return -math.log2(max(p_dist))


def renyi_entropy_2(p_dist):
    """Rényi entropy of order 2 (collision entropy) in bits."""
    return -math.log2(sum(p ** 2 for p in p_dist))


# ============================================================================
# EXPERIMENT 1: Per-User KLD (Maurya et al. methodology)
# ============================================================================
def experiment_per_user_kld():
    """
    Compute per-user KLD: for each user, compare their EC x-coordinate
    distribution P_user against the x-coordinate distribution Q_population
    of all OTHER users. Average across users. This matches the methodology
    described in Maurya et al. (2025).
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: PER-USER KLD (Maurya et al. methodology)")
    print("=" * 80)

    curve, _ = _get_curve()
    p = curve.field.p
    results = []

    for db_name in DATABASES:
        subjects = load_chimeric_data(db_name, min_iris=2)
        print(f"\n--- {DB_NAMES[db_name]} ({db_name}): {len(subjects)} subjects ---")

        for factor in FACTORS:
            # Precompute each subject's EC x-coordinates from impression 0
            user_xs = {}
            for subj in subjects:
                cid = subj["chimeric_id"]
                mins = subj["minutiae"].get(0, [])
                if not mins:
                    continue
                quantized = quantize_minutiae(mins, factor)
                xs = []
                for m in quantized:
                    pt = _ec_multiply(m)
                    xs.append(pt.x)
                if xs:
                    user_xs[cid] = xs

            if len(user_xs) < 2:
                continue

            # Pool all x-coordinates across all users
            all_xs = []
            for xs in user_xs.values():
                all_xs.extend(xs)

            # Per-user KLD: D_KL(P_user || Q_others)
            kld_values = []
            for cid, xs_user in user_xs.items():
                # Q_others = all x-coords except this user's
                xs_others = []
                for cid2, xs2 in user_xs.items():
                    if cid2 != cid:
                        xs_others.extend(xs2)

                bins_user = histogram(xs_user, p)
                bins_others = histogram(xs_others, p)
                prob_user = to_prob(bins_user, laplace=True)
                prob_others = to_prob(bins_others, laplace=True)
                kld = kl_divergence(prob_user, prob_others)
                kld_values.append(kld)

            mean_kld = np.mean(kld_values)
            std_kld = np.std(kld_values)
            max_kld = np.max(kld_values)
            min_kld = np.min(kld_values)

            # Also compute aggregate Shannon, min-entropy, Rényi-2 of all x-coords
            bins_all = histogram(all_xs, p)
            prob_all = to_prob(bins_all, laplace=True)
            h_shannon = shannon_entropy(prob_all)
            h_min = min_entropy(prob_all)
            h_renyi2 = renyi_entropy_2(prob_all)

            results.append({
                "database": db_name,
                "factor": factor,
                "n_subjects": len(user_xs),
                "n_total_points": len(all_xs),
                "mean_peruser_kld": round(mean_kld, 6),
                "std_peruser_kld": round(std_kld, 6),
                "max_peruser_kld": round(max_kld, 6),
                "min_peruser_kld": round(min_kld, 6),
                "shannon_entropy": round(h_shannon, 6),
                "min_entropy": round(h_min, 6),
                "renyi2_entropy": round(h_renyi2, 6),
            })

            print(f"  f={factor:2d}: per-user KLD={mean_kld:.4f}±{std_kld:.4f} bits, "
                  f"H={h_shannon:.4f}, H_inf={h_min:.4f}, H_2={h_renyi2:.4f}")

    # Write CSV
    outpath = RESULTS_DIR / "entropy_per_user_kld.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  → Saved {outpath}")
    return results


# ============================================================================
# EXPERIMENT 2: Genuine vs. Chaff Indistinguishability (Attack-Relevant)
# ============================================================================
def experiment_genuine_vs_chaff():
    """
    For each user, create a real vault and compare the x-coordinate distribution
    of genuine points vs chaff points. This measures actual vault security:
    can an adversary statistically distinguish genuine from chaff points?
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: GENUINE vs. CHAFF INDISTINGUISHABILITY")
    print("=" * 80)

    curve, _ = _get_curve()
    p = curve.field.p
    results = []

    for db_name in DATABASES:
        subjects = load_chimeric_data(db_name, min_iris=2)
        print(f"\n--- {DB_NAMES[db_name]} ({db_name}): {len(subjects)} subjects ---")

        for factor in FACTORS:
            for degree in [5, 7, 9, 11]:
                kld_values = []
                h_vault_values = []
                h_min_vault_values = []
                n_gen_list = []
                n_chaff_list = []

                for subj in subjects:
                    mins = subj["minutiae"].get(0, [])
                    if not mins:
                        continue
                    quantized = quantize_minutiae(mins, factor)
                    if len(quantized) < degree + 1:
                        continue

                    # Create a real vault
                    vault = create_vault(quantized, degree, chaff_multiplier=5,
                                         seed=subj["chimeric_id"])

                    # Separate genuine and chaff x-coordinates
                    # Genuine points are the first n_genuine in the pre-shuffle list,
                    # but the vault shuffles them. We need to identify genuine vs chaff.
                    # Reconstruct: genuine x-coords are the EC mappings of quantized minutiae
                    genuine_xs = set()
                    for m in quantized:
                        pt = _ec_multiply(m)
                        genuine_xs.add(pt.x)

                    vault_genuine_xs = []
                    vault_chaff_xs = []
                    for (vx, vy) in vault.vault_points:
                        if vx in genuine_xs:
                            vault_genuine_xs.append(vx)
                        else:
                            vault_chaff_xs.append(vx)

                    if not vault_genuine_xs or not vault_chaff_xs:
                        continue

                    n_gen_list.append(len(vault_genuine_xs))
                    n_chaff_list.append(len(vault_chaff_xs))

                    # KLD between genuine and chaff x-coordinate distributions
                    bins_g = histogram(vault_genuine_xs, p)
                    bins_c = histogram(vault_chaff_xs, p)
                    prob_g = to_prob(bins_g, laplace=True)
                    prob_c = to_prob(bins_c, laplace=True)
                    kld = kl_divergence(prob_g, prob_c)
                    kld_values.append(kld)

                    # Entropy of the entire vault x-coordinate distribution
                    all_vault_xs = [vx for (vx, _) in vault.vault_points]
                    bins_v = histogram(all_vault_xs, p)
                    prob_v = to_prob(bins_v, laplace=True)
                    h_vault_values.append(shannon_entropy(prob_v))
                    h_min_vault_values.append(min_entropy(prob_v))

                if not kld_values:
                    continue

                results.append({
                    "database": db_name,
                    "factor": factor,
                    "degree": degree,
                    "n_subjects": len(kld_values),
                    "mean_n_genuine": round(np.mean(n_gen_list), 1),
                    "mean_n_chaff": round(np.mean(n_chaff_list), 1),
                    "mean_kld_genuine_chaff": round(np.mean(kld_values), 6),
                    "std_kld_genuine_chaff": round(np.std(kld_values), 6),
                    "max_kld_genuine_chaff": round(np.max(kld_values), 6),
                    "mean_vault_shannon": round(np.mean(h_vault_values), 6),
                    "mean_vault_min_entropy": round(np.mean(h_min_vault_values), 6),
                })

                if degree == 7:  # Print representative
                    print(f"  f={factor:2d}, k={degree}: "
                          f"KLD(gen||chaff)={np.mean(kld_values):.4f}±{np.std(kld_values):.4f}, "
                          f"H_vault={np.mean(h_vault_values):.4f}, "
                          f"H_inf={np.mean(h_min_vault_values):.4f}, "
                          f"gen/chaff={np.mean(n_gen_list):.0f}/{np.mean(n_chaff_list):.0f}")

    outpath = RESULTS_DIR / "entropy_genuine_vs_chaff.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  → Saved {outpath}")
    return results


# ============================================================================
# EXPERIMENT 3: IBFV vs Unimodal Vault Entropy Comparison
# ============================================================================
def experiment_ibfv_entropy():
    """
    Compare vault entropy between unimodal vaults and IBFV vaults.
    IBFV adds K bonus points — does this affect the entropy profile?
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: IBFV vs UNIMODAL VAULT ENTROPY")
    print("=" * 80)

    from architectures.architecture_ibfv import lock_vault_ibfv

    curve, _ = _get_curve()
    p = curve.field.p
    results = []
    factor = 24  # Representative
    degree = 7

    for db_name in DATABASES:
        subjects = load_chimeric_data(db_name, min_iris=2, load_iris=True)
        print(f"\n--- {DB_NAMES[db_name]} ({db_name}) ---")

        uni_kld_list = []
        ibfv_kld_list = []
        uni_h_list = []
        ibfv_h_list = []
        uni_hmin_list = []
        ibfv_hmin_list = []

        for subj in subjects:
            mins = subj["minutiae"].get(0, [])
            iris = subj.get("iris", {}).get(0)
            if not mins:
                continue
            quantized = quantize_minutiae(mins, factor)
            if len(quantized) < degree + 1:
                continue

            # --- Unimodal vault ---
            vault_uni = create_vault(quantized, degree, chaff_multiplier=5,
                                     seed=subj["chimeric_id"])
            genuine_xs_uni = set()
            for m in quantized:
                genuine_xs_uni.add(_ec_multiply(m).x)

            gen_x_uni = [vx for (vx, _) in vault_uni.vault_points if vx in genuine_xs_uni]
            chaff_x_uni = [vx for (vx, _) in vault_uni.vault_points if vx not in genuine_xs_uni]

            if gen_x_uni and chaff_x_uni:
                prob_g = to_prob(histogram(gen_x_uni, p), laplace=True)
                prob_c = to_prob(histogram(chaff_x_uni, p), laplace=True)
                uni_kld_list.append(kl_divergence(prob_g, prob_c))

                all_x = [vx for (vx, _) in vault_uni.vault_points]
                prob_v = to_prob(histogram(all_x, p), laplace=True)
                uni_h_list.append(shannon_entropy(prob_v))
                uni_hmin_list.append(min_entropy(prob_v))

            # --- IBFV vault ---
            iris_data = subj.get("iris")
            iris_imp0 = iris_data.get(0) if iris_data else None
            if iris_imp0 is not None:
                ibfv_vault = lock_vault_ibfv(
                    mins, iris_imp0["code"], iris_imp0["mask"],
                    factor, degree, n_bonus=4, chaff_multiplier=5,
                    block_size=255, seed=subj["chimeric_id"]
                )
                if ibfv_vault is not None:
                    # IBFV vault: we know n_genuine_fp + n_bonus genuine,
                    # rest are chaff. Identify FP genuine x-coords, then
                    # remaining genuine points must be the bonus ones.
                    genuine_fp_xs = set()
                    for m in quantized:
                        genuine_fp_xs.add(_ec_multiply(m).x)

                    # All vault x-coordinates
                    all_x_ibfv = [vx for (vx, _) in ibfv_vault.vault_points]

                    # For entropy of the whole vault distribution
                    prob_v = to_prob(histogram(all_x_ibfv, p), laplace=True)
                    ibfv_h_list.append(shannon_entropy(prob_v))
                    ibfv_hmin_list.append(min_entropy(prob_v))

                    # Genuine vs chaff: we know FP genuine x-coords.
                    # Bonus x-coords are unknown without the iris key,
                    # so from an attacker's perspective, treat only FP
                    # genuine as identifiable. For KLD: compare distribution
                    # of known-genuine FP x-coords vs all other vault points.
                    gen_x = [vx for (vx, _) in ibfv_vault.vault_points
                             if vx in genuine_fp_xs]
                    other_x = [vx for (vx, _) in ibfv_vault.vault_points
                               if vx not in genuine_fp_xs]
                    if gen_x and other_x:
                        prob_g = to_prob(histogram(gen_x, p), laplace=True)
                        prob_o = to_prob(histogram(other_x, p), laplace=True)
                        ibfv_kld_list.append(kl_divergence(prob_g, prob_o))

        results.append({
            "database": db_name,
            "factor": factor,
            "degree": degree,
            "uni_mean_kld_gc": round(np.mean(uni_kld_list), 6) if uni_kld_list else None,
            "uni_mean_shannon": round(np.mean(uni_h_list), 6) if uni_h_list else None,
            "uni_mean_min_entropy": round(np.mean(uni_hmin_list), 6) if uni_hmin_list else None,
            "ibfv_mean_kld_gc": round(np.mean(ibfv_kld_list), 6) if ibfv_kld_list else None,
            "ibfv_mean_shannon": round(np.mean(ibfv_h_list), 6) if ibfv_h_list else None,
            "ibfv_mean_min_entropy": round(np.mean(ibfv_hmin_list), 6) if ibfv_hmin_list else None,
            "n_uni_vaults": len(uni_kld_list),
            "n_ibfv_vaults": len(ibfv_kld_list),
        })

        print(f"  Unimodal: KLD(g||c)={np.mean(uni_kld_list):.4f}, "
              f"H={np.mean(uni_h_list):.4f}, H_inf={np.mean(uni_hmin_list):.4f} "
              f"({len(uni_kld_list)} vaults)")
        if ibfv_kld_list:
            print(f"  IBFV:     KLD(g||c)={np.mean(ibfv_kld_list):.4f}, "
                  f"H={np.mean(ibfv_h_list):.4f}, H_inf={np.mean(ibfv_hmin_list):.4f} "
                  f"({len(ibfv_kld_list)} vaults)")

    outpath = RESULTS_DIR / "entropy_ibfv_comparison.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  → Saved {outpath}")
    return results


# ============================================================================
# EXPERIMENT 4: Quantization Collision Analysis
# ============================================================================
def experiment_quantization_collisions():
    """
    Measure how many minutiae map to the same quantized value as a function
    of factor f. High collision rates mean information loss.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: QUANTIZATION COLLISION ANALYSIS")
    print("=" * 80)

    results = []

    for db_name in DATABASES:
        subjects = load_chimeric_data(db_name, min_iris=2)
        print(f"\n--- {DB_NAMES[db_name]} ({db_name}) ---")

        for factor in FACTORS:
            raw_counts = []
            quant_counts = []
            collision_rates = []

            for subj in subjects:
                mins = subj["minutiae"].get(0, [])
                if not mins:
                    continue
                n_raw = len(mins)
                quantized = quantize_minutiae(mins, factor)
                n_quant = len(quantized)
                raw_counts.append(n_raw)
                quant_counts.append(n_quant)
                collision_rates.append(1.0 - n_quant / n_raw if n_raw > 0 else 0.0)

            results.append({
                "database": db_name,
                "factor": factor,
                "mean_raw_minutiae": round(np.mean(raw_counts), 1),
                "mean_unique_quantized": round(np.mean(quant_counts), 1),
                "mean_collision_rate": round(np.mean(collision_rates), 4),
                "std_collision_rate": round(np.std(collision_rates), 4),
            })

            print(f"  f={factor:2d}: raw={np.mean(raw_counts):.1f} → "
                  f"unique={np.mean(quant_counts):.1f} "
                  f"(collision={np.mean(collision_rates)*100:.1f}%)")

    outpath = RESULTS_DIR / "quantization_collisions.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  → Saved {outpath}")
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("COMPREHENSIVE VAULT ENTROPY ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    t0 = time.time()

    # Run all experiments
    r1 = experiment_per_user_kld()
    r2 = experiment_genuine_vs_chaff()
    r3 = experiment_ibfv_entropy()
    r4 = experiment_quantization_collisions()

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"ALL ENTROPY EXPERIMENTS COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 80}")

    # Print summary for paper
    print("\n\n=== SUMMARY FOR PAPER ===\n")
    print("Per-User KLD (f=24, comparable to Maurya et al.):")
    for r in r1:
        if r["factor"] == 24:
            print(f"  {DB_NAMES[r['database']]}: D_KL = {r['mean_peruser_kld']:.4f} ± {r['std_peruser_kld']:.4f} bits")

    print("\nGenuine vs Chaff KLD (f=24, k=7, per-vault average):")
    for r in r2:
        if r["factor"] == 24 and r["degree"] == 7:
            print(f"  {DB_NAMES[r['database']]}: D_KL(gen||chaff) = {r['mean_kld_genuine_chaff']:.4f} ± {r['std_kld_genuine_chaff']:.4f} bits")

    print("\nIBFV vs Unimodal Vault Entropy (f=24, k=7):")
    for r in r3:
        print(f"  {DB_NAMES[r['database']]}: Uni H={r['uni_mean_shannon']}, "
              f"IBFV H={r['ibfv_mean_shannon']}, "
              f"Uni H_inf={r['uni_mean_min_entropy']}, "
              f"IBFV H_inf={r['ibfv_mean_min_entropy']}")


if __name__ == "__main__":
    import time
    main()
