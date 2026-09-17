#!/usr/bin/env python3
"""
Formal D_sys unlinkability analysis following:

  Gomez-Barrero, Galbally, Rathgeb, Busch,
  "General Framework to Evaluate Unlinkability in Biometric Template
   Protection Systems", IEEE TIFS, vol. 13, no. 6, pp. 1406-1420, 2018.

Methodology:
  1. For each subject, create TWO vaults with DIFFERENT application-specific
     parameters (different seed / polynomial / chaff).
  2. Compute a cross-vault "linkage score" = |intersection of x-coordinate sets|
     (the Jaccard coefficient is also computed but raw count is more standard).
  3. Mated scores: both vaults from the same subject.
     Non-mated scores: vaults from different subjects.
  4. Compute D_sys from the normalized histogram of mated vs non-mated scores.
  5. D_sys ≈ 0 → fully unlinkable; D_sys ≈ 1 → fully linkable.

We evaluate under TWO scenarios:
  (a) Same quantization parameters (f1 = f2) — worst case for unlinkability.
  (b) Different quantization parameters (f1 ≠ f2) — recommended deployment.
"""
import sys
import json
import csv
import time
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import quantize_minutiae, create_vault, read_minutiae_file
from architectures.architecture_ibfv import lock_vault_ibfv

# ============================================================================
# PATHS
# ============================================================================
CODE_DIR   = Path(__file__).resolve().parent
CHIMERIC   = CODE_DIR / "chimeric_db"
RESULTS    = CODE_DIR / "results" / "paper_experiments"
FIG_DIR    = CODE_DIR.parent / "paper_ieee" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONSTANTS
# ============================================================================
DATABASES  = ["fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"]
DB_NAMES   = {"fvc2002_1": "DB1", "fvc2002_2": "DB2",
              "fvc2002_3": "DB3", "fvc2004_1": "DB4"}

# Best config per DB
BEST_CONFIGS = {
    "fvc2002_1": (22, 7),
    "fvc2002_2": (22, 7),
    "fvc2002_3": (16, 5),
    "fvc2004_1": (22, 5),
}

N_BONUS     = 4
BLOCK_SIZE  = 255
OMEGA       = 1.0   # weighting parameter (standard = 1)
N_BINS      = 100   # histogram bins


# ============================================================================
# LOAD DATA
# ============================================================================
def load_chimeric_data(fp_db, min_iris=2):
    mapping_path = CHIMERIC / f"chimeric_{fp_db}_min{min_iris}.json"
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
# EXTRACT X-COORDINATES FROM VAULT
# ============================================================================
def extract_x_coords(vault):
    """Extract the set of x-coordinates from a vault (VaultData or MultimodalVaultIBFV).
    Both dataclasses store vault_points as List[Tuple[int, int]]."""
    if vault is None:
        return set()
    return {x for x, y in vault.vault_points}


# ============================================================================
# LINKAGE SCORE: raw intersection count
# ============================================================================
def linkage_score(x_set_1, x_set_2):
    """
    Compute the linkage score between two vaults.
    Returns the number of common x-coordinates.
    """
    return len(x_set_1 & x_set_2)


# ============================================================================
# D_sys COMPUTATION (Gomez-Barrero et al. 2018)
# ============================================================================
def compute_dsys(mated_scores, nonmated_scores, omega=1.0, n_bins=100):
    """
    Compute the D^sys_↔ unlinkability metric.

    D_sys = 0 → fully unlinkable (mated and non-mated indistinguishable)
    D_sys = 1 → fully linkable (perfectly separable)

    Parameters
    ----------
    mated_scores    : array of linkage scores from same-user vault pairs
    nonmated_scores : array of linkage scores from different-user vault pairs
    omega           : prior odds ratio (default 1.0 = equal priors)
    n_bins          : number of histogram bins

    Returns
    -------
    dict with D_sys, bin_centers, D_local, y_mated, y_nonmated
    """
    mated    = np.array(mated_scores, dtype=float)
    nonmated = np.array(nonmated_scores, dtype=float)

    # Bin edges spanning both distributions
    lo = min(mated.min(), nonmated.min())
    hi = max(mated.max(), nonmated.max())
    bin_edges   = np.linspace(lo, hi, n_bins + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    # Normalized histograms (probability densities)
    y_m, _ = np.histogram(mated,    bins=bin_edges, density=True)
    y_n, _ = np.histogram(nonmated, bins=bin_edges, density=True)

    # Likelihood ratio and local D
    LR = np.divide(y_m, y_n, out=np.ones_like(y_m), where=y_n != 0)
    D_local = 2.0 * (omega * LR / (1.0 + omega * LR)) - 1.0
    D_local[omega * LR <= 1.0] = 0.0
    D_local[y_n == 0] = 1.0  # definition: if non-mated has 0 density, D=1

    # Integrate: D_sys = ∫ D(s) · p_m(s) ds
    D_sys = float(np.trapezoid(y=D_local * y_m, x=bin_centers))

    return {
        "D_sys": D_sys,
        "bin_centers": bin_centers,
        "D_local": D_local,
        "y_mated": y_m,
        "y_nonmated": y_n,
        "bin_edges": bin_edges,
    }


# ============================================================================
# GENERATE UNLINKABILITY FIGURE
# ============================================================================
def plot_unlinkability(result, title, out_path):
    """Generate the standard D_sys figure (score distributions + local D)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

    bc = result["bin_centers"]
    bw = bc[1] - bc[0] if len(bc) > 1 else 1.0

    # Score distributions as step plots
    ax1.step(bc, result["y_mated"], where="mid", color="#4CAF50",
             label="Mated", linewidth=1.5)
    ax1.step(bc, result["y_nonmated"], where="mid", color="#F44336",
             label="Non-mated", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("Probability Density")
    ax1.set_xlabel("Linkage Score (# common x-coords)")

    # Local D on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(bc, result["D_local"], color="#2196F3", linewidth=2.0,
             label=r"$D_{\leftrightarrow}(s)$")
    ax2.set_ylabel(r"$D_{\leftrightarrow}(s)$")
    ax2.set_ylim([0, 1.1])

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=7, framealpha=0.9)

    ax1.set_title(f"{title}, " + r"$D_{\leftrightarrow}^{sys}$"
                  + f" = {result['D_sys']:.3f}", fontsize=8)
    ax1.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"    Saved figure: {out_path.name}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_unlinkability_analysis():
    print("=" * 70)
    print("FORMAL D_sys UNLINKABILITY ANALYSIS")
    print("  Gomez-Barrero et al., IEEE TIFS 2018")
    print("=" * 70)

    all_results = []

    for db_name in DATABASES:
        db_label = DB_NAMES[db_name]
        factor, degree = BEST_CONFIGS[db_name]
        print(f"\n{'='*60}")
        print(f"  {db_label} (f={factor}, k={degree})")
        print(f"{'='*60}")

        subjects = load_chimeric_data(db_name, min_iris=2)
        n_subj = len(subjects)
        print(f"  Loaded {n_subj} subjects")

        # ================================================================
        # SCENARIO A: Same parameters (f1 = f2) — worst case
        # ================================================================
        print(f"\n  --- Scenario A: Same f (worst case) ---")

        # Create TWO vaults per subject using DIFFERENT impressions
        # (impression 0 for vault 1, impression 1 for vault 2)
        # This simulates realistic re-enrollment in different applications.
        vaults_a1 = {}  # vault set 1: impression 0
        vaults_a2 = {}  # vault set 2: impression 1

        for subj in subjects:
            cid = subj["chimeric_id"]
            mins0 = subj["minutiae"].get(0, [])
            mins1 = subj["minutiae"].get(1, [])
            iris0 = subj["iris"].get(0)
            iris1 = subj["iris"].get(1)
            if not mins0 or not mins1 or iris0 is None or iris1 is None:
                continue

            try:
                v1 = lock_vault_ibfv(mins0, iris0["code"], iris0["mask"],
                                     factor, degree, n_bonus=N_BONUS,
                                     block_size=BLOCK_SIZE, seed=cid)
                v2 = lock_vault_ibfv(mins1, iris1["code"], iris1["mask"],
                                     factor, degree, n_bonus=N_BONUS,
                                     block_size=BLOCK_SIZE, seed=cid + 10000)
                if v1 is not None and v2 is not None:
                    vaults_a1[cid] = v1
                    vaults_a2[cid] = v2
            except Exception as e:
                pass

        n_vaults = len(vaults_a1)
        print(f"    Created {n_vaults} vault pairs")

        # Extract x-coordinate sets
        xsets_a1 = {}
        xsets_a2 = {}
        for cid in vaults_a1:
            xs1 = extract_x_coords(vaults_a1[cid])
            xs2 = extract_x_coords(vaults_a2[cid])
            if xs1 and xs2:
                xsets_a1[cid] = xs1
                xsets_a2[cid] = xs2

        if not xsets_a1:
            print("    WARNING: No x-coordinates extracted — skipping this DB")
            continue

        print(f"    Extracted x-coords from {len(xsets_a1)} vault pairs")
        print(f"    Avg vault size: {np.mean([len(xs) for xs in xsets_a1.values()]):.1f} points")

        # Compute mated and non-mated linkage scores
        mated_a = []
        nonmated_a = []
        cids = sorted(xsets_a1.keys())

        for i, cid_i in enumerate(cids):
            # Mated: same subject, different vaults
            score = linkage_score(xsets_a1[cid_i], xsets_a2[cid_i])
            mated_a.append(score)

            # Non-mated: different subjects
            for j, cid_j in enumerate(cids):
                if i != j:
                    score_nm = linkage_score(xsets_a1[cid_i], xsets_a2[cid_j])
                    nonmated_a.append(score_nm)

        mated_a    = np.array(mated_a)
        nonmated_a = np.array(nonmated_a)

        print(f"    Mated scores:     n={len(mated_a)}, "
              f"mean={mated_a.mean():.2f}, std={mated_a.std():.2f}, "
              f"range=[{mated_a.min()}, {mated_a.max()}]")
        print(f"    Non-mated scores: n={len(nonmated_a)}, "
              f"mean={nonmated_a.mean():.2f}, std={nonmated_a.std():.2f}, "
              f"range=[{nonmated_a.min()}, {nonmated_a.max()}]")

        # Compute D_sys
        result_a = compute_dsys(mated_a, nonmated_a, omega=OMEGA, n_bins=N_BINS)
        print(f"    D_sys (same f) = {result_a['D_sys']:.4f}")

        # Plot
        plot_unlinkability(result_a, f"{db_label} — Same $f$",
                           FIG_DIR / f"dsys_{db_label}_same_f.pdf")

        all_results.append({
            "database": db_label, "scenario": "same_f",
            "factor1": factor, "factor2": factor,
            "degree": degree,
            "n_subjects": n_vaults,
            "mated_mean": round(float(mated_a.mean()), 4),
            "mated_std": round(float(mated_a.std()), 4),
            "nonmated_mean": round(float(nonmated_a.mean()), 4),
            "nonmated_std": round(float(nonmated_a.std()), 4),
            "D_sys": round(result_a["D_sys"], 6),
        })

        # ================================================================
        # SCENARIO B: Different parameters (f1 ≠ f2) — recommended
        # ================================================================
        print(f"\n  --- Scenario B: Different f (recommended) ---")

        # Use f2 = f + 2 for the second enrollment
        factor2 = factor + 2

        vaults_b1 = {}
        vaults_b2 = {}

        for subj in subjects:
            cid = subj["chimeric_id"]
            mins0 = subj["minutiae"].get(0, [])
            mins1 = subj["minutiae"].get(1, [])
            iris0 = subj["iris"].get(0)
            iris1 = subj["iris"].get(1)
            if not mins0 or not mins1 or iris0 is None or iris1 is None:
                continue

            try:
                v1 = lock_vault_ibfv(mins0, iris0["code"], iris0["mask"],
                                     factor, degree, n_bonus=N_BONUS,
                                     block_size=BLOCK_SIZE, seed=cid)
                v2 = lock_vault_ibfv(mins1, iris1["code"], iris1["mask"],
                                     factor2, degree, n_bonus=N_BONUS,
                                     block_size=BLOCK_SIZE, seed=cid + 10000)
                if v1 is not None and v2 is not None:
                    vaults_b1[cid] = v1
                    vaults_b2[cid] = v2
            except Exception as e:
                pass

        n_vaults_b = len(vaults_b1)
        print(f"    Created {n_vaults_b} vault pairs (f1={factor}, f2={factor2})")

        xsets_b1 = {}
        xsets_b2 = {}
        for cid in vaults_b1:
            xs1 = extract_x_coords(vaults_b1[cid])
            xs2 = extract_x_coords(vaults_b2[cid])
            if xs1 and xs2:
                xsets_b1[cid] = xs1
                xsets_b2[cid] = xs2

        if not xsets_b1:
            print("    WARNING: Could not extract x-coordinates for Scenario B")
            continue

        mated_b = []
        nonmated_b = []
        cids_b = sorted(xsets_b1.keys())

        for i, cid_i in enumerate(cids_b):
            score = linkage_score(xsets_b1[cid_i], xsets_b2[cid_i])
            mated_b.append(score)
            for j, cid_j in enumerate(cids_b):
                if i != j:
                    score_nm = linkage_score(xsets_b1[cid_i], xsets_b2[cid_j])
                    nonmated_b.append(score_nm)

        mated_b    = np.array(mated_b)
        nonmated_b = np.array(nonmated_b)

        print(f"    Mated scores:     n={len(mated_b)}, "
              f"mean={mated_b.mean():.2f}, std={mated_b.std():.2f}, "
              f"range=[{mated_b.min()}, {mated_b.max()}]")
        print(f"    Non-mated scores: n={len(nonmated_b)}, "
              f"mean={nonmated_b.mean():.2f}, std={nonmated_b.std():.2f}, "
              f"range=[{nonmated_b.min()}, {nonmated_b.max()}]")

        result_b = compute_dsys(mated_b, nonmated_b, omega=OMEGA, n_bins=N_BINS)
        print(f"    D_sys (diff f) = {result_b['D_sys']:.4f}")

        plot_unlinkability(result_b, f"{db_label} — Diff $f$",
                           FIG_DIR / f"dsys_{db_label}_diff_f.pdf")

        all_results.append({
            "database": db_label, "scenario": "diff_f",
            "factor1": factor, "factor2": factor2,
            "degree": degree,
            "n_subjects": n_vaults_b,
            "mated_mean": round(float(mated_b.mean()), 4),
            "mated_std": round(float(mated_b.std()), 4),
            "nonmated_mean": round(float(nonmated_b.mean()), 4),
            "nonmated_std": round(float(nonmated_b.std()), 4),
            "D_sys": round(result_b["D_sys"], 6),
        })

        # ================================================================
        # SCENARIO C: Diverse parameters (coprime f) — best practice
        # ================================================================
        # Each application independently picks f. Use coprime values to
        # ensure minimal quantization-grid overlap.
        DIVERSE_F = {22: 15, 16: 23}  # coprime pairs
        factor3 = DIVERSE_F.get(factor, factor + 7)
        print(f"\n  --- Scenario C: Diverse f (best practice) ---")

        vaults_c1 = {}
        vaults_c2 = {}

        for subj in subjects:
            cid = subj["chimeric_id"]
            mins0 = subj["minutiae"].get(0, [])
            mins1 = subj["minutiae"].get(1, [])
            iris0 = subj["iris"].get(0)
            iris1 = subj["iris"].get(1)
            if not mins0 or not mins1 or iris0 is None or iris1 is None:
                continue

            try:
                v1 = lock_vault_ibfv(mins0, iris0["code"], iris0["mask"],
                                     factor, degree, n_bonus=N_BONUS,
                                     block_size=BLOCK_SIZE, seed=cid)
                v2 = lock_vault_ibfv(mins1, iris1["code"], iris1["mask"],
                                     factor3, degree, n_bonus=N_BONUS,
                                     block_size=BLOCK_SIZE, seed=cid + 20000)
                if v1 is not None and v2 is not None:
                    vaults_c1[cid] = v1
                    vaults_c2[cid] = v2
            except Exception as e:
                pass

        n_vaults_c = len(vaults_c1)
        print(f"    Created {n_vaults_c} vault pairs (f1={factor}, f2={factor3})")

        xsets_c1 = {}
        xsets_c2 = {}
        for cid in vaults_c1:
            xs1 = extract_x_coords(vaults_c1[cid])
            xs2 = extract_x_coords(vaults_c2[cid])
            if xs1 and xs2:
                xsets_c1[cid] = xs1
                xsets_c2[cid] = xs2

        if xsets_c1:
            mated_c = []
            nonmated_c = []
            cids_c = sorted(xsets_c1.keys())

            for i, cid_i in enumerate(cids_c):
                score = linkage_score(xsets_c1[cid_i], xsets_c2[cid_i])
                mated_c.append(score)
                for j, cid_j in enumerate(cids_c):
                    if i != j:
                        score_nm = linkage_score(xsets_c1[cid_i], xsets_c2[cid_j])
                        nonmated_c.append(score_nm)

            mated_c    = np.array(mated_c)
            nonmated_c = np.array(nonmated_c)

            print(f"    Mated scores:     n={len(mated_c)}, "
                  f"mean={mated_c.mean():.2f}, std={mated_c.std():.2f}, "
                  f"range=[{mated_c.min()}, {mated_c.max()}]")
            print(f"    Non-mated scores: n={len(nonmated_c)}, "
                  f"mean={nonmated_c.mean():.2f}, std={nonmated_c.std():.2f}, "
                  f"range=[{nonmated_c.min()}, {nonmated_c.max()}]")

            result_c = compute_dsys(mated_c, nonmated_c, omega=OMEGA, n_bins=N_BINS)
            print(f"    D_sys (diverse f) = {result_c['D_sys']:.4f}")

            plot_unlinkability(result_c, f"{db_label} — Diverse $f$",
                               FIG_DIR / f"dsys_{db_label}_diverse_f.pdf")

            all_results.append({
                "database": db_label, "scenario": "diverse_f",
                "factor1": factor, "factor2": factor3,
                "degree": degree,
                "n_subjects": n_vaults_c,
                "mated_mean": round(float(mated_c.mean()), 4),
                "mated_std": round(float(mated_c.std()), 4),
                "nonmated_mean": round(float(nonmated_c.mean()), 4),
                "nonmated_std": round(float(nonmated_c.std()), 4),
                "D_sys": round(result_c["D_sys"], 6),
            })

    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    csv_path = RESULTS / "unlinkability_dsys_results.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Saved: {csv_path}")

    # ====================================================================
    # COMBINED FIGURE (all DBs, both scenarios)
    # ====================================================================
    generate_combined_figure(all_results)

    # Summary
    print("\n" + "=" * 70)
    print("UNLINKABILITY SUMMARY")
    print("=" * 70)
    for r in all_results:
        status = "UNLINKABLE" if r["D_sys"] < 0.05 else \
                 "PARTIALLY" if r["D_sys"] < 0.30 else "LINKABLE"
        print(f"  {r['database']:4s} {r['scenario']:8s}: D_sys = {r['D_sys']:.4f}  [{status}]"
              f"  mated={r['mated_mean']:.1f}±{r['mated_std']:.1f}"
              f"  non-mated={r['nonmated_mean']:.1f}±{r['nonmated_std']:.1f}")


def generate_combined_figure(all_results):
    """Generate a combined bar chart of D_sys across all DBs and scenarios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not all_results:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    dbs = ["DB1", "DB2", "DB3", "DB4"]
    same_f    = {r["database"]: r["D_sys"] for r in all_results if r["scenario"] == "same_f"}
    diff_f    = {r["database"]: r["D_sys"] for r in all_results if r["scenario"] == "diff_f"}
    diverse_f = {r["database"]: r["D_sys"] for r in all_results if r["scenario"] == "diverse_f"}

    x = np.arange(len(dbs))
    width = 0.25

    ax.bar(x - width, [same_f.get(db, 0) for db in dbs], width,
           label=r"Same $f$", color="#F44336", edgecolor="black", linewidth=0.5)
    ax.bar(x,         [diff_f.get(db, 0) for db in dbs], width,
           label=r"$\Delta f = 2$", color="#FF9800", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, [diverse_f.get(db, 0) for db in dbs], width,
           label=r"Diverse $f$", color="#4CAF50", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(dbs)
    ax.set_ylabel(r"$D_{\leftrightarrow}^{\mathrm{sys}}$")
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(3.6, 0.07, "Unlinkable", fontsize=6, ha="right", color="gray")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    out = FIG_DIR / "dsys_unlinkability.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved combined figure: {out.name}")


if __name__ == "__main__":
    run_unlinkability_analysis()
