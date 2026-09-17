#!/usr/bin/env python3
"""
Generate NEW figures for the IBFV paper from sweep and bootstrap data.

Figures:
1. Iris GAR vs Block Size (security-utility tradeoff)
2. EER vs Block Size for all architectures
3. FHD threshold sweep for Arch A
4. Bootstrap CI comparison (error bar plot)

All PDFs output to paper_ieee/figures/
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# === PATHS ===
CODE_DIR = Path(__file__).resolve().parent
PAPER_EXP = CODE_DIR / "results" / "paper_experiments"
FIG_DIR = CODE_DIR.parent / "paper_ieee" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# === IEEE STYLE ===
COL_W = 3.5
DBL_W = 7.16

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.4,
    'lines.markersize': 4,
    'mathtext.fontset': 'cm',
})

# Colorblind-safe palette
PAL = {
    'Uni':  '#0072B2',
    'A':    '#D55E00',
    'IBFV': '#009E73',
    'B':    '#E69F00',
    'C':    '#CC79A7',
    'D':    '#56B4E9',
    'GAR':  '#009E73',
    'FAR':  '#D55E00',
}
M = {'Uni': 'o', 'A': 's', 'IBFV': '^', 'B': 'v', 'C': 'D', 'D': 'P'}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ============================================================================
# FIGURE 1: IRIS GAR vs BLOCK SIZE (Security-Utility Tradeoff)
# ============================================================================
def fig_iris_gar_vs_bs():
    """Dual y-axis: GAR (left) and FAR (right) vs block size."""
    csv_path = PAPER_EXP / "iris_gar_bs_sweep.csv"
    if not csv_path.exists():
        print("  SKIP: iris_gar_bs_sweep.csv not found")
        return

    data = load_csv(csv_path)

    # Average across databases (they're nearly identical since same iris codes)
    bs_vals = sorted(set(int(r["block_size"]) for r in data))
    avg_gar = []
    avg_far = []
    avg_keybits = []
    N_IRIS = 49152  # 2 wavelengths × 24 rows × 512 cols × 2 phase bits
    for bs in bs_vals:
        rows = [r for r in data if int(r["block_size"]) == bs]
        avg_gar.append(np.mean([float(r["gar_pct"]) for r in rows]))
        avg_far.append(np.mean([float(r["far_pct"]) for r in rows]))
        avg_keybits.append(N_IRIS // bs)

    fig, ax1 = plt.subplots(figsize=(COL_W, 2.2))

    # GAR on left y-axis
    l1, = ax1.plot(bs_vals, avg_gar, 'o-', color=PAL['GAR'], label='Iris GAR', zorder=3)
    ax1.set_xlabel(r'Block size $B_s$')
    ax1.set_ylabel('Genuine Accept Rate (%)', color=PAL['GAR'])
    ax1.tick_params(axis='y', labelcolor=PAL['GAR'])
    ax1.set_ylim(0, 100)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(bs_vals)
    ax1.set_xticklabels([str(b) for b in bs_vals])
    ax1.grid(True, alpha=0.2)

    # FAR on right y-axis
    ax2 = ax1.twinx()
    l2, = ax2.plot(bs_vals, avg_far, 's--', color=PAL['FAR'], label='Iris FAR', zorder=3)
    ax2.set_ylabel('False Accept Rate (%)', color=PAL['FAR'])
    ax2.tick_params(axis='y', labelcolor=PAL['FAR'])
    ax2.set_ylim(-1, 40)

    # Annotate key bits on GAR line
    for i, (bs, gar, kb) in enumerate(zip(bs_vals, avg_gar, avg_keybits)):
        ax1.annotate(f'{int(kb)} bits',
                     (bs, gar), textcoords="offset points",
                     xytext=(0, 8), fontsize=5.5, ha='center', color='#333')

    # Highlight secure zone (Bs <= 63)
    ax1.axvspan(bs_vals[0] * 0.8, 90, alpha=0.08, color='green')
    ax1.text(45, 95, 'Secure\nzone', fontsize=6, ha='center', color='green', alpha=0.7)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', framealpha=0.9)

    fig.savefig(FIG_DIR / "iris_gar_vs_bs.pdf")
    plt.close(fig)
    print(f"  Saved: iris_gar_vs_bs.pdf")


# ============================================================================
# FIGURE 2: EER vs BLOCK SIZE FOR ALL ARCHITECTURES
# ============================================================================
def fig_eer_vs_bs():
    """EER vs block size for B/C/D, IBFV, and unimodal."""
    bcd_path = PAPER_EXP / "arch_bcd_bs_sweep.csv"
    ibfv_path = PAPER_EXP / "ibfv_unimodal_bs_sweep.csv"

    if not bcd_path.exists() or not ibfv_path.exists():
        print("  SKIP: Sweep CSVs not yet available for EER vs Bs figure")
        return

    bcd_data = load_csv(bcd_path)
    ibfv_data = load_csv(ibfv_path)

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    # Average EER across databases for each architecture and block_size
    for arch_label, data_list, color, marker in [
        ("B", bcd_data, PAL['B'], M['B']),
        ("C", bcd_data, PAL['C'], M['C']),
        ("D", bcd_data, PAL['D'], M['D']),
    ]:
        arch_rows = [r for r in data_list if r["architecture"] == f"Arch{arch_label}"]
        if not arch_rows:
            arch_rows = [r for r in data_list if r["architecture"] == arch_label]
        bs_vals = sorted(set(int(r["block_size"]) for r in arch_rows))
        eers = []
        for bs in bs_vals:
            rows = [r for r in arch_rows if int(r["block_size"]) == bs]
            eers.append(np.mean([float(r["eer_pct"]) for r in rows]))
        if eers:
            ax.plot(bs_vals, eers, f'{marker}--', color=color, label=f'Arch {arch_label}',
                    markersize=4, alpha=0.8)

    # IBFV
    ibfv_rows = [r for r in ibfv_data if r["architecture"] == "IBFV"]
    if ibfv_rows:
        bs_vals = sorted(set(int(r["block_size"]) for r in ibfv_rows))
        eers = []
        for bs in bs_vals:
            rows = [r for r in ibfv_rows if int(r["block_size"]) == bs]
            eers.append(np.mean([float(r["eer_pct"]) for r in rows]))
        ax.plot(bs_vals, eers, '^-', color=PAL['IBFV'], label='IBFV', linewidth=2, markersize=5)

    # Unimodal
    uni_rows = [r for r in ibfv_data if r["architecture"] == "Unimodal"]
    if uni_rows:
        bs_vals = sorted(set(int(r["block_size"]) for r in uni_rows))
        eers = []
        for bs in bs_vals:
            rows = [r for r in uni_rows if int(r["block_size"]) == bs]
            eers.append(np.mean([float(r["eer_pct"]) for r in rows]))
        ax.axhline(y=eers[0], color=PAL['Uni'], linestyle='-', label='Unimodal', alpha=0.7)

    ax.set_xlabel(r'Block size $B_s$')
    ax.set_ylabel('EER (%)')
    ax.set_xscale('log', base=2)
    ax.set_xticks([31, 63, 127, 255, 511])
    ax.set_xticklabels(['31', '63', '127', '255', '511'])
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(bottom=-0.5)

    fig.savefig(FIG_DIR / "eer_vs_bs.pdf")
    plt.close(fig)
    print(f"  Saved: eer_vs_bs.pdf")


# ============================================================================
# FIGURE 3: FHD THRESHOLD SWEEP FOR ARCH A
# ============================================================================
def fig_arch_a_fhd_sweep():
    """EER vs FHD threshold for Architecture A."""
    csv_path = PAPER_EXP / "arch_a_fhd_sweep.csv"
    if not csv_path.exists():
        print("  SKIP: arch_a_fhd_sweep.csv not found")
        return

    data = load_csv(csv_path)
    thresholds = sorted(set(float(r["fhd_threshold"]) for r in data))

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))

    # Average across databases
    eers = []
    gars = []
    fars = []
    for thr in thresholds:
        rows = [r for r in data if abs(float(r["fhd_threshold"]) - thr) < 0.001]
        eers.append(np.mean([float(r["eer_pct"]) for r in rows]))
        fars.append(np.mean([float(r["far_pct"]) for r in rows]))

    ax.plot(thresholds, eers, 's-', color=PAL['A'], label='Arch A EER', linewidth=1.5)
    ax.plot(thresholds, fars, 'o--', color=PAL['FAR'], label='Arch A FAR', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('FHD Threshold $\\tau$')
    ax.set_ylabel('Rate (%)')
    ax.grid(True, alpha=0.2)
    ax.legend(framealpha=0.9)
    ax.set_ylim(bottom=-0.2)

    fig.savefig(FIG_DIR / "arch_a_fhd_sweep.pdf")
    plt.close(fig)
    print(f"  Saved: arch_a_fhd_sweep.pdf")


# ============================================================================
# FIGURE 4: BOOTSTRAP CI COMPARISON
# ============================================================================
def fig_bootstrap_ci():
    """Error bar plot: EER with 95% CIs for all architectures, all 4 DBs."""
    csv_path = PAPER_EXP / "bootstrap_ci_results.csv"
    if not csv_path.exists():
        print("  SKIP: bootstrap_ci_results.csv not found")
        return

    data = load_csv(csv_path)

    arch_order = ['Unimodal', 'IBFV', 'A', 'B', 'C', 'D']
    arch_colors = [PAL['Uni'], PAL['IBFV'], PAL['A'], PAL['B'], PAL['C'], PAL['D']]

    # Best config per DB (matching the paper table)
    db_configs = {
        'DB1': ('22', '7'),
        'DB2': ('22', '7'),
        'DB3': ('16', '5'),
        'DB4': ('22', '5'),
    }

    fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 2.2), sharey=True)

    for idx, (db_name, (f_val, k_val)) in enumerate(db_configs.items()):
        ax = axes[idx]
        db_rows = [r for r in data if r["database"] == db_name
                   and r["factor"] == f_val and r["degree"] == k_val]

        x_pos = np.arange(len(arch_order))
        for i, arch in enumerate(arch_order):
            rows = [r for r in db_rows if r["architecture"] == arch]
            if not rows:
                continue
            r = rows[0]
            eer = float(r["eer"])
            lo = float(r["eer_ci_lo"])
            hi = float(r["eer_ci_hi"])
            ax.bar(i, eer, yerr=[[eer - lo], [hi - eer]], color=arch_colors[i],
                   capsize=3, edgecolor='black', linewidth=0.4, alpha=0.85, width=0.7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(arch_order, rotation=45, ha='right', fontsize=6)
        ax.set_title(f'{db_name} ($f$={f_val}, $k$={k_val})', fontsize=7)
        ax.grid(True, axis='y', alpha=0.2)
        if idx == 0:
            ax.set_ylabel('EER (%) with 95% CI')

    fig.tight_layout()
    fig.savefig(FIG_DIR / "bootstrap_ci_comparison.pdf")
    plt.close(fig)
    print(f"  Saved: bootstrap_ci_comparison.pdf")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Generating sweep & bootstrap figures...")
    fig_iris_gar_vs_bs()
    fig_eer_vs_bs()
    fig_arch_a_fhd_sweep()
    fig_bootstrap_ci()
    print("Done.")
