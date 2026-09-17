#!/usr/bin/env python3
"""
Generate all figures for the IBFV IEEE paper.
Outputs PDF figures to paper_ieee/figures/
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Paths
CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CODE_DIR / "results"
PAPER_EXPERIMENTS = RESULTS_DIR / "paper_experiments"
COMPLETE_S1 = RESULTS_DIR / "complete_experiments" / "setup1_results_20260405_134407.csv"
COMPLETE_S2 = RESULTS_DIR / "complete_experiments" / "setup2_results_20260406_005116.csv"
FIG_DIR = CODE_DIR.parent / "paper_ieee" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

DB_LABELS = {
    'fvc2002_1': 'DB1', 'fvc2002_2': 'DB2',
    'fvc2002_3': 'DB3', 'fvc2004_1': 'DB4'
}
ARCH_COLORS = {'Unimodal': '#2196F3', 'A': '#F44336', 'IBFV': '#4CAF50'}
ARCH_MARKERS = {'Unimodal': 'o', 'A': 's', 'IBFV': '^'}
ARCH_LABELS = {'Unimodal': 'Unimodal', 'A': 'Arch. A (AND)', 'IBFV': 'IBFV (Ours)'}

def load_complete_results():
    """Load both setup results into a list of dicts."""
    rows = []
    for fpath in [COMPLETE_S1, COMPLETE_S2]:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['eer_pct'] = float(row['eer_pct'])
                row['far_pct'] = float(row['far_pct'])
                row['frr_pct'] = float(row['frr_pct'])
                row['factor'] = int(row['factor'])
                row['degree'] = int(row['degree'])
                row['setup'] = int(row['setup'])
                rows.append(row)
    return rows


def fig1_eer_vs_factor():
    """EER vs Quantization Factor curves for DB3 and DB4 (Setup 1 & 2)."""
    rows = load_complete_results()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), sharex=True)
    targets = [
        ('fvc2002_3', 1), ('fvc2004_1', 1),
        ('fvc2002_3', 2), ('fvc2004_1', 2),
    ]

    for ax, (db, setup) in zip(axes.flat, targets):
        for arch in ['Unimodal', 'A', 'IBFV']:
            # Best EER across degrees for each factor
            factor_eer = defaultdict(lambda: 999)
            for r in rows:
                if r['database'] == db and r['setup'] == setup and r['architecture'] == arch:
                    f = r['factor']
                    if r['eer_pct'] < factor_eer[f]:
                        factor_eer[f] = r['eer_pct']
            factors = sorted(factor_eer.keys())
            eers = [factor_eer[f] for f in factors]
            ax.plot(factors, eers, marker=ARCH_MARKERS[arch], color=ARCH_COLORS[arch],
                    label=ARCH_LABELS[arch], linewidth=1.5, markersize=5)

        ax.set_title(f'{DB_LABELS[db]} — Setup {setup}', fontsize=10)
        ax.set_ylabel('Best EER (%)')
        ax.set_xlabel('Quantization Factor $f$')
        ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'eer_vs_factor.pdf')
    plt.close()
    print(f"  Saved: eer_vs_factor.pdf")


def fig2_delta_eer_heatmap():
    """Heatmap of ΔEER (Unimodal - IBFV) for Setup 2."""
    rows = load_complete_results()
    dbs = ['fvc2002_1', 'fvc2002_2', 'fvc2002_3', 'fvc2004_1']

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8))

    for ax, db in zip(axes, dbs):
        factors = sorted(set(r['factor'] for r in rows if r['setup'] == 2))
        degrees = sorted(set(r['degree'] for r in rows if r['setup'] == 2))

        # Build EER lookup
        eer_lookup = {}
        for r in rows:
            if r['database'] == db and r['setup'] == 2:
                eer_lookup[(r['architecture'], r['factor'], r['degree'])] = r['eer_pct']

        delta = np.zeros((len(degrees), len(factors)))
        for i, k in enumerate(degrees):
            for j, f in enumerate(factors):
                uni = eer_lookup.get(('Unimodal', f, k), 0)
                ibfv = eer_lookup.get(('IBFV', f, k), 0)
                delta[i, j] = uni - ibfv

        im = ax.imshow(delta, aspect='auto', cmap='Greens', vmin=0,
                        vmax=max(3, delta.max()),
                        extent=[factors[0]-1, factors[-1]+1, degrees[-1]+1, degrees[0]-1])
        ax.set_xticks(factors[::2])
        ax.set_yticks(degrees)
        ax.set_title(DB_LABELS[db], fontsize=9)
        if db == dbs[0]:
            ax.set_ylabel('Degree $k$')
        ax.set_xlabel('Factor $f$')

        # Annotate cells
        for i, k in enumerate(degrees):
            for j, f in enumerate(factors):
                val = delta[i, j]
                if val > 0:
                    ax.text(f, k, f'{val:.1f}', ha='center', va='center',
                            fontsize=5, color='black' if val < 2 else 'white')

    fig.suptitle('$\\Delta$EER (Unimodal $-$ IBFV) in pp — Setup 2', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'delta_eer_heatmap.pdf')
    plt.close()
    print(f"  Saved: delta_eer_heatmap.pdf")


def fig3_roc_curves():
    """ROC curves from roc_det_data.csv."""
    roc_file = PAPER_EXPERIMENTS / "roc_det_data.csv"
    if not roc_file.exists():
        print("  SKIP: roc_det_data.csv not found")
        return

    data = defaultdict(lambda: defaultdict(list))  # data[db][setup][arch] = [(far, gar)]
    with open(roc_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            db = row['database']
            setup = int(row['setup'])
            arch = row['architecture']
            far = float(row['far_pct'])
            gar = float(row['gar_pct'])
            data[(db, setup)][arch].append((far, gar))

    # Plot Setup 2 only (more interesting)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))
    dbs = ['fvc2002_1', 'fvc2002_2', 'fvc2002_3', 'fvc2004_1']

    for ax, db in zip(axes, dbs):
        for arch in ['Unimodal', 'IBFV']:
            points = sorted(data[(db, 2)][arch])
            if not points:
                continue
            fars = [p[0] for p in points]
            gars = [p[1] for p in points]
            ax.plot(fars, gars, marker=ARCH_MARKERS[arch], color=ARCH_COLORS[arch],
                    label=ARCH_LABELS[arch], linewidth=1.5, markersize=4)

        ax.set_title(DB_LABELS[db], fontsize=9)
        ax.set_xlabel('FAR (%)')
        if db == dbs[0]:
            ax.set_ylabel('GAR (%)')
        ax.legend(fontsize=6, loc='lower right')
        ax.set_xlim(-0.5, None)
        ax.set_ylim(None, 101)

    fig.suptitle('ROC Curves — Setup 2', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'roc_curves_setup2.pdf')
    plt.close()
    print(f"  Saved: roc_curves_setup2.pdf")


def fig4_architecture_comparison_bar():
    """Bar chart comparing best EER of all architectures (Setup 2)."""
    rows = load_complete_results()
    dbs = ['fvc2002_1', 'fvc2002_2', 'fvc2002_3', 'fvc2004_1']
    archs = ['Unimodal', 'A', 'IBFV']

    # Find best EER per db per arch for Setup 2
    best_eer = defaultdict(lambda: defaultdict(lambda: 999))
    for r in rows:
        if r['setup'] == 2:
            db, arch = r['database'], r['architecture']
            if r['eer_pct'] < best_eer[db][arch]:
                best_eer[db][arch] = r['eer_pct']

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(dbs))
    width = 0.25

    for i, arch in enumerate(archs):
        vals = [best_eer[db][arch] for db in dbs]
        ax.bar(x + i * width, vals, width, label=ARCH_LABELS[arch],
               color=ARCH_COLORS[arch], edgecolor='white', linewidth=0.5)
        # Value labels
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.15, f'{v:.2f}', ha='center', fontsize=6)

    ax.set_xticks(x + width)
    ax.set_xticklabels([DB_LABELS[db] for db in dbs])
    ax.set_ylabel('Best EER (%)')
    ax.set_title('Architecture Comparison — Setup 2', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylim(0, max(best_eer[db]['A'] for db in dbs) + 1.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'arch_comparison_setup2.pdf')
    plt.close()
    print(f"  Saved: arch_comparison_setup2.pdf")


def fig5_nbonus_ablation():
    """n_bonus ablation study if data available."""
    ablation_file = PAPER_EXPERIMENTS / "nbonus_ablation_results.csv"
    if not ablation_file.exists():
        print("  SKIP: nbonus_ablation_results.csv not yet available")
        return

    data = defaultdict(list)  # data[(db, factor, degree)] = [(n_bonus, eer)]
    with open(ablation_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            db = row['database']
            f_val = int(row['factor'])
            k_val = int(row['degree'])
            nb = int(row['n_bonus'])
            eer = float(row['eer_pct'])
            data[(db, f_val, k_val)].append((nb, eer))

    # Get unique DBs
    dbs_in_data = sorted(set(k[0] for k in data.keys()))

    fig, axes = plt.subplots(1, len(dbs_in_data), figsize=(3.5 * len(dbs_in_data), 3))
    if len(dbs_in_data) == 1:
        axes = [axes]

    for ax, db in zip(axes, dbs_in_data):
        configs = [(f, k) for (d, f, k) in data.keys() if d == db]
        # Pick a representative config (mid-range factor)
        configs.sort()
        for f_val, k_val in configs[:4]:  # show up to 4 configs
            points = sorted(data[(db, f_val, k_val)])
            nbs = [p[0] for p in points]
            eers = [p[1] for p in points]
            ax.plot(nbs, eers, marker='o', label=f'$f={f_val}, k={k_val}$',
                    linewidth=1.2, markersize=4)

        ax.set_title(DB_LABELS.get(db, db), fontsize=9)
        ax.set_xlabel('Number of Bonus Points $K$')
        if db == dbs_in_data[0]:
            ax.set_ylabel('EER (%)')
        ax.legend(fontsize=6, loc='best')

    fig.suptitle('Ablation: Effect of Bonus Points $K$ — Setup 1', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'nbonus_ablation.pdf')
    plt.close()
    print(f"  Saved: nbonus_ablation.pdf")


def fig6_iris_gar_bar():
    """Bar chart of iris GAR by database."""
    gar_file = PAPER_EXPERIMENTS / "iris_gar_results.csv"
    if not gar_file.exists():
        print("  SKIP: iris_gar_results.csv not found")
        return

    gar_data = {}
    with open(gar_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            db = row['database']
            setup = int(row['setup'])
            gar_data[(db, setup)] = float(row['iris_gar_pct'])

    dbs = ['fvc2002_1', 'fvc2002_2', 'fvc2002_3', 'fvc2004_1']
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    x = np.arange(len(dbs))
    width = 0.35

    s1 = [gar_data.get((db, 1), 0) for db in dbs]
    s2 = [gar_data.get((db, 2), 0) for db in dbs]

    ax.bar(x - width/2, s1, width, label='Setup 1', color='#42A5F5', edgecolor='white')
    ax.bar(x + width/2, s2, width, label='Setup 2', color='#FF7043', edgecolor='white')

    for i, (v1, v2) in enumerate(zip(s1, s2)):
        ax.text(x[i] - width/2, v1 + 0.8, f'{v1:.1f}%', ha='center', fontsize=7)
        ax.text(x[i] + width/2, v2 + 0.8, f'{v2:.1f}%', ha='center', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([DB_LABELS[db] for db in dbs])
    ax.set_ylabel('Iris Key Recovery GAR (%)')
    ax.set_title('Iris Subsystem Performance ($B = 63$)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 65)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'iris_gar.pdf')
    plt.close()
    print(f"  Saved: iris_gar.pdf")


def fig7_det_curves():
    """DET curves (FRR vs FAR on log scale) — Setup 2."""
    roc_file = PAPER_EXPERIMENTS / "roc_det_data.csv"
    if not roc_file.exists():
        print("  SKIP: roc_det_data.csv not found")
        return

    data = defaultdict(lambda: defaultdict(list))
    with open(roc_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            db = row['database']
            setup = int(row['setup'])
            arch = row['architecture']
            far = float(row['far_pct'])
            frr = float(row['frr_pct'])
            data[(db, setup)][arch].append((far, frr))

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))
    dbs = ['fvc2002_1', 'fvc2002_2', 'fvc2002_3', 'fvc2004_1']

    for ax, db in zip(axes, dbs):
        for arch in ['Unimodal', 'IBFV']:
            points = sorted(data[(db, 2)][arch])
            if not points:
                continue
            fars = [max(p[0], 0.001) for p in points]  # avoid log(0)
            frrs = [max(p[1], 0.001) for p in points]
            ax.plot(fars, frrs, marker=ARCH_MARKERS[arch], color=ARCH_COLORS[arch],
                    label=ARCH_LABELS[arch], linewidth=1.5, markersize=4)

        ax.set_title(DB_LABELS[db], fontsize=9)
        ax.set_xlabel('FAR (%)')
        if db == dbs[0]:
            ax.set_ylabel('FRR (%)')
        ax.legend(fontsize=6, loc='upper right')
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.suptitle('DET Curves — Setup 2', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'det_curves_setup2.pdf')
    plt.close()
    print(f"  Saved: det_curves_setup2.pdf")


if __name__ == '__main__':
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    fig1_eer_vs_factor()
    fig2_delta_eer_heatmap()
    fig3_roc_curves()
    fig4_architecture_comparison_bar()
    fig5_nbonus_ablation()
    fig6_iris_gar_bar()
    fig7_det_curves()

    print("\nDone. All figures saved to:", FIG_DIR)
