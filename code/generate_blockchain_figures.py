#!/usr/bin/env python3
"""Generate blockchain benchmark figures for the paper.

Figure A: Concurrent TPS vs Workers (line chart, 2 panels: store + lookup)
Figure B: Scalability — Verification Latency vs Enrolled Users
"""

from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / 'paper_extended' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = Path(__file__).resolve().parent / 'dibfv' / 'results'

COL_W = 3.5
DBL_W = 7.16

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
})

COLORS = {
    'Ganache': '#2C3E50',
    'Fabric': '#3498DB',
}
MARKERS = {
    'Ganache': 'o',
    'Fabric': 's',
}


def _read_csv(name):
    with open(RES_DIR / name) as f:
        return list(csv.DictReader(f))


# =====================================================================
# Figure: Concurrent TPS vs Workers
# =====================================================================

def fig_concurrent_tps():
    rows = _read_csv('concurrent_throughput.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.0))

    for pname in ['Ganache', 'Fabric']:
        prows = [r for r in rows if r['Platform'] == pname]
        workers = [int(r['Workers']) for r in prows]
        store_tps = [float(r['Store_TPS']) for r in prows]
        lookup_tps = [float(r['Lookup_TPS']) for r in prows]

        ax1.plot(workers, store_tps, marker=MARKERS[pname],
                 color=COLORS[pname], label=pname, linewidth=1.2,
                 markersize=4)
        ax2.plot(workers, lookup_tps, marker=MARKERS[pname],
                 color=COLORS[pname], label=pname, linewidth=1.2,
                 markersize=4)

    for ax, title in [(ax1, 'Store (Write)'), (ax2, 'Lookup (Read)')]:
        ax.set_xlabel('Concurrent Workers')
        ax.set_ylabel('Throughput (TPS)')
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(frameon=True, fancybox=False, edgecolor='#ccc')

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG_DIR / 'fig_concurrent_tps.pdf')
    plt.close(fig)
    print('  [1/2] fig_concurrent_tps.pdf')


# =====================================================================
# Figure: Scalability — Verification Latency vs Enrolled Users
# =====================================================================

def fig_scalability():
    rows = _read_csv('scalability.csv')

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))

    for pname in ['Ganache', 'Fabric']:
        prows = [r for r in rows if r['Platform'] == pname]
        users = [int(r['Enrolled_Users']) for r in prows]
        mean = [float(r['Mean_ms']) for r in prows]
        p95 = [float(r['P95_ms']) for r in prows]

        ax.plot(users, mean, marker=MARKERS[pname],
                color=COLORS[pname], label=f'{pname} (mean)',
                linewidth=1.2, markersize=4)
        ax.plot(users, p95, marker=MARKERS[pname],
                color=COLORS[pname], label=f'{pname} (p95)',
                linewidth=0.8, markersize=3, linestyle='--', alpha=0.6)

    ax.set_xlabel('Number of Enrolled Users')
    ax.set_ylabel('Verification Latency (ms)')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(frameon=True, fancybox=False, edgecolor='#ccc')
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig_scalability.pdf')
    plt.close(fig)
    print('  [2/2] fig_scalability.pdf')


if __name__ == '__main__':
    print('Generating blockchain benchmark figures...')
    fig_concurrent_tps()
    fig_scalability()
    print('Done.')
