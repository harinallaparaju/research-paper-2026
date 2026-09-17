"""
Generate publication-quality figures for D-IBFV system experiments (Section VIII).
Reads CSVs from results/ and outputs PDFs to paper_extended/figures/.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = Path(__file__).resolve().parent.parent.parent / "paper_extended" / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

GANACHE_COLOR = '#2196F3'
FABRIC_COLOR  = '#FF9800'
ACCENT        = '#4CAF50'


# ═══════════════════════════════════════════════════════════
# Figure 1: Enrollment Latency Breakdown (Stacked Bar)
# ═══════════════════════════════════════════════════════════
def fig_enrollment_latency():
    rows = list(csv.DictReader(open(RESULTS / "enrollment_latency.csv")))
    phases = ['serialize', 'split', 'ipfs', 'blockchain']
    labels = ['Serialization', 'Shamir Split', 'IPFS Upload (×5)', 'Blockchain TX']
    colors = ['#E8F5E9', '#81C784', '#42A5F5', '#EF5350']

    data = {}
    for r in rows:
        p = r['Platform']
        if p not in data:
            data[p] = {}
        data[p][r['Phase']] = (float(r['Mean_ms']), float(r['Std_ms']))

    platforms = ['Ganache', 'Fabric']
    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(platforms))
    width = 0.55
    bottoms = np.zeros(len(platforms))

    for phase, label, color in zip(phases, labels, colors):
        vals = [data[p][phase][0] for p in platforms]
        bars = ax.bar(x, vals, width, bottom=bottoms, label=label, color=color,
                      edgecolor='white', linewidth=0.5)
        bottoms += vals

    # Total labels on top
    for i, p in enumerate(platforms):
        total = data[p]['total'][0]
        std = data[p]['total'][1]
        ax.text(i, total + 15, f'{total:.0f}±{std:.0f} ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontweight='bold')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Enrollment Latency Breakdown')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, bottoms.max() * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = FIGURES / "enrollment_latency_breakdown.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════
# Figure 2: Verification Latency Breakdown (Stacked Bar)
# ═══════════════════════════════════════════════════════════
def fig_verification_latency():
    rows = list(csv.DictReader(open(RESULTS / "verification_latency.csv")))
    phases = ['blockchain_lookup', 'ipfs_download', 'shamir_reconstruct', 'deserialize']
    labels = ['Blockchain Lookup', 'IPFS Download (×3)', 'Shamir Reconstruct', 'Deserialization']
    colors = ['#EF5350', '#42A5F5', '#81C784', '#E8F5E9']

    data = {}
    for r in rows:
        p = r['Platform']
        if p not in data:
            data[p] = {}
        data[p][r['Phase']] = (float(r['Mean_ms']), float(r['Std_ms']))

    platforms = ['Ganache', 'Fabric']
    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(platforms))
    width = 0.55
    bottoms = np.zeros(len(platforms))

    for phase, label, color in zip(phases, labels, colors):
        vals = [data[p][phase][0] for p in platforms]
        bars = ax.bar(x, vals, width, bottom=bottoms, label=label, color=color,
                      edgecolor='white', linewidth=0.5)
        bottoms += vals

    for i, p in enumerate(platforms):
        total = data[p]['total'][0]
        std = data[p]['total'][1]
        ax.text(i, total + 1.5, f'{total:.1f}±{std:.1f} ms',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontweight='bold')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Verification Latency Breakdown')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, bottoms.max() * 1.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = FIGURES / "verification_latency_breakdown.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════
# Figure 3: Throughput Comparison (Grouped Bar)
# ═══════════════════════════════════════════════════════════
def fig_throughput():
    rows = list(csv.DictReader(open(RESULTS / "throughput.csv")))
    data = {}
    for r in rows:
        data[(r['Platform'], r['Operation'])] = float(r['TPS'])

    ops = ['store', 'lookup', 'revoke']
    op_labels = ['Store', 'Lookup', 'Revoke']

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(ops))
    w = 0.3

    ganache_vals = [data.get(('Ganache', o), 0) for o in ops]
    fabric_vals  = [data.get(('Fabric', o), 0) for o in ops]

    bars1 = ax.bar(x - w/2, ganache_vals, w, label='Ganache (EVM)',
                   color=GANACHE_COLOR, edgecolor='white')
    bars2 = ax.bar(x + w/2, fabric_vals, w, label='Fabric 2.5',
                   color=FABRIC_COLOR, edgecolor='white')

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(op_labels, fontweight='bold')
    ax.set_ylabel('Transactions per Second (TPS)')
    ax.set_title('Throughput by Operation')
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, max(ganache_vals) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = FIGURES / "throughput_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════
# Figure 4: Shamir Parameter Trade-off (Dual-Axis)
# ═══════════════════════════════════════════════════════════
def fig_shamir_tradeoff():
    rows = list(csv.DictReader(open(RESULTS / "shamir_params.csv")))

    configs = [r['Config'] for r in rows]
    split_ms = [float(r['Split_ms']) for r in rows]
    recon_ms = [float(r['Recon_ms']) for r in rows]
    storage  = [float(r['Storage_KB']) for r in rows]
    fault_tol = [int(r['Fault_Tolerance']) for r in rows]

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(configs))

    # Left axis: timing
    l1, = ax1.plot(x, split_ms, 'o-', color='#EF5350', label='Split time', linewidth=2, markersize=7)
    l2, = ax1.plot(x, recon_ms, 's--', color='#42A5F5', label='Reconstruct time', linewidth=2, markersize=7)
    ax1.set_ylabel('Time (ms)', color='#333')
    ax1.set_ylim(0, max(split_ms) * 1.4)

    # Right axis: storage
    ax2 = ax1.twinx()
    l3 = ax2.bar(x, storage, 0.35, alpha=0.25, color='#9E9E9E', label='Storage (KB)')
    ax2.set_ylabel('Total Storage (KB)', color='#777')
    ax2.set_ylim(0, max(storage) * 1.3)

    # Fault-tolerance annotations
    for i, ft in enumerate(fault_tol):
        ax1.annotate(f'FT={ft}', (x[i], split_ms[i]),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=8, color='#666')

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontweight='bold')
    ax1.set_xlabel('Shamir Configuration $(t, n)$')
    ax1.set_title('Shamir Parameter Trade-off')

    lines = [l1, l2, l3]
    labels = ['Split time (ms)', 'Reconstruct time (ms)', 'Storage (KB)']
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    out = FIGURES / "shamir_tradeoff.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════
# Figure 5: Node Failure Resilience (Step Chart)
# ═══════════════════════════════════════════════════════════
def fig_node_failure():
    rows = list(csv.DictReader(open(RESULTS / "node_failure.csv")))
    failures = [int(r['Failures']) for r in rows]
    available = [int(r['Available']) for r in rows]
    rate_str = [r['Recon_Success_Rate'] for r in rows]
    rates = [float(r.replace('%', '')) for r in rate_str]

    fig, ax = plt.subplots(figsize=(5, 3))

    colors = [ACCENT if r == 100 else '#EF5350' for r in rates]
    bars = ax.bar(failures, rates, color=colors, edgecolor='white', width=0.6)

    # Threshold line (no annotation text to avoid overlap with bars)
    ax.axvline(x=2.5, color='#999', linestyle='--', linewidth=1)

    for i, (f, r) in enumerate(zip(failures, rates)):
        label = f'{r:.0f}%'
        ax.text(f, r + 2, label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Number of Failed Nodes')
    ax.set_ylabel('Reconstruction Success Rate (%)')
    ax.set_title('Node Failure Resilience: $(3,5)$-Shamir')
    ax.set_ylim(-5, 120)
    ax.set_xticks(failures)
    ax.set_xticklabels([f'{f}\n({a} avail.)' for f, a in zip(failures, available)], fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = FIGURES / "node_failure_resilience.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════
# Figure 6: Storage Cost Breakdown (Stacked Bar)
# ═══════════════════════════════════════════════════════════
def fig_storage_costs():
    rows = list(csv.DictReader(open(RESULTS / "storage_costs.csv")))

    configs = [r['Config'] for r in rows]
    ipfs_kb = [float(r['IPFS_Total_KB']) for r in rows]
    chain_kb = [float(r['Blockchain_KB']) for r in rows]

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(configs))
    w = 0.5

    ax.bar(x, ipfs_kb, w, label='IPFS (off-chain)', color='#42A5F5', edgecolor='white')
    ax.bar(x, chain_kb, w, bottom=ipfs_kb, label='Blockchain (on-chain)', color='#EF5350', edgecolor='white')

    for i in range(len(configs)):
        total = ipfs_kb[i] + chain_kb[i]
        ax.text(i, total + 5, f'{total:.0f} KB', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontweight='bold')
    ax.set_xlabel('Shamir Configuration $(t, n)$')
    ax.set_ylabel('Per-User Storage (KB)')
    ax.set_title('Storage Cost Breakdown')
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, max(ipfs_kb) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out = FIGURES / "storage_cost_breakdown.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════
# Figure 7: Platform Radar Chart
# ═══════════════════════════════════════════════════════════
def fig_platform_radar():
    """Radar/spider chart comparing Ganache vs Fabric across 5 dimensions."""
    categories = ['Enrollment\nSpeed', 'Verification\nSpeed', 'Store\nTPS',
                  'Lookup\nTPS', 'Fault\nTolerance']

    # Normalize to 0-1 scale (higher = better)
    # Enrollment: lower is better → invert
    ganache_raw = [277.4, 62.72, 7.0, 18.6, 2]
    fabric_raw  = [805.5, 56.81, 1.5, 15.5, 2]

    # Normalize: for latency, use 1-x/max; for TPS and FT, use x/max
    maxvals = [max(a, b) for a, b in zip(ganache_raw, fabric_raw)]
    
    def norm(raw, maxv, invert=False):
        v = raw / maxv
        return 1 - v if invert else v
    
    ganache_norm = [
        norm(ganache_raw[0], 1000, invert=True),  # enrollment: 1 - 277/1000
        norm(ganache_raw[1], 100, invert=True),    # verification
        norm(ganache_raw[2], 10, invert=False),    # store TPS
        norm(ganache_raw[3], 25, invert=False),    # lookup TPS
        norm(ganache_raw[4], 4, invert=False),     # fault tolerance
    ]
    fabric_norm = [
        norm(fabric_raw[0], 1000, invert=True),
        norm(fabric_raw[1], 100, invert=True),
        norm(fabric_raw[2], 10, invert=False),
        norm(fabric_raw[3], 25, invert=False),
        norm(fabric_raw[4], 4, invert=False),
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    ganache_norm += ganache_norm[:1]
    fabric_norm += fabric_norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    
    ax.fill(angles, ganache_norm, alpha=0.15, color=GANACHE_COLOR)
    ax.plot(angles, ganache_norm, 'o-', color=GANACHE_COLOR, linewidth=2, label='Ganache (EVM)', markersize=6)
    
    ax.fill(angles, fabric_norm, alpha=0.15, color=FABRIC_COLOR)
    ax.plot(angles, fabric_norm, 's--', color=FABRIC_COLOR, linewidth=2, label='Fabric 2.5', markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=7)
    ax.set_title('Platform Capability Profile', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, -0.05), framealpha=0.9)

    out = FIGURES / "platform_radar.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating D-IBFV system figures...")
    fig_enrollment_latency()
    fig_verification_latency()
    fig_throughput()
    fig_shamir_tradeoff()
    fig_node_failure()
    fig_storage_costs()
    fig_platform_radar()
    print(f"\nAll figures saved to: {FIGURES}")
