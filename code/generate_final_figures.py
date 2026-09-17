#!/usr/bin/env python3
"""
Generate ALL publication-quality figures for the IBFV IEEE paper.

- 5 architectural diagrams (grid-aligned, precise arrows)
- 11 data plots (IEEE color palette, professional styling)

All PDFs output to paper_ieee/figures/
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict

# === PATHS ===
CODE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CODE_DIR / "results"
PAPER_EXP = RESULTS_DIR / "paper_experiments"
COMPLETE_S1 = RESULTS_DIR / "complete_experiments" / "setup1_results_20260411_100643.csv"
COMPLETE_S2 = RESULTS_DIR / "complete_experiments" / "setup2_results_20260411_100643.csv"
BCD_CSV = PAPER_EXP / "arch_bcd_minimal.csv"
ROC_CSV = PAPER_EXP / "roc_det_data.csv"
IRIS_GAR_CSV = PAPER_EXP / "iris_gar_results.csv"
FIG_DIR = CODE_DIR.parent / "paper_ieee" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# === IEEE STYLE ===
COL_W = 3.5   # single column width (inches)
DBL_W = 7.16  # double column width (inches)

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

# === LABELS AND COLORS (IEEE colorblind-safe palette) ===
DB_MAP = {'fvc2002_1': 'DB1', 'fvc2002_2': 'DB2',
          'fvc2002_3': 'DB3', 'fvc2004_1': 'DB4'}
DBS = ['fvc2002_1', 'fvc2002_2', 'fvc2002_3', 'fvc2004_1']

# Wong colorblind-safe palette for data figures
PAL = {
    'Uni':  '#0072B2',  # blue
    'A':    '#D55E00',  # vermilion/red
    'IBFV': '#009E73',  # bluish green
    'B':    '#E69F00',  # orange
    'C':    '#CC79A7',  # reddish purple
    'D':    '#56B4E9',  # sky blue
}
M = {'Uni': 'o', 'A': 's', 'IBFV': '^', 'B': 'v', 'C': 'D', 'D': 'P'}
LS = {'Uni': '-', 'A': '--', 'IBFV': '-', 'B': ':', 'C': '-.', 'D': ':'}

# ============================================================================
# PART 1: ARCHITECTURAL DIAGRAM ENGINE — Grid-aligned, edge-connected arrows
# ============================================================================

# Diagram styling
DG_BOX_FC  = '#EAF2FB'   # light blue fill
DG_BOX_EC  = '#2C3E50'   # dark blue-gray edge
DG_BOX_LW  = 0.8
DG_DARK_FC = '#2C3E50'   # dark box fill
DG_DARK_TC = 'white'
DG_ACCENT  = '#D5E8D4'   # light green accent
DG_WARN_FC = '#FFF3CD'   # warning yellow fill
DG_WARN_EC = '#856404'   # warning yellow edge
DG_OK_FC   = '#D4EDDA'   # success green fill
DG_OK_EC   = '#155724'   # success green edge
DG_FAIL_FC = '#F8D7DA'   # fail red fill
DG_FAIL_EC = '#721C24'   # fail red edge
DG_ARR_C   = '#2C3E50'   # arrow color


class Box:
    """A positioned box with edge-point getters for precise arrow connections."""
    __slots__ = ('cx', 'cy', 'w', 'h')

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h

    @property
    def top(self):    return (self.cx, self.cy + self.h / 2)
    @property
    def bot(self):    return (self.cx, self.cy - self.h / 2)
    @property
    def left(self):   return (self.cx - self.w / 2, self.cy)
    @property
    def right(self):  return (self.cx + self.w / 2, self.cy)
    @property
    def top_left(self):  return (self.cx - self.w / 2, self.cy + self.h / 2)
    @property
    def top_right(self): return (self.cx + self.w / 2, self.cy + self.h / 2)
    @property
    def bot_left(self):  return (self.cx - self.w / 2, self.cy - self.h / 2)
    @property
    def bot_right(self): return (self.cx + self.w / 2, self.cy - self.h / 2)


def _fig(w, h):
    """Create figure with axis in inch-space, no decorations."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _box(ax, cx, cy, w, h, text, fc=None, ec=None, fs=7.5, fw='normal',
         tc='black', ls='-'):
    """Rounded rectangle with centered text. Returns a Box object."""
    if fc is None: fc = DG_BOX_FC
    if ec is None: ec = DG_BOX_EC
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle='round,pad=0.04',
                        facecolor=fc, edgecolor=ec, linewidth=DG_BOX_LW,
                        linestyle=ls, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight=fw, color=tc, zorder=3)
    return Box(cx, cy, w, h)


def _dia(ax, cx, cy, w, h, text, fc='white', ec=None, fs=7):
    """Diamond with centered text. Returns a Box (bounding box)."""
    if ec is None: ec = DG_BOX_EC
    pts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
    p = Polygon(pts, closed=True, fc=fc, ec=ec, lw=DG_BOX_LW, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, zorder=3)
    return Box(cx, cy, w, h)


def _arr(ax, pt_from, pt_to, lbl='', lo=(0, 0.04), color=None,
         style='->', lw=0.8, ls='-'):
    """Arrow between two (x,y) tuples with optional label."""
    if color is None: color = DG_ARR_C
    x1, y1 = pt_from
    x2, y2 = pt_to
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=0, shrinkB=0, linestyle=ls),
                zorder=1)
    if lbl:
        mx = (x1 + x2) / 2 + lo[0]
        my = (y1 + y2) / 2 + lo[1]
        ax.text(mx, my, lbl, ha='center', va='center', fontsize=6.5,
                style='italic', color='#444', zorder=4,
                bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85))


def _circ(ax, cx, cy, r, text, fc='#F8D7DA', ec='#C0392B', fs=6):
    """Small labeled circle (for attack points)."""
    c = plt.Circle((cx, cy), r, fc=fc, ec=ec, lw=0.6, zorder=4)
    ax.add_patch(c)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=ec, zorder=5)


def _lbl(ax, x, y, text, fs=8, fw='bold', ha='left'):
    """Section label like (a), (b)."""
    ax.text(x, y, text, fontsize=fs, fontweight=fw, ha=ha, va='center')


# ============================================================================
# PART 2: FIVE ARCHITECTURAL DIAGRAMS
# ============================================================================

def diagram_fuzzy_vault():
    """Fig 1: Fuzzy Vault Concept — single column, grid-aligned."""
    W, H = COL_W, 3.6
    fig, ax = _fig(W, H)
    mid = W / 2  # 1.75

    # --- (a) Vault Locking ---
    _lbl(ax, 0.08, 3.42, '(a) Vault Locking')

    # Row 1: inputs — two boxes centered on left/right halves
    bw, bh = 1.20, 0.28
    lx, rx = mid - 0.90, mid + 0.90
    b1 = _box(ax, lx, 3.05, bw, bh, 'Secret  $s$')
    b2 = _box(ax, rx, 3.05, bw, bh, 'Fingerprint')

    # Row 2: processing
    b3 = _box(ax, lx, 2.55, bw, bh, 'Polynomial $p(x)$')
    b4 = _box(ax, rx, 2.55, bw, 0.32, 'Minutiae Extraction\n& EC Quantization', fs=6.5)

    # Arrows row1 → row2 (bot of b1→top of b3, etc.)
    _arr(ax, b1.bot, b3.top)
    _arr(ax, b2.bot, b4.top)

    # Row 3: merge (wide box spanning both columns)
    mw = 2.9
    b5 = _box(ax, mid, 2.0, mw, 0.32,
              'Evaluate $p(x_i)$ on genuine $x$-coords  +  Add chaff', fs=6.8)
    _arr(ax, b3.bot, (b5.cx - 0.65, b5.top[1]))
    _arr(ax, b4.bot, (b5.cx + 0.65, b5.top[1]))

    # Row 4: output
    b6 = _box(ax, mid, 1.48, 1.7, bh, 'Stored Vault  $\\mathcal{V}$',
              fc=DG_DARK_FC, tc=DG_DARK_TC, fw='bold')
    _arr(ax, b5.bot, b6.top)

    # --- (b) Vault Unlocking ---
    _lbl(ax, 0.08, 1.10, '(b) Vault Unlocking')

    b7 = _box(ax, lx, 0.78, bw, bh, 'Query FP')
    b8 = _box(ax, rx, 0.78, 1.05, bh, 'Vault $\\mathcal{V}$')

    b9 = _box(ax, mid, 0.32, 2.7, 0.32,
              'Match  $\\to$  Lagrange on $(k{+}1)$-subsets  $\\to$  $H(\\hat{s}) =^{?} H(s)$',
              fs=6.5)
    _arr(ax, b7.bot, (b9.cx - 0.65, b9.top[1]))
    _arr(ax, b8.bot, (b9.cx + 0.65, b9.top[1]))

    ax.text(3.38, 0.32, '$\\to$ Accept\n     / Reject', fontsize=6.5, va='center')

    fig.savefig(FIG_DIR / 'fig_fuzzy_vault_concept.pdf')
    plt.close(fig)
    print("  [1/5] fig_fuzzy_vault_concept.pdf")


def diagram_and_vs_ibfv():
    """Fig 2: AND-Fusion Penalty vs IBFV Boost — single column, stacked."""
    W, H = COL_W, 4.2
    fig, ax = _fig(W, H)

    # === (a) AND-Fusion (Gating) ===
    _lbl(ax, 0.08, 4.02, '(a) AND-Fusion (Gating)')

    bh = 0.30
    # Horizontal flow: Iris Gate → Pass? → Decode Vault
    a1 = _box(ax, 0.62, 3.60, 0.95, bh, 'Iris Gate')
    d1 = _dia(ax, 1.65, 3.60, 0.48, 0.40, 'Pass?')
    a2 = _box(ax, 2.80, 3.60, 0.90, bh, 'Decode\nVault', fs=7)

    _arr(ax, a1.right, d1.left)
    _arr(ax, d1.right, a2.left, lbl='Yes', lo=(0, 0.10))

    # No path → REJECT below the diamond
    a3 = _box(ax, 1.65, 3.00, 1.25, bh, 'REJECT  (~46%)',
              fc=DG_FAIL_FC, ec=DG_FAIL_EC, fw='bold')
    _arr(ax, d1.bot, a3.top, lbl='No', lo=(0.18, 0), color=DG_FAIL_EC)

    # Result annotation
    ax.text(0.12, 2.55,
            'Result:  GAR$_{\\mathrm{AND}}$ = GAR$_{\\mathrm{FP}}$'
            ' $\\times$ GAR$_{\\mathrm{Iris}}$  $\\leq$  GAR$_{\\mathrm{FP}}$',
            fontsize=7, style='italic',
            bbox=dict(boxstyle='round,pad=0.15', fc=DG_WARN_FC, ec=DG_WARN_EC, lw=0.5))

    # Divider
    ax.plot([0.12, W - 0.12], [2.25, 2.25], color='#999', lw=0.4, ls='--')

    # === (b) IBFV (Boosting) ===
    _lbl(ax, 0.08, 2.08, '(b) IBFV (Boosting)')

    # Layout: two rows feeding right into a merge column then Lagrange
    #   Row 1 (top):   FP Decode ──────────────────────→ ┐
    #                                                      ├─→ Lagrange Decode
    #   Row 2 (bot):   Iris Key Recovery → Key OK? ─Yes─→ ┘
    #                                          └── No: +0

    y_top = 1.72   # FP row
    y_bot = 1.10   # Iris row
    y_mid = (y_top + y_bot) / 2  # Lagrange centered between the two

    c1 = _box(ax, 0.55, y_top, 0.82, bh, 'FP\nDecode', fs=7)
    c2 = _box(ax, 0.55, y_bot, 0.82, bh, 'Iris Key\nRecovery', fs=6.5)
    d2 = _dia(ax, 1.55, y_bot, 0.48, 0.40, 'Key\nOK?', fs=6.5)
    _arr(ax, c2.right, d2.left)

    # Yes branch: goes right with bonus label, then turns up to merge point
    mid_x = 2.35
    _arr(ax, d2.right, (mid_x, y_bot), color=DG_OK_EC)
    ax.text(mid_x + 0.02, y_bot + 0.10, '+$K$ bonus', fontsize=6.5,
            ha='center', color=DG_OK_EC, fontweight='bold')
    # Vertical segment up to Lagrange input height
    _arr(ax, (mid_x, y_bot), (mid_x, y_mid - 0.02), color=DG_OK_EC, style='->')

    # No branch: goes down-right as dashed
    ax.text(1.55, y_bot - 0.35, '+0 (no penalty)', fontsize=6, ha='center',
            color='#888', style='italic')
    _arr(ax, d2.bot, (1.55, y_bot - 0.27), color='#999', ls='--')

    # FP path: horizontal arrow to Lagrange
    c3 = _box(ax, 3.10, y_mid, 0.65, 0.34, 'Lagrange\nDecode', fs=6.5)
    _arr(ax, c1.right, (mid_x, y_top))
    _arr(ax, (mid_x, y_top), (mid_x, y_mid + 0.02))
    # Merge point → Lagrange
    _arr(ax, (mid_x, y_mid), c3.left)

    # Small merge dot
    ax.plot(mid_x, y_mid, 'o', color=DG_BOX_EC, markersize=4, zorder=5)

    # Result
    ax.text(0.12, 0.42,
            'Result:  GAR$_{\\mathrm{IBFV}}$  $\\geq$  '
            'GAR$_{\\mathrm{FP}}$   (iris can only help, never hurt)',
            fontsize=7, style='italic',
            bbox=dict(boxstyle='round,pad=0.15', fc=DG_OK_FC, ec=DG_OK_EC, lw=0.5))

    fig.savefig(FIG_DIR / 'fig_and_vs_ibfv.pdf')
    plt.close(fig)
    print("  [2/5] fig_and_vs_ibfv.pdf")


def diagram_enrollment():
    """Fig 3: IBFV Enrollment Pipeline — single column, vertical flow."""
    W, H = COL_W, 4.2
    fig, ax = _fig(W, H)
    mid = W / 2

    _lbl(ax, 0.08, 4.05, 'IBFV Enrollment')

    cx_fp = mid - 0.85  # 0.90
    cx_ir = mid + 0.85  # 2.60
    bw = 1.15
    bh = 0.28
    gap = 0.40  # vertical gap between box centers

    # Column headers
    ax.text(cx_fp, 3.82, 'Fingerprint Path', fontsize=7.5, ha='center',
            fontweight='bold', color=DG_BOX_EC)
    ax.text(cx_ir, 3.82, 'Iris Path', fontsize=7.5, ha='center',
            fontweight='bold', color=DG_BOX_EC)

    # FP column — equally spaced from y=3.55 down
    fp_labels = ['FP Image', 'Minutiae\nExtraction', 'Quantize ($f$)',
                 'EC Point Map', '$n$ Genuine\nPoints']
    fp_boxes = []
    for i, txt in enumerate(fp_labels):
        y = 3.55 - i * gap
        b = _box(ax, cx_fp, y, bw, bh, txt, fs=7)
        fp_boxes.append(b)
    for i in range(len(fp_boxes) - 1):
        _arr(ax, fp_boxes[i].bot, fp_boxes[i + 1].top)

    # Iris column — equally spaced, same grid
    ir_labels = ['Iris Image', 'Daugman\nEncoding', 'Fuzzy\nCommitment',
                 'PRF\n(SHA-256)', '$K$ Bonus\nPoints']
    ir_boxes = []
    for i, txt in enumerate(ir_labels):
        y = 3.55 - i * gap
        b = _box(ax, cx_ir, y, bw, bh, txt, fs=7)
        ir_boxes.append(b)
    for i in range(len(ir_boxes) - 1):
        _arr(ax, ir_boxes[i].bot, ir_boxes[i + 1].top)

    # Merge box
    merge = _box(ax, mid, 1.38, 2.9, 0.32,
                 'Genuine Pts ($n$)  +  Bonus Pts ($K$)  +  Chaff', fs=7)
    _arr(ax, fp_boxes[-1].bot, (merge.cx - 0.65, merge.top[1]))
    _arr(ax, ir_boxes[-1].bot, (merge.cx + 0.65, merge.top[1]))

    # Shuffle
    shuf = _box(ax, mid, 0.88, 1.8, bh, 'Shuffle & Store')
    _arr(ax, merge.bot, shuf.top)

    # Output vault
    vault = _box(ax, mid, 0.42, 2.1, 0.30,
                 'Vault  $\\mathcal{V}$  +  $H(s)$  +  $H(\\kappa)$',
                 fc=DG_DARK_FC, tc=DG_DARK_TC, fw='bold')
    _arr(ax, shuf.bot, vault.top)

    fig.savefig(FIG_DIR / 'fig_ibfv_enrollment.pdf')
    plt.close(fig)
    print("  [3/5] fig_ibfv_enrollment.pdf")


def diagram_verification():
    """Fig 4: IBFV Verification Pipeline — single column, vertical flow."""
    W, H = COL_W, 4.6
    fig, ax = _fig(W, H)
    mid = W / 2

    _lbl(ax, 0.08, 4.42, 'IBFV Verification')

    cx_fp = mid - 0.85
    cx_ir = mid + 0.85
    bw = 1.15
    bh = 0.28

    # Inputs row — y=4.05
    fp_in = _box(ax, cx_fp, 4.05, bw, bh, 'Query FP')
    ir_in = _box(ax, cx_ir, 4.05, bw, bh, 'Query Iris')
    v_in  = _box(ax, mid, 4.05, 0.58, bh, 'Vault\n$\\mathcal{V}$',
                 fc=DG_ACCENT, fs=6)

    # FP processing — y=3.55, y=3.05
    fp_q = _box(ax, cx_fp, 3.55, bw, bh, 'Quantize + EC Map', fs=7)
    _arr(ax, fp_in.bot, fp_q.top)
    fp_c = _box(ax, cx_fp, 3.05, bw, bh, 'FP Candidates')
    _arr(ax, fp_q.bot, fp_c.top)

    # Iris processing — y=3.55
    ir_r = _box(ax, cx_ir, 3.55, bw, bh, 'Iris Recovery')
    _arr(ax, ir_in.bot, ir_r.top)

    # Diamond y=2.95
    d1 = _dia(ax, cx_ir, 2.95, 0.55, 0.42, 'Key\nOK?', fs=7)
    _arr(ax, ir_r.bot, d1.top)

    # Yes path → bonus points y=2.35
    bonus = _box(ax, cx_ir, 2.35, bw, 0.32,
                 '+$K$ Bonus Points\n(~54% of attempts)',
                 fc=DG_OK_FC, ec=DG_OK_EC, fs=6.5)
    _arr(ax, d1.bot, bonus.top, lbl='Yes', lo=(0.20, 0), color=DG_OK_EC)

    # No path → fallback (dashed to left)
    _arr(ax, d1.left, (cx_fp + bw/2 + 0.03, d1.cy),
         lbl='No', lo=(0, 0.08), color='#999', ls='--')
    ax.text(mid, d1.cy - 0.16, 'FP only', fontsize=5.5, ha='center',
            color='#888', style='italic')

    # Combine row y=1.75
    comb = _box(ax, mid, 1.75, 2.6, 0.32,
                'Combine FP candidates + bonus (if any)', fs=7)
    _arr(ax, fp_c.bot, (comb.cx - 0.65, comb.top[1]))
    _arr(ax, bonus.bot, (comb.cx + 0.65, comb.top[1]), color=DG_OK_EC)

    # Lagrange y=1.22
    lag = _box(ax, mid, 1.22, 2.3, 0.30,
               'Lagrange Interpolation on $(k{+}1)$-subsets', fs=7)
    _arr(ax, comb.bot, lag.top)

    # Verify diamond y=0.65
    vd = _dia(ax, mid, 0.65, 0.65, 0.48, '$H(\\hat{s})$\n$=H(s)$?', fs=7)
    _arr(ax, lag.bot, vd.top)

    # Accept / Reject y=0.15
    acc = _box(ax, mid - 0.95, 0.15, 0.75, 0.24, 'Accept',
               fc=DG_OK_FC, ec=DG_OK_EC, fw='bold')
    rej = _box(ax, mid + 0.95, 0.15, 0.75, 0.24, 'Reject',
               fc=DG_FAIL_FC, ec=DG_FAIL_EC, fw='bold')
    _arr(ax, vd.left, acc.top, lbl='Yes', lo=(-0.12, 0.06), color=DG_OK_EC)
    _arr(ax, vd.right, rej.top, lbl='No', lo=(0.12, 0.06), color=DG_FAIL_EC)

    fig.savefig(FIG_DIR / 'fig_ibfv_verification.pdf')
    plt.close(fig)
    print("  [4/5] fig_ibfv_verification.pdf")


def diagram_threat_model():
    """Fig 5: Threat Model — full page width, clean simple style."""
    W, H = DBL_W, 2.4
    fig, ax = _fig(W, H)

    bw, bh = 1.10, 0.40

    # ---- Top row: User → Sensor → Feature Extract. → IBFV Engine ----
    y1 = 1.80
    gap_x = 1.60
    x0 = 0.75
    xs = [x0 + i * gap_x for i in range(4)]  # 0.75, 2.35, 3.95, 5.55
    top_labels = ['User', 'Sensor', 'Feature\nExtractor', 'IBFV\nEngine']
    top_boxes = []
    for x, lbl in zip(xs, top_labels):
        b = _box(ax, x, y1, bw, bh, lbl, fs=8)
        top_boxes.append(b)

    # Arrows between top-row boxes + AP labels above each link
    ap_top = ['AP-1', 'AP-2', 'AP-3', 'AP-4']
    # AP-1 at User (spoofing)
    ax.text(xs[0], y1 + bh/2 + 0.08, ap_top[0],
            fontsize=6, ha='center', color='#888', style='italic')
    for i in range(3):
        _arr(ax, top_boxes[i].right, top_boxes[i + 1].left)
        mx = (top_boxes[i].right[0] + top_boxes[i + 1].left[0]) / 2
        ax.text(mx, y1 + bh/2 + 0.08, ap_top[i + 1],
                fontsize=6, ha='center', color='#888', style='italic')

    # ---- Template Database — directly below IBFV Engine ----
    y2 = 0.85
    db_box = _box(ax, xs[3], y2, bw, bh, 'Template\nDatabase', fs=8)
    _arr(ax, top_boxes[3].bot, db_box.top)
    # AP-5 label on Engine→DB link
    ax.text(xs[3] + bw/2 + 0.08, (y1 + y2) / 2, 'AP-5',
            fontsize=6, ha='left', color='#888', style='italic')
    # AP-6 label next to Database
    ax.text(xs[3] + bw/2 + 0.08, y2, 'AP-6',
            fontsize=6, ha='left', color='#888', style='italic')

    # ---- Bottom row: Decision Module ← DB,  Application ← Decision ----
    dec_box = _box(ax, xs[2], y2, bw, bh, 'Decision\nModule', fs=8)
    app_box = _box(ax, xs[1], y2, bw, bh, 'Application', fs=8)

    _arr(ax, db_box.left, dec_box.right)
    ax.text((db_box.left[0] + dec_box.right[0]) / 2, y2 + bh/2 + 0.08,
            'AP-7', fontsize=6, ha='center', color='#888', style='italic')

    _arr(ax, dec_box.left, app_box.right)
    ax.text((dec_box.left[0] + app_box.right[0]) / 2, y2 + bh/2 + 0.08,
            'AP-8', fontsize=6, ha='center', color='#888', style='italic')

    # ---- Client / Server boundary ----
    bnd = (xs[1] + xs[2]) / 2
    ax.plot([bnd, bnd], [0.45, 2.15], color='#aaa', lw=0.6, ls=':')
    ax.text(bnd - 0.35, 0.38, 'Client', fontsize=7, ha='center', color='#999')
    ax.text(bnd + 0.35, 0.38, 'Server', fontsize=7, ha='center', color='#999')

    fig.savefig(FIG_DIR / 'fig_threat_model.pdf')
    plt.close(fig)
    print("  [5/5] fig_threat_model.pdf")


# ============================================================================
# PART 3: DATA LOADING
# ============================================================================

def load_complete_results():
    """Load Setup 1 + 2 results into list of dicts."""
    rows = []
    for fpath in [COMPLETE_S1, COMPLETE_S2]:
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found")
            continue
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


def load_bcd_results():
    """Load Architecture B/C/D results."""
    rows = []
    if not BCD_CSV.exists():
        return rows
    with open(BCD_CSV) as f:
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


def load_roc_data():
    """Load ROC/DET data."""
    data = defaultdict(lambda: defaultdict(list))
    if not ROC_CSV.exists():
        return data
    with open(ROC_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            db = row['database']
            setup = int(row['setup'])
            arch = row['architecture']
            far = float(row['far_pct'])
            frr = float(row['frr_pct'])
            gar = float(row['gar_pct'])
            data[(db, setup)][arch].append((far, frr, gar))
    return data


# ============================================================================
# PART 4: DATA FIGURES
# ============================================================================

def fig_arch_comparison():
    """Architecture comparison bar chart — all 6 architectures, log scale."""
    rows = load_complete_results()
    bcd_rows = load_bcd_results()

    best = defaultdict(lambda: defaultdict(lambda: 999.0))
    for r in rows:
        if r['setup'] == 1:
            db, arch, eer = r['database'], r['architecture'], r['eer_pct']
            if eer < best[db][arch]:
                best[db][arch] = eer
    for r in bcd_rows:
        if r['setup'] == 1:
            db, arch, eer = r['database'], r['architecture'], r['eer_pct']
            if eer < best[db][arch]:
                best[db][arch] = eer

    archs = ['Unimodal', 'A', 'B', 'C', 'D', 'IBFV']
    fills = [PAL['Uni'], PAL['A'], PAL['B'], PAL['C'], PAL['D'], PAL['IBFV']]
    hatches = ['', '', '', '', '', '']
    labels = {'Unimodal': 'Unimodal', 'A': 'A (AND)', 'B': 'B (Blind)',
              'C': 'C (Chaff)', 'D': 'D (AES)', 'IBFV': 'IBFV (Ours)'}

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    x = np.arange(len(DBS))
    w = 0.12

    for i, arch in enumerate(archs):
        vals = [max(best[db][arch], 0.0005) for db in DBS]
        offset = (i - len(archs)/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=labels[arch],
                      color=fills[i], hatch=hatches[i],
                      edgecolor='white', lw=0.3, alpha=0.9)
        for j, v in enumerate(vals):
            lbl = f'{v:.3f}' if v < 1 else f'{v:.1f}'
            ax.text(x[j] + offset, v * 1.25, lbl, ha='center', fontsize=4.5,
                    rotation=90, va='bottom')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([DB_MAP[db] for db in DBS])
    ax.set_ylabel('Best EER (%) — log scale')
    ax.legend(fontsize=5.5, ncol=3, loc='upper left',
              framealpha=0.9, handlelength=1.5)
    ax.set_ylim(0.0003, 60)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(FIG_DIR / 'arch_comparison_all.pdf')
    plt.close(fig)
    print("  arch_comparison_all.pdf")


def fig_eer_vs_factor():
    """EER vs Quantization Factor for DB3 and DB4 (Setup 1 & 2)."""
    rows = load_complete_results()

    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.5), sharex=True)
    targets = [
        ('fvc2002_3', 1), ('fvc2004_1', 1),
        ('fvc2002_3', 2), ('fvc2004_1', 2),
    ]

    for ax, (db, setup) in zip(axes.flat, targets):
        for arch, color, marker, label, ls in [
            ('Unimodal', PAL['Uni'], M['Uni'], 'Unimodal', '-'),
            ('A', PAL['A'], M['A'], 'Arch. A', '--'),
            ('IBFV', PAL['IBFV'], M['IBFV'], 'IBFV (Ours)', '-'),
        ]:
            factor_eer = defaultdict(lambda: 999)
            for r in rows:
                if r['database'] == db and r['setup'] == setup and r['architecture'] == arch:
                    if r['eer_pct'] < factor_eer[r['factor']]:
                        factor_eer[r['factor']] = r['eer_pct']
            if not factor_eer:
                continue
            factors = sorted(factor_eer.keys())
            eers = [factor_eer[f] for f in factors]
            ax.plot(factors, eers, marker=marker, color=color, label=label,
                    linewidth=1.2, markersize=3.5, linestyle=ls)

        ax.set_title(f'{DB_MAP[db]} — Setup {setup}', fontsize=8)
        ax.set_ylabel('Best EER (%)', fontsize=7)
        ax.set_xlabel('Factor $f$', fontsize=7)
        ax.legend(fontsize=5.5, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    fig.savefig(FIG_DIR / 'eer_vs_factor.pdf')
    plt.close(fig)
    print("  eer_vs_factor.pdf")


def fig_delta_eer_heatmap():
    """ΔEER (Unimodal − IBFV) heatmap for all DBs, Setup 2."""
    rows = load_complete_results()

    fig, axes = plt.subplots(1, 4, figsize=(COL_W, 2.2))

    for ax, db in zip(axes, DBS):
        factors = sorted(set(r['factor'] for r in rows if r['setup'] == 2 and r['database'] == db))
        degrees = sorted(set(r['degree'] for r in rows if r['setup'] == 2 and r['database'] == db))
        if not factors or not degrees:
            ax.set_title(DB_MAP[db], fontsize=7)
            continue

        lookup = {}
        for r in rows:
            if r['database'] == db and r['setup'] == 2:
                lookup[(r['architecture'], r['factor'], r['degree'])] = r['eer_pct']

        delta = np.zeros((len(degrees), len(factors)))
        for i, k in enumerate(degrees):
            for j, f in enumerate(factors):
                uni = lookup.get(('Unimodal', f, k), 0)
                ibfv = lookup.get(('IBFV', f, k), 0)
                delta[i, j] = uni - ibfv

        im = ax.imshow(delta, aspect='auto', cmap='YlGn', vmin=0,
                        vmax=max(3, np.max(delta)),
                        extent=[factors[0]-1, factors[-1]+1,
                                degrees[-1]+1, degrees[0]-1])
        ax.set_xticks(factors[::2])
        ax.set_yticks(degrees)
        ax.set_title(DB_MAP[db], fontsize=7)
        if db == DBS[0]:
            ax.set_ylabel('Degree $k$', fontsize=7)
        ax.set_xlabel('$f$', fontsize=7)
        ax.tick_params(labelsize=5)

        for i, k in enumerate(degrees):
            for j, f in enumerate(factors):
                val = delta[i, j]
                if val > 0:
                    ax.text(f, k, f'{val:.1f}', ha='center', va='center',
                            fontsize=3.5, color='black' if val < 2 else 'white')

    fig.suptitle('$\\Delta$EER (Unimodal $-$ IBFV) in pp — Setup 2',
                 fontsize=8, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'delta_eer_heatmap.pdf')
    plt.close(fig)
    print("  delta_eer_heatmap.pdf")


def fig_far_frr_crossover():
    """NEW: FAR/FRR crossover curves — standard biometric figure."""
    roc = load_roc_data()
    if not roc:
        print("  SKIP: far_frr_crossover (no ROC data)")
        return

    # For each DB (Setup 1), plot FAR and FRR vs degree k at best factor
    # First, find best factor per DB from complete results
    rows = load_complete_results()
    best_factor = {}
    for db in DBS:
        best_eer = 999
        best_f = 14
        for r in rows:
            if r['database'] == db and r['setup'] == 1 and r['architecture'] == 'Unimodal':
                if r['eer_pct'] < best_eer:
                    best_eer = r['eer_pct']
                    best_f = r['factor']
        best_factor[db] = best_f

    fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 1.8))

    for ax, db in zip(axes.flat, DBS):
        bf = best_factor[db]
        for arch, color, ls, marker, label in [
            ('Unimodal', PAL['Uni'], '-', M['Uni'], 'Unimodal'),
            ('IBFV', PAL['IBFV'], '-', M['IBFV'], 'IBFV'),
        ]:
            pts = []
            for r in rows:
                if (r['database'] == db and r['setup'] == 1
                        and r['architecture'] == arch and r['factor'] == bf):
                    pts.append((r['degree'], r['far_pct'], r['frr_pct']))
            if not pts:
                continue
            pts.sort()
            ks = [p[0] for p in pts]
            fars = [p[1] for p in pts]
            frrs = [p[2] for p in pts]

            ax.plot(ks, fars, marker=marker, color=color, linestyle=ls,
                    markersize=4, label=f'FAR ({label})')
            ax.plot(ks, frrs, marker=marker, color=color, linestyle='--',
                    markersize=4, label=f'FRR ({label})')

        ax.set_title(f'{DB_MAP[db]} ($f={bf}$)', fontsize=8)
        ax.set_xlabel('Degree $k$', fontsize=7.5)
        if db == DBS[0]:
            ax.set_ylabel('Error Rate (%)', fontsize=7.5)
        ax.legend(fontsize=5.5, loc='best', ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xticks([5, 7, 9, 11])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG_DIR / 'far_frr_crossover.pdf')
    plt.close(fig)
    print("  far_frr_crossover.pdf  [NEW]")


def fig_roc_curves():
    """ROC curves with AUC shading — both setups."""
    roc = load_roc_data()
    if not roc:
        print("  SKIP: roc_curves (no data)")
        return

    for setup in [1, 2]:
        fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 1.8))
        for ax, db in zip(axes, DBS):
            for arch, color, marker, label in [
                ('Unimodal', PAL['Uni'], M['Uni'], 'Unimodal'),
                ('IBFV', PAL['IBFV'], M['IBFV'], 'IBFV'),
            ]:
                pts = sorted(roc[(db, setup)][arch])
                if not pts:
                    continue
                fars = [p[0] for p in pts]
                gars = [p[2] for p in pts]

                if arch == 'IBFV':
                    ax.fill_between(fars, gars, alpha=0.12, color=color)

                ax.plot(fars, gars, marker=marker, color=color,
                        label=label, markersize=4, linewidth=1.3)

            ax.set_title(DB_MAP[db], fontsize=8)
            if db == DBS[0]:
                ax.set_ylabel('GAR (%)', fontsize=7.5)
            ax.set_xlabel('FAR (%)', fontsize=7.5)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6, loc='lower right')
            ax.set_xlim(-0.3, None)
            ax.set_ylim(None, 101)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.tight_layout(w_pad=0.8)
        fname = f'roc_curves_setup{setup}.pdf'
        fig.savefig(FIG_DIR / fname)
        plt.close(fig)
        print(f"  {fname}")


def fig_det_curves():
    """DET curves (FRR vs FAR, log-log) — both setups."""
    roc = load_roc_data()
    if not roc:
        print("  SKIP: det_curves (no data)")
        return

    for setup in [1, 2]:
        fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 1.8))
        for ax, db in zip(axes, DBS):
            for arch, color, marker, label in [
                ('Unimodal', PAL['Uni'], M['Uni'], 'Unimodal'),
                ('IBFV', PAL['IBFV'], M['IBFV'], 'IBFV'),
            ]:
                pts = sorted(roc[(db, setup)][arch])
                if not pts:
                    continue
                fars = [max(p[0], 0.001) for p in pts]
                frrs = [max(p[1], 0.001) for p in pts]
                ax.plot(fars, frrs, marker=marker, color=color,
                        label=label, markersize=4, linewidth=1.3)

            ax.set_title(DB_MAP[db], fontsize=8)
            ax.set_xscale('log')
            ax.set_yscale('log')
            if db == DBS[0]:
                ax.set_ylabel('FRR (%)', fontsize=7.5)
            ax.set_xlabel('FAR (%)', fontsize=7.5)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6, loc='best')
            ax.grid(True, alpha=0.25, linestyle='--', which='both')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.tight_layout(w_pad=0.8)
        fname = f'det_curves_setup{setup}.pdf'
        fig.savefig(FIG_DIR / fname)
        plt.close(fig)
        print(f"  {fname}")


def fig_iris_gar():
    """Iris key recovery GAR by database."""
    if not IRIS_GAR_CSV.exists():
        print("  SKIP: iris_gar (no data)")
        return

    gar_data = {}
    with open(IRIS_GAR_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            db = row['database']
            setup = int(row['setup'])
            gar_data[(db, setup)] = float(row['iris_gar_pct'])

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    x = np.arange(len(DBS))
    w = 0.30

    s1 = [gar_data.get((db, 1), 0) for db in DBS]
    s2 = [gar_data.get((db, 2), 0) for db in DBS]

    ax.bar(x - w/2, s1, w, label='Setup 1', color=PAL['Uni'], edgecolor='white', lw=0.3, alpha=0.9)
    ax.bar(x + w/2, s2, w, label='Setup 2', color=PAL['IBFV'], edgecolor='white', lw=0.3, alpha=0.9)

    for i, (v1, v2) in enumerate(zip(s1, s2)):
        ax.text(x[i] - w/2, v1 + 1.0, f'{v1:.1f}%', ha='center', fontsize=6)
        ax.text(x[i] + w/2, v2 + 1.0, f'{v2:.1f}%', ha='center', fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([DB_MAP[db] for db in DBS])
    ax.set_ylabel('Iris Key Recovery GAR (%)')
    ax.set_title('Iris Subsystem ($B_s = 255$)', fontsize=8)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 72)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(FIG_DIR / 'iris_gar.pdf')
    plt.close(fig)
    print("  iris_gar.pdf")


def fig_nbonus_ablation():
    """K-bonus ablation study."""
    # Confirmed data from paper table (DB3, f=16, k=5, Setup 1, B_s=255)
    ablation = {0: 4.281, 1: 4.281, 2: 3.755, 4: 2.181, 6: 1.742, 8: 1.742}

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ks = sorted(ablation.keys())
    eers = [ablation[k] for k in ks]

    ax.plot(ks, eers, marker='o', color=PAL['IBFV'], linewidth=1.5, markersize=5)

    # Annotate
    for kv, ev in zip(ks, eers):
        ax.annotate(f'{ev:.2f}', (kv, ev), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=6.5)

    # Mark saturation region
    ax.axhspan(1.6, 1.9, xmin=0.45, xmax=1.0, alpha=0.08, color='green')
    ax.text(6.5, 1.85, 'saturation', fontsize=6, color='#2e7d32',
            ha='center', style='italic')

    ax.set_xlabel('Number of Bonus Points $K$')
    ax.set_ylabel('EER (%)')
    ax.set_title('Ablation: EER vs.~$K$ (DB3, $f=16$, $k=5$, Setup 1)', fontsize=7.5)
    ax.set_xticks([0, 1, 2, 4, 6, 8])
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(FIG_DIR / 'nbonus_ablation.pdf')
    plt.close(fig)
    print("  nbonus_ablation.pdf")


def fig_arch_comparison_setup2():
    """Architecture comparison — Setup 2 (Unimodal, A, IBFV only)."""
    rows = load_complete_results()

    best = defaultdict(lambda: defaultdict(lambda: 999.0))
    for r in rows:
        if r['setup'] == 2:
            if r['eer_pct'] < best[r['database']][r['architecture']]:
                best[r['database']][r['architecture']] = r['eer_pct']

    archs = ['Unimodal', 'A', 'IBFV']
    fills = [PAL['Uni'], PAL['A'], PAL['IBFV']]
    hatches = ['', '', '']
    labels = {'Unimodal': 'Unimodal', 'A': 'Arch. A (AND)', 'IBFV': 'IBFV (Ours)'}

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    x = np.arange(len(DBS))
    w = 0.22

    for i, arch in enumerate(archs):
        vals = [best[db][arch] for db in DBS]
        offset = (i - len(archs)/2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=labels[arch],
               color=fills[i], hatch=hatches[i], edgecolor='white', lw=0.3)
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.15, f'{v:.2f}', ha='center', fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([DB_MAP[db] for db in DBS])
    ax.set_ylabel('Best EER (%)')
    ax.set_title('Architecture Comparison — Setup 2', fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(FIG_DIR / 'arch_comparison_setup2.pdf')
    plt.close(fig)
    print("  arch_comparison_setup2.pdf")


# ============================================================================
# MAIN
# ============================================================================

def fig_entropy_analysis():
    """Generate entropy analysis figure: 2x2 grid showing key entropy metrics."""
    # Load per-user KLD data
    kld_csv = PAPER_EXP / "entropy_per_user_kld.csv"
    gc_csv = PAPER_EXP / "entropy_genuine_vs_chaff.csv"

    kld_data = {}  # {db: {factor: {metric: value}}}
    with open(kld_csv) as f:
        for row in csv.DictReader(f):
            db = row['database']
            fac = int(row['factor'])
            if db not in kld_data:
                kld_data[db] = {}
            kld_data[db][fac] = {
                'peruser_kld': float(row['mean_peruser_kld']),
                'shannon': float(row['shannon_entropy']),
                'min_entropy': float(row['min_entropy']),
                'renyi2': float(row['renyi2_entropy']),
            }

    gc_data = {}  # {db: {factor: kld_gc}} for k=7
    with open(gc_csv) as f:
        for row in csv.DictReader(f):
            db = row['database']
            fac = int(row['factor'])
            deg = int(row['degree'])
            if deg != 7:
                continue
            if db not in gc_data:
                gc_data[db] = {}
            gc_data[db][fac] = float(row['mean_kld_genuine_chaff'])

    factors = [14, 16, 18, 20, 22, 24, 26, 28, 30]
    fig, axes = plt.subplots(2, 2, figsize=(DBL_W, 3.8))
    db_colors = {'fvc2002_1': '#0072B2', 'fvc2002_2': '#D55E00',
                 'fvc2002_3': '#009E73', 'fvc2004_1': '#E69F00'}
    db_markers = {'fvc2002_1': 'o', 'fvc2002_2': 's',
                  'fvc2002_3': '^', 'fvc2004_1': 'D'}

    # (a) Per-user KLD vs factor
    ax = axes[0, 0]
    for db in DBS:
        vals = [kld_data[db][f]['peruser_kld'] for f in factors]
        ax.plot(factors, vals, color=db_colors[db], marker=db_markers[db],
                markersize=4, linewidth=1.2, label=DB_MAP[db])
    ax.set_xlabel('Quantization factor $f$')
    ax.set_ylabel('$D_{\\mathrm{KL}}$ (bits)')
    ax.set_title('(a) Per-User KLD')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (b) Shannon entropy vs factor
    ax = axes[0, 1]
    for db in DBS:
        vals = [kld_data[db][f]['shannon'] for f in factors]
        ax.plot(factors, vals, color=db_colors[db], marker=db_markers[db],
                markersize=4, linewidth=1.2, label=DB_MAP[db])
    ax.axhline(y=8.0, color='gray', linestyle=':', linewidth=0.8, label='$\\log_2(256)$')
    ax.set_xlabel('Quantization factor $f$')
    ax.set_ylabel('Shannon $H$ (bits)')
    ax.set_title('(b) Shannon Entropy')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (c) Min-entropy vs factor
    ax = axes[1, 0]
    for db in DBS:
        vals = [kld_data[db][f]['min_entropy'] for f in factors]
        ax.plot(factors, vals, color=db_colors[db], marker=db_markers[db],
                markersize=4, linewidth=1.2, label=DB_MAP[db])
    ax.set_xlabel('Quantization factor $f$')
    ax.set_ylabel('Min-entropy $H_\\infty$ (bits)')
    ax.set_title('(c) Min-Entropy')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (d) Genuine vs Chaff KLD vs factor (k=7)
    ax = axes[1, 1]
    for db in DBS:
        vals = [gc_data[db][f] for f in factors]
        ax.plot(factors, vals, color=db_colors[db], marker=db_markers[db],
                markersize=4, linewidth=1.2, label=DB_MAP[db])
    ax.set_xlabel('Quantization factor $f$')
    ax.set_ylabel('$D_{\\mathrm{KL}}(\\mathrm{gen} \\| \\mathrm{chaff})$ (bits)')
    ax.set_title('(d) Genuine vs.\\ Chaff Indistinguishability')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    fig.savefig(FIG_DIR / 'entropy_analysis.pdf')
    plt.close(fig)
    print("  entropy_analysis.pdf")


def main():
    print("=" * 60)
    print("Generating ALL final figures for IBFV IEEE paper")
    print("=" * 60)

    print("\n--- Architectural Diagrams (single-column, 3.5in) ---")
    diagram_fuzzy_vault()
    diagram_and_vs_ibfv()
    diagram_enrollment()
    diagram_verification()
    diagram_threat_model()

    print("\n--- Data Figures (improved IEEE style) ---")
    fig_arch_comparison()
    fig_eer_vs_factor()
    fig_delta_eer_heatmap()
    fig_far_frr_crossover()
    fig_roc_curves()
    fig_det_curves()
    fig_iris_gar()
    fig_nbonus_ablation()
    fig_arch_comparison_setup2()
    fig_entropy_analysis()

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
