#!/usr/bin/env python3
"""Generate Figure 2 (AND-Fusion vs IBFV) and Figure 5 (D-IBFV Architecture).

Style: identical to fig_threat_model — uniform light-blue boxes, thin dark
borders, simple arrows, italic annotations.  Minimal, compact, no clutter.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------- paths ----------
FIG_DIR = Path(__file__).resolve().parent.parent / 'paper_extended' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- page geometry ----------
COL_W = 3.5   # single-column width (inches)
DBL_W = 7.16  # double-column width (inches)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# ---------- diagram palette (same as threat model) ----------
BOX_FC = '#EAF2FB'
BOX_EC = '#2C3E50'
BOX_LW = 0.8
ARR_C  = '#2C3E50'


# ---------- helpers ----------

class Box:
    __slots__ = ('cx', 'cy', 'w', 'h')
    def __init__(self, cx, cy, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
    @property
    def top(self):   return (self.cx, self.cy + self.h / 2)
    @property
    def bot(self):   return (self.cx, self.cy - self.h / 2)
    @property
    def left(self):  return (self.cx - self.w / 2, self.cy)
    @property
    def right(self): return (self.cx + self.w / 2, self.cy)


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def _box(ax, cx, cy, w, h, text, fs=8, fw='normal', fc=BOX_FC, ec=BOX_EC):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle='round,pad=0.04',
                        facecolor=fc, edgecolor=ec, linewidth=BOX_LW,
                        zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontweight=fw, color='black', zorder=3)
    return Box(cx, cy, w, h)


def _arr(ax, pt_from, pt_to, color=ARR_C, lw=0.8, ls='-'):
    ax.annotate('', xy=pt_to, xytext=pt_from,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=0, shrinkB=0, linestyle=ls),
                zorder=1)


def _note(ax, x, y, text, fs=6.5, ha='center', color='#666'):
    ax.text(x, y, text, fontsize=fs, ha=ha, va='center',
            style='italic', color=color, zorder=4)


# ============================================================================
# FIGURE 2: AND-Fusion vs IBFV  (single-column width)
# ============================================================================

def diagram_and_vs_ibfv():
    W, H = COL_W, 3.2
    fig, ax = _fig(W, H)

    bw, bh = 0.85, 0.34
    bw_sm = 0.68

    # ── (a) AND-Fusion (Gating) ──
    ax.text(0.08, 3.02, '(a) AND-Fusion (Gating)',
            fontsize=8, fontweight='bold')

    y_a = 2.62
    a1 = _box(ax, 0.55, y_a, bw, bh, 'Iris Gate', fs=7.5)
    a2 = _box(ax, 1.65, y_a, bw_sm, bh, 'Pass?', fs=7.5)
    a3 = _box(ax, 2.85, y_a, bw, bh, 'Decode\nVault', fs=7)
    _arr(ax, a1.right, a2.left)
    _arr(ax, a2.right, a3.left)
    _note(ax, (a2.right[0] + a3.left[0]) / 2, y_a + bh/2 + 0.08, 'Yes')

    y_rej = 2.08
    a4 = _box(ax, 1.65, y_rej, bw, bh, 'REJECT', fw='bold', fs=7.5)
    _arr(ax, a2.bot, a4.top)
    _note(ax, a2.cx + bw_sm/2 + 0.12, (a2.bot[1] + a4.top[1]) / 2,
          'No', ha='left')

    ax.text(0.08, 1.72,
            r'$\mathrm{GAR}_{\mathrm{AND}}'
            r' = \mathrm{GAR}_{\mathrm{FP}}'
            r' \times \mathrm{GAR}_{\mathrm{Iris}}'
            r' \;\leq\; \mathrm{GAR}_{\mathrm{FP}}$',
            fontsize=7, style='italic', color='#444')

    # ── Divider ──
    ax.plot([0.08, W - 0.08], [1.55, 1.55], color='#aaa', lw=0.5, ls='--')

    # ── (b) IBFV (Boosting) ──
    ax.text(0.08, 1.40, '(b) IBFV (Boosting)',
            fontsize=8, fontweight='bold')

    y_fp = 1.05
    y_ir = 0.52
    y_m  = (y_fp + y_ir) / 2

    b1 = _box(ax, 0.55, y_fp, bw, bh, 'FP Decode', fs=7.5)
    b2 = _box(ax, 0.55, y_ir, bw, bh, 'Iris Key\nRecovery', fs=7)
    b3 = _box(ax, 1.65, y_ir, bw_sm, bh, 'Key OK?', fs=7.5)
    _arr(ax, b2.right, b3.left)

    mx = 2.40

    # FP path -> merge
    _arr(ax, b1.right, (mx, y_fp))
    _arr(ax, (mx, y_fp), (mx, y_m + 0.04))

    # Iris success -> +K -> merge
    _arr(ax, b3.right, (mx, y_ir))
    _arr(ax, (mx, y_ir), (mx, y_m - 0.04))
    _note(ax, mx + 0.30, y_ir + 0.01, r'+$K$ bonus', color='#444')

    # Iris fail (dashed arrow only, no text to avoid overlap)
    _arr(ax, b3.bot, (b3.cx, y_ir - bh/2 - 0.04), color='#bbb', ls='--')

    # Merge dot -> Lagrange Decode
    ax.plot(mx, y_m, 'o', color=BOX_EC, markersize=3.5, zorder=5)
    b4 = _box(ax, 3.12, y_m, 0.70, bh, 'Lagrange\nDecode', fs=7)
    _arr(ax, (mx, y_m), b4.left)

    fig.savefig(FIG_DIR / 'fig_and_vs_ibfv.pdf')
    plt.close(fig)
    print('  [1/2] fig_and_vs_ibfv.pdf')


# ============================================================================
# FIGURE 5: D-IBFV Architecture  (double-column width)
#
# Design: 6 evenly-spaced columns (CIDs and n-shares become arrow labels).
# Two rows (enrollment, verification) + Accept/Reject below IBFV Verify.
# Layer boundary between Vault and Shamir.
# ============================================================================

def diagram_dibfv_architecture():
    W, H = DBL_W, 3.2
    fig, ax = _fig(W, H)

    bw, bh = 0.95, 0.38

    # 6 evenly spaced columns
    margin = 0.62
    step = (W - 2 * margin) / 5
    cols = [margin + i * step for i in range(6)]

    # ── ENROLLMENT (6 boxes, cols 0-5, all arrows left→right) ──
    ax.text(0.08, 2.98, 'Enrollment', fontsize=8, fontweight='bold')
    y1 = 2.62

    e_labels = ['Biometric\nInput', 'IBFV\nEnrollment', 'Vault\n(~16.6 KB)',
                '(t,n)-Shamir\nSplit', 'IPFS\nUpload', 'Blockchain\nStore']
    eb = [_box(ax, x, y1, bw, bh, lbl, fs=7)
          for x, lbl in zip(cols, e_labels)]
    for i in range(5):
        _arr(ax, eb[i].right, eb[i + 1].left)

    _note(ax, (cols[3] + cols[4]) / 2, y1 + bh/2 + 0.10, 'n shares')
    _note(ax, (cols[4] + cols[5]) / 2, y1 + bh/2 + 0.10, 'CIDs')

    # ── VERIFICATION ──
    # Correct order (all arrows left→right):
    # BC Lookup → IPFS Retrieve → Shamir Recon → Vault → IBFV Verify
    # Biometric Query feeds into IBFV Verify from above.
    ax.text(0.08, 1.92, 'Verification', fontsize=8, fontweight='bold')

    y_bq = 1.62       # Biometric Query (above IBFV Verify)
    y2   = 1.08        # main retrieval chain

    # Retrieval chain: 5 boxes at cols 0-4, all arrows left→right
    v_labels = ['Blockchain\nLookup', 'IPFS\nRetrieve', 'Shamir\nRecon.',
                'Vault', 'IBFV\nVerify']
    vb = [_box(ax, cols[i], y2, bw, bh, lbl, fs=7)
          for i, lbl in enumerate(v_labels)]
    for i in range(4):
        _arr(ax, vb[i].right, vb[i + 1].left)

    # Biometric Query above IBFV Verify
    bq = _box(ax, cols[4], y_bq, bw, bh, 'Biometric\nQuery', fs=7)
    _arr(ax, bq.bot, vb[4].top)

    # Arrow annotations
    _note(ax, (cols[0] + cols[1]) / 2, y2 + bh/2 + 0.10, 'CIDs')
    _note(ax, (cols[1] + cols[2]) / 2, y2 + bh/2 + 0.10, 't shares')
    _note(ax, cols[3], y2 - bh/2 - 0.12, 'erase after use',
          fs=6, color='#999')

    # Accept / Reject — plain bold text (no box)
    y_ar = y2 - bh/2 - 0.30
    _arr(ax, vb[4].bot, (cols[4], y_ar + 0.07))
    ax.text(cols[4], y_ar, 'Accept / Reject', ha='center', va='center',
            fontsize=7, fontweight='bold', color=BOX_EC, zorder=3)

    fig.savefig(FIG_DIR / 'fig_dibfv_architecture.pdf')
    plt.close(fig)
    print('  [2/2] fig_dibfv_architecture.pdf')


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print('Generating Figure 2 and Figure 5...')
    diagram_and_vs_ibfv()
    diagram_dibfv_architecture()
    print('Done.')
