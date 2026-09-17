#!/usr/bin/env python3
"""
Sweep BCH parameters: block_size, bch_t, min_block_valid.
Find the configuration that maximizes GAR.
"""

import sys, os
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from iris_stabilizer import IrisStabilizerBCH, IrisStabilizer, create_iris_commitment_bch, recover_iris_key_bch

CODE_DIR = Path(__file__).parent.parent / "data" / "Iris" / "iris_codes"
MAX_SUBJECTS = 50
MAX_SHIFT = 8

def load_iris_codes(code_dir, max_subjects=0):
    subjects = defaultdict(list)
    for f in sorted(code_dir.glob("*.npz")):
        data = np.load(f)
        subj = f.stem.split("_")[0]
        subjects[subj].append((data["code"], data["mask"], f.stem))
    subj_list = sorted(subjects.keys())
    if max_subjects > 0:
        subj_list = subj_list[:max_subjects]
    return {s: subjects[s] for s in subj_list}

def evaluate(subjects, bs, bch_t, min_bv):
    n_gen = 0
    n_ok = 0
    n_enrolled = 0
    for subj, samples in subjects.items():
        if len(samples) < 2:
            continue
        c_e, m_e, _ = samples[0]
        try:
            stab = IrisStabilizerBCH(block_size=bs, bch_t=bch_t, min_block_valid=min_bv)
            comm = stab.enroll(c_e, m_e, seed=42)
        except (ValueError, RuntimeError):
            continue
        n_enrolled += 1
        for c_q, m_q, _ in samples[1:]:
            r = stab.recover(c_q, m_q, comm, max_shift=MAX_SHIFT)
            n_gen += 1
            if r.success:
                n_ok += 1
    gar = n_ok / n_gen * 100 if n_gen > 0 else 0
    return gar, n_ok, n_gen, n_enrolled

def main():
    subjects = load_iris_codes(CODE_DIR, MAX_SUBJECTS)
    n_total = sum(len(v) for v in subjects.values())
    print(f"Loaded {n_total} codes from {len(subjects)} subjects\n")

    # Also get original baselines
    print("=== Original (no BCH) baselines ===")
    for bs in [63, 127, 255, 511, 1023]:
        stab = IrisStabilizer(block_size=bs)
        n_gen = n_ok = 0
        for samples in subjects.values():
            if len(samples) < 2: continue
            c_e, m_e, _ = samples[0]
            comm = stab.enroll(c_e, m_e, seed=42)
            for c_q, m_q, _ in samples[1:]:
                r = stab.recover(c_q, m_q, comm, max_shift=MAX_SHIFT)
                n_gen += 1
                if r.success: n_ok += 1
        gar = n_ok / n_gen * 100 if n_gen > 0 else 0
        print(f"  Bs={bs:4d}: GAR={gar:.2f}% ({n_ok}/{n_gen})")

    print("\n=== BCH sweep ===")
    print(f"{'Bs':>4s} {'t':>3s} {'mbv':>4s} | {'GAR':>7s} {'ok/gen':>10s} {'enrolled':>8s}")
    print("-" * 50)

    results = []
    for bs in [127, 255, 511]:
        for bch_t in [3, 5, 10]:
            for min_bv in [1, 5, 10, 20]:
                gar, n_ok, n_gen, n_enrolled = evaluate(subjects, bs, bch_t, min_bv)
                print(f"{bs:4d} {bch_t:3d} {min_bv:4d} | {gar:6.2f}% {n_ok:4d}/{n_gen:<4d}  {n_enrolled:4d}")
                results.append((gar, bs, bch_t, min_bv, n_ok, n_gen))

    results.sort(reverse=True)
    print(f"\n=== Top 5 ===")
    for gar, bs, t, mbv, ok, gen in results[:5]:
        print(f"  Bs={bs}, t={t}, min_valid={mbv}: GAR={gar:.2f}% ({ok}/{gen})")

if __name__ == "__main__":
    main()
