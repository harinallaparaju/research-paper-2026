#!/usr/bin/env python3
"""
Model the concatenated code (inner repetition + outer BCH) performance.

Inner code: Repetition with majority vote (block_size B) — corrects ~49% per bit
Outer code: BCH(2047, k, t) — corrects t residual block errors
Net: much higher GAR with only modest key bit reduction
"""
import math
from scipy.stats import norm, binom

# ============================================
# Empirical data from iris_gar_bs_sweep.csv
# ============================================
# At Bs=63: GAR=54%, 2080 blocks → p_block_fail = 1 - 0.54^(1/2080) = 0.000296
# BUT this is averaged across 100 subjects. Let's model properly.

# The per-block failure probability is ~0.03% (from observed data).
# With N blocks, the number of failed blocks follows Binomial(N, p_fail).
# GAR = P(0 failures) = (1-p)^N

# With outer BCH(2047, k, t): GAR = P(failures <= t) = sum_{i=0}^{t} C(N,i) p^i (1-p)^(N-i)

observed = {
    63:  {"gar": 0.54, "n_blocks": 2080},
    31:  {"gar": 0.27, "n_blocks": 4228},
    127: {"gar": 0.66, "n_blocks": 1032},
    255: {"gar": 0.71, "n_blocks": 514},
    511: {"gar": 0.88, "n_blocks": 256},
}

print("=" * 80)
print("CONCATENATED CODE ANALYSIS: Inner Repetition + Outer BCH")
print("=" * 80)

# Back-calculate per-block failure rates
print("\n--- Per-block failure rates (from observed GAR) ---\n")
for bs, data in sorted(observed.items()):
    n = data["n_blocks"]
    gar = data["gar"]
    p = 1 - gar**(1/n)
    data["p_fail"] = p
    expected = p * n
    print(f"Bs={bs:4d}: N={n:5d} blocks, GAR={gar:.0%}, "
          f"p_fail={p:.6f}, E[errors]={expected:.2f}")

# BCH codes available for outer layer
# We need n_BCH >= n_blocks. For Bs=63: n_blocks=2080, use BCH(2047,k,t) + pad
# Actually 2080 > 2047, so we need to use 2047 blocks (trim 33 blocks)
# OR use larger BCH: n=4095 (m=12)
# OR better: just use first 2047 blocks from the 2080, giving 2047 info bits

print("\n--- Concatenated scheme: Rep(Bs=63) + BCH(2047,k,t) ---")
print(f"Use 2047 of 2080 blocks for BCH encoding (33 blocks unused/parity backup)")
print()

p_fail = observed[63]["p_fail"]
N = 2047  # BCH codeword length

print(f"{'BCH outer code':30} {'t':>3} {'Key bits':>10} {'GAR':>8} {'Security':>10}")
print("-" * 70)

for d, k, t in [
    (3, 2036, 1), (5, 2025, 2), (7, 2014, 3), (9, 2003, 4),
    (11, 1992, 5), (15, 1970, 7), (21, 1937, 10), (31, 1882, 15),
]:
    # GAR = P(X <= t) where X ~ Binom(N, p_fail)
    gar = binom.cdf(t, N, p_fail)
    security_bits = k  # actual secret key bits
    print(f"BCH(2047, {k}, t={t:2d}){'':10} {t:3d} {k:10d} {gar*100:7.2f}% {security_bits:10d}")

print()
print("--- Same analysis for Bs=31 (more blocks, higher error rate) ---")
# Bs=31: 4228 blocks, need BCH(4095, k, t) for m=12
p31 = observed[31]["p_fail"]
N31 = 4095
print(f"p_fail={p31:.6f}, N={N31}\n")

print(f"{'BCH outer code':30} {'t':>3} {'Key bits':>10} {'GAR':>8}")
print("-" * 60)
for t in [1, 2, 3, 5, 7, 10, 15]:
    gar = binom.cdf(t, N31, p31)
    # Approximate k: BCH(4095,k,t) overhead ≈ 12*t
    k_approx = N31 - 12*t
    print(f"BCH(4095, ~{k_approx}, t={t:2d}){'':9} {t:3d} {k_approx:10d} {gar*100:7.2f}%")

print()
print("=" * 80)
print("COMBINED PATH 1+2: Encoding improvement reduces FHD → lower p_fail → higher base GAR")
print("=" * 80)
print()

# If encoding improvement reduces genuine FHD from 0.26 to 0.20:
# The per-block failure rate would drop significantly
# Current: at Bs=63, each block has ~35 valid bits, mean error = 0.26
# Improved: mean error = 0.20
# Majority vote fails if > 50% of valid bits are wrong
# P(fail) = P(Binomial(35, 0.26) > 17) vs P(Binomial(35, 0.20) > 17)

for fhd_mean, label in [(0.26, "Current FHD=0.26"), (0.22, "Improved FHD=0.22"), 
                          (0.20, "Improved FHD=0.20"), (0.18, "Improved FHD=0.18")]:
    # Per-block failure: P(errors > B/2) where errors ~ Binom(valid_bits, fhd)
    valid_bits = 35  # average at Bs=63
    p_block = 1 - binom.cdf(valid_bits // 2, valid_bits, fhd_mean)
    n_blocks = 2047
    
    # Without outer BCH
    gar_no_bch = (1 - p_block) ** n_blocks
    
    # With BCH(2047, 1992, t=5)
    gar_bch_t5 = binom.cdf(5, n_blocks, p_block)
    
    # With BCH(2047, 1937, t=10) 
    gar_bch_t10 = binom.cdf(10, n_blocks, p_block)
    
    expected = p_block * n_blocks
    print(f"{label}: p_block={p_block:.8f}, E[errors]={expected:.3f}")
    print(f"  No outer BCH:     GAR = {gar_no_bch*100:.2f}%")
    print(f"  BCH(2047,1992,t=5):  GAR = {gar_bch_t5*100:.2f}% (1992 key bits)")
    print(f"  BCH(2047,1937,t=10): GAR = {gar_bch_t10*100:.2f}% (1937 key bits)")
    print()
