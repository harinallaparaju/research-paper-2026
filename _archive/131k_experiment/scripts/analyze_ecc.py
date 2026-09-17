#!/usr/bin/env python3
"""Analyze error correction strategies for iris stabilizer."""
import math
from scipy.stats import norm

print("=== Root Cause: Why GAR=54% at Bs=63 ===\n")

GAR = 0.54
N_blocks = 2080  # 131072 // 63

p_block_fail = 1 - GAR**(1/N_blocks)
print(f"Per-block failure rate: {p_block_fail:.6f} ({p_block_fail*100:.4f}%)")
print(f"Expected failed blocks per attempt: {p_block_fail * N_blocks:.2f}")
print(f"Even 0.03% per-block failure × 2080 blocks = key hash broken\n")

print("=== Projected GAR for different code designs ===\n")
print(f"{'Design':42} {'Blocks':>7} {'Key bits':>9} {'p_fail':>10} {'GAR':>8}")
print("-" * 82)

codes = [
    ("Repetition Bs=63 (CURRENT)", 63, 1, 31),
    ("Repetition Bs=127", 127, 1, 63),
    ("Repetition Bs=255", 255, 1, 127),
    ("BCH(63,7,t=15) interleaved", 63, 7, 15),
    ("BCH(127,22,t=23) interleaved", 127, 22, 23),
    ("BCH(255,21,t=55) interleaved", 255, 21, 55),
    ("BCH(255,37,t=45) interleaved", 255, 37, 45),
]

for label, n_bits, k_bits, t_correct in codes:
    n_blocks = 131072 // n_bits
    total_key = n_blocks * k_bits
    valid_per_block = n_bits * 0.56  # average valid bits with ~56% mask
    mean_err = 0.26
    std_err = math.sqrt(mean_err * (1 - mean_err) / max(valid_per_block, 1))
    threshold = t_correct / n_bits
    z = (threshold - mean_err) / std_err if std_err > 0 else 999
    p_fail = 1 - norm.cdf(z)
    gar = (1 - p_fail) ** n_blocks
    print(f"{label:42} {n_blocks:7d} {total_key:9d} {p_fail:10.6f} {gar*100:7.2f}%")

print()
print("=== Key insight ===")
print("The problem is NOT per-block error correction (99.97% of blocks succeed)")
print("The problem is N_blocks=2080: needing ALL to succeed drops GAR to 54%")
print()
print("Solution: BCH(255,21,t=55) with interleaving:")
print("  - 514 blocks (4× fewer than current 2080)")
print("  - 21 info bits per block → 10,794 total key bits (vs 2080 current)")
print("  - t/n = 0.216 < 0.26 mean FHD... BUT with interleaving,")
print("    many bits are masked (invalid), so effective error rate on valid bits")
print("    determines success, not raw FHD")
print()

# More accurate model: account for masking
print("=== Accounting for masking ===\n")
# With masking, only valid bits contribute. BCH decoding uses ALL n bits,
# but masked bits are treated as erasures.
# For BCH(n,k,t): can correct t errors and s erasures if 2t+s <= n-k
# So erasures count as half an error
# 
# At 56% mask: 44% of bits are erasures
# For BCH(255,k,t): s = 0.44*255 = 112 erasures, need 2t+112 <= 255-k → t <= (143-k)/2

print("BCH with erasure decoding (56% mask, 44% erasures):")
print(f"{'BCH(n,k)':20} {'t_errors':>10} {'s_erasures':>12} {'2t+s':>6} {'n-k':>5} {'Feasible':>10}")
for n, k, t in [(255,21,55), (255,37,45), (255,45,43), (127,22,23), (63,7,15)]:
    s = int(0.44 * n)
    check = 2*t + s
    bound = n - k
    feasible = "YES" if check <= bound else "NO"
    print(f"BCH({n},{k}):{'':12} {t:10d} {s:12d} {check:6d} {bound:5d} {feasible:>10}")

print()
print("=== REVISED approach: don't use BCH on raw bits with erasures ===")
print("Instead: keep the MAJORITY VOTE (repetition code) per-bit,")
print("but use BCH as an OUTER code on the recovered bits")
print()
print("Two-level scheme:")
print("  Inner: Repetition with majority vote (block_size B, corrects ~49% per bit)")
print("  Outer: BCH on the N recovered bits (corrects remaining bit errors)")
print()

# Two-level: inner rep code at various block sizes, outer BCH
print("=== Two-level: Inner repetition + Outer BCH ===\n")

# At Bs=63: 2080 info bits from majority vote, p_fail_per_bit ≈ 0.03%
# At Bs=31: 4228 info bits, but higher p_fail_per_bit
# At Bs=15: 8738 info bits, even higher p_fail per bit

# Back-calculate p_fail_per_bit from observed GAR
for bs, gar_obs in [(63, 0.54), (31, 0.27), (127, 0.66), (255, 0.71), (511, 0.88)]:
    n_blocks = 131072 // bs
    if gar_obs > 0 and gar_obs < 1:
        p_fail = 1 - gar_obs**(1/n_blocks)
    else:
        p_fail = 0
    expected_errors = p_fail * n_blocks
    print(f"Bs={bs:4d}: n_blocks={n_blocks:5d}, GAR={gar_obs:.0%}, "
          f"p_fail/block={p_fail:.6f}, expected_errors={expected_errors:.1f}")

print()
print("If we add an outer BCH code to correct the ~0.6 expected block errors:")
print()

# With Bs=63: 2080 bits, ~0.6 errors on average
# BCH(2047, k, t) — need n >= 2080, closest is 2^11-1 = 2047... too small
# Or just use Reed-Solomon / BCH on groups

# Actually, the cleanest approach:
# Use Bs=31 (4228 info bits, 1.8 expected errors, but some have 3-5)
# Then BCH outer code with t=10 would catch almost all failures
# Net key bits: BCH(4228, 4228-11*10, t=10) ≈ 4228-110 = 4118 key bits
# But we can't do BCH(4228,...) — BCH needs n = 2^m - 1

# SIMPLER: Use Bs=63, get 2080 bits, pad to 2047 (BCH limit for m=11)
# or just use a different approach: concatenated code

# Actually the SIMPLEST effective approach:
# 1. Keep majority vote as-is (inner code)
# 2. Before hashing, apply BCH outer code to the info bits
# 3. During enrollment: generate k random key bits, BCH-encode to n=2080 bits, 
#    use those as the info bits for the repetition blocks
# 4. During recovery: majority-vote all blocks, BCH-decode the n-bit result
#    to get back the k key bits, then hash
# 
# BCH needs n = 2^m - 1. Closest to 2080: 2047 (m=11). Use first 2047 blocks.
# BCH(2047, k, t): for m=11, various (k,t) exist

import galois

print("=== BCH codes for n=2047 (m=11) ===")
# Try several
for t in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
    try:
        bch = galois.BCH(2047, t=t)
        print(f"  BCH(2047, {bch.k}, t={bch.t}) — key bits={bch.k}, overhead={2047-bch.k}")
    except Exception as e:
        print(f"  t={t}: {e}")
