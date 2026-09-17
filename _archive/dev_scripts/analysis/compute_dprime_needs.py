"""Compute exactly what iris d' we need for guaranteed EER improvement."""
import numpy as np
from scipy import stats

imp_mean = 0.440
imp_std = 0.026

unimodal = [
    ('DB1 (FVC2002_1)', 0.000, 0.0, 0.000),
    ('DB2 (FVC2002_2)', 0.040, 0.0, 0.020),
    ('DB3 (FVC2002_3)', 0.260, 5.263, 2.762),
    ('DB4 (FVC2004_1)', 0.808, 5.0, 2.904),
]

scenarios = [
    ("d'=2.0 (CURRENT)", 0.260, 0.127),
    ("d'=2.5", 0.240, 0.10),
    ("d'=3.0", 0.220, 0.08),
    ("d'=3.5", 0.200, 0.07),
    ("d'=4.0", 0.180, 0.06),
]

print("=" * 90)
print("WHAT d' DO WE NEED FOR GUARANTEED EER IMPROVEMENT?")
print("=" * 90)

for dp_label, gm, gs in scenarios:
    print(f"\n--- {dp_label} (gen_mean={gm}, gen_std={gs}) ---")
    actual_dp = (imp_mean - gm) / np.sqrt((gs**2 + imp_std**2) / 2)
    print(f"  Actual d' = {actual_dp:.2f}")
    
    for db, fp_far, fp_frr, fp_eer in unimodal:
        best_eer = 999
        best_tau = 0
        best_far = 0
        best_frr = 0
        for tau_100 in range(20, 50):
            tau = tau_100 / 100
            gen_pass = stats.norm.cdf(tau, gm, gs)
            imp_pass = stats.norm.cdf(tau, imp_mean, imp_std)
            # AND gate: both must pass
            combined_far = fp_far * imp_pass  # P(FP accept AND iris accept)
            combined_frr = 100 - (100 - fp_frr) * gen_pass  # reject if either rejects
            combined_eer = (combined_far + combined_frr) / 2
            if combined_eer < best_eer:
                best_eer = combined_eer
                best_tau = tau
                best_far = combined_far
                best_frr = combined_frr
        
        if fp_eer == 0:
            delta = "N/A (baseline=0)"
        elif best_eer < fp_eer:
            delta = f"BETTER by {fp_eer - best_eer:.3f}%"
        else:
            delta = f"WORSE by {best_eer - fp_eer:.3f}%"
        
        print(f"  {db}: Uni={fp_eer:.3f}% → Multi={best_eer:.3f}% (τ={best_tau:.2f}, FAR={best_far:.4f}%, FRR={best_frr:.2f}%) [{delta}]")

# Now compute Architecture B/D potential at each d'
print("\n" + "=" * 90)
print("STABILIZER-BASED ARCHITECTURES (B/D) — ESTIMATED FRR AT DIFFERENT d' VALUES")
print("Block size B=255 → 192 blocks, correction capacity 49.8%")
print("Genuine key recovery fails when block BER > 49.8%")
print("=" * 90)

for dp_label, gm, gs in scenarios:
    # Per-block BER = genuine FHD at that d'
    # With interleaving, each block sees the average FHD
    # P(block fails) = P(block BER > 49.8%) 
    # For B=255 with ~143 valid bits per block,
    # threshold = 127.5 errors. At genuine FHD = gm,
    # expected errors = 143*gm, std = sqrt(143*gm*(1-gm))
    n_valid = 143  # ~56% mask × 255
    threshold = 127  # majority threshold
    expected_errors = n_valid * gm
    std_errors = np.sqrt(n_valid * gm * (1 - gm))
    p_block_fail = 1 - stats.norm.cdf(threshold, expected_errors, std_errors)
    n_blocks = 192
    p_key_recover = (1 - p_block_fail) ** n_blocks
    key_frr = (1 - p_key_recover) * 100
    print(f"  {dp_label}: BER={gm:.3f}, P(block fail)={p_block_fail:.6f}, FRR≈{key_frr:.1f}%")
