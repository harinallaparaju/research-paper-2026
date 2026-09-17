"""Diagnostic: why iris stabilizer still fails with interleaving."""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import load_chimeric_data
from iris_stabilizer import IrisStabilizer
from iris_extraction.matching import fractional_hamming_distance

subjects = load_chimeric_data('fvc2002_1', min_iris=2)

stabilizer = IrisStabilizer(block_size=255)

for i in range(10):
    s = subjects[i]
    iris0 = s['iris'].get(0)
    iris1 = s['iris'].get(1)
    if iris0 is None or iris1 is None:
        print(f'Subject {s["chimeric_id"]}: missing iris')
        continue

    code0, mask0 = iris0['code'], iris0['mask']
    code1, mask1 = iris1['code'], iris1['mask']

    # Compute FHD
    fhd = fractional_hamming_distance(code0, mask0, code1, mask1)

    # Mask stats
    mask0_pct = np.mean(mask0) * 100
    mask1_pct = np.mean(mask1) * 100
    combined = mask0 & mask1
    combined_pct = np.mean(combined) * 100

    # Enroll and attempt recovery
    commitment = stabilizer.enroll(code0, mask0, seed=42)
    result = stabilizer.recover(code1, mask1, commitment, max_shift=16)

    # Detailed block analysis at best shift
    used_len = commitment.n_blocks * commitment.block_size
    n_blocks = commitment.n_blocks
    full_len = len(code1)

    # Reshape FULL code and rotate
    n_segments = full_len // 512
    query_2d = code1.reshape(n_segments, 512)
    qmask_2d = mask1.reshape(n_segments, 512)
    q_shifted = np.roll(query_2d, result.best_shift, axis=1).flatten()[:used_len]
    m_shifted = np.roll(qmask_2d, result.best_shift, axis=1).flatten()[:used_len]

    noisy = q_shifted.astype(np.uint8) ^ commitment.helper_data
    combined_mask = commitment.enrollment_mask & m_shifted[:len(commitment.enrollment_mask)]

    # Analyze blocks (interleaved)
    block_errors = []
    block_valids = []
    for b in range(n_blocks):
        block = noisy[b::n_blocks]
        mask = combined_mask[b::n_blocks]
        n_valid = np.sum(mask)
        if n_valid > 0:
            ones = np.sum(block[mask.astype(bool)])
            err_rate = min(ones, n_valid - ones) / n_valid
        else:
            err_rate = 0.5
        block_errors.append(err_rate)
        block_valids.append(n_valid)

    block_errors = np.array(block_errors)
    block_valids = np.array(block_valids)

    n_failing = np.sum(block_errors > 0.5)
    n_borderline = np.sum((block_errors > 0.45) & (block_errors <= 0.5))

    print(f'Subject {s["chimeric_id"]:3d}: FHD={fhd:.3f} '
          f'mask0={mask0_pct:.0f}% mask1={mask1_pct:.0f}% combined={combined_pct:.0f}% '
          f'blocks_fail={n_failing}/{n_blocks} borderline={n_borderline} '
          f'mean_err={np.mean(block_errors):.3f} max_err={np.max(block_errors):.3f} '
          f'mean_valid={np.mean(block_valids):.0f} min_valid={np.min(block_valids)} '
          f'recovery={"OK" if result.success else "FAIL"} best_shift={result.best_shift}')
