#!/usr/bin/env python3
"""Test the iris stabilizer with real CASIA iris codes."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_stabilizer import IrisStabilizer

codes_dir = Path(__file__).resolve().parent.parent / "data" / "Iris" / "iris_codes"

# Load a few codes from the same subject and different subjects
subjects = {}
for f in sorted(codes_dir.glob("*.npz"))[:50]:
    parts = f.stem.split("_")
    subj_id = parts[0]
    data = np.load(f)
    if subj_id not in subjects:
        subjects[subj_id] = []
    subjects[subj_id].append((f.name, data["code"], data["mask"]))

# Pick 2 subjects with at least 3 images each
test_subjects = [sid for sid, codes in subjects.items() if len(codes) >= 3][:2]
print(f"Testing with subjects: {test_subjects}")

# Initialize stabilizer: block_size=255 for robust averaging
stabilizer = IrisStabilizer(block_size=255)

# Enroll with first image of first subject
enroll_sid = test_subjects[0]
enroll_name, enroll_code, enroll_mask = subjects[enroll_sid][0]
print(f"\nEnrolling: {enroll_name}")
print(f"  Code length: {len(enroll_code)}")
print(f"  Valid bits: {np.sum(enroll_mask)} ({np.mean(enroll_mask)*100:.1f}%)")

commitment = stabilizer.enroll(enroll_code, enroll_mask, seed=42)
print(f"  Blocks: {commitment.n_blocks}")
print(f"  Key bits: {commitment.key_length_bits}")
print(f"  Key hash: {commitment.key_hash[:32]}...")

# Test genuine recovery with detailed diagnostics
print(f"\n--- Genuine Recovery Tests ---")
for name, code, mask in subjects[enroll_sid][1:4]:  # Just first 3
    result = stabilizer.recover(code, mask, commitment, max_shift=16)
    status = "OK" if result.success else "FAIL"
    print(f"  {name}: [{status}] shift={result.best_shift}")

    # Debug: check per-block error for best rotation
    from iris_extraction.matching import fractional_hamming_distance
    fhd = fractional_hamming_distance(enroll_code, enroll_mask, code, mask, max_shift=16)
    print(f"    FHD: {fhd:.4f}")

    # Check how many blocks differ at best shift
    used_len = len(commitment.helper_data)
    # Align to 512 for rotation, then truncate to used_len
    full_len = len(enroll_code)
    n_seg = full_len // 512
    q2d = code[:n_seg * 512].reshape(n_seg, 512)
    m2d = mask[:n_seg * 512].reshape(n_seg, 512)
    q_flat = np.roll(q2d, result.best_shift, axis=1).flatten()
    m_flat = np.roll(m2d, result.best_shift, axis=1).flatten()
    q_shifted = q_flat[:used_len]
    m_shifted = m_flat[:used_len]
    emask = enroll_mask[:used_len]
    combined = emask & m_shifted
    noisy = q_shifted.astype(np.uint8) ^ commitment.helper_data

    B = 255
    n_blocks = len(noisy) // B
    block_errors = []
    for i in range(n_blocks):
        s = i * B
        e = s + B
        bm = combined[s:e]
        nv = np.sum(bm)
        if nv > 0:
            ones = np.sum(noisy[s:e][bm.astype(bool)])
            err_rate = min(ones, nv - ones) / nv
            block_errors.append((nv, err_rate))
        else:
            block_errors.append((0, 0.5))

    valid_counts = [v for v, _ in block_errors]
    err_rates = [e for _, e in block_errors]
    print(f"    Valid bits/block: min={min(valid_counts)} max={max(valid_counts)} mean={np.mean(valid_counts):.1f}")
    print(f"    Block error rate: min={min(err_rates):.3f} max={max(err_rates):.3f} mean={np.mean(err_rates):.3f}")
    bad_blocks = sum(1 for _, e in block_errors if e > 0.45)
    print(f"    Blocks with >45% error: {bad_blocks}/{n_blocks}")

# Test impostor recovery (different subject)
print(f"\n--- Impostor Recovery Tests ---")
imp_sid = test_subjects[1]
for name, code, mask in subjects[imp_sid][:3]:
    result = stabilizer.recover(code, mask, commitment, max_shift=16)
    status = "OK" if result.success else "FAIL"
    print(f"  {name}: [{status}] shift={result.best_shift} decode_rate={result.block_decode_rate:.3f}")
