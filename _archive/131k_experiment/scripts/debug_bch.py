#!/usr/bin/env python3
"""Debug self-match failure in BCH stabilizer."""

import numpy as np
import sys
import hashlib

sys.path.insert(0, ".")
from iris_stabilizer import IrisStabilizerBCH
from pathlib import Path

d = Path("..") / "data" / "Iris" / "iris_codes"

# S019_L_S1019L06 — decode_rate=0.9746, should work but doesn't
f = np.load(d / "S019_L_S1019L06.npz")
code, mask = f["code"], f["mask"]
print(f"Code len: {len(code)}, mask coverage: {mask.mean():.4f}")

bs = 255
stab = IrisStabilizerBCH(block_size=bs, bch_t=5)
comm = stab.enroll(code, mask, seed=42)
n_blocks = comm.n_blocks
used = n_blocks * bs

# Self-match manually
noisy = code[:used].astype(np.uint8) ^ comm.helper_data[:used]
combined = mask[:used] & comm.enrollment_mask[:used]

noisy_2d = noisy.reshape(bs, n_blocks)
mask_2d = combined.reshape(bs, n_blocks)
n_valid = mask_2d.sum(axis=0)
ones = (noisy_2d * mask_2d).sum(axis=0)
voted = (ones > n_valid / 2).astype(np.uint8)

# Reconstruct enrollment codeword
k_bytes = comm.bch_k // 8
np.random.seed(42)
key_data = np.random.bytes(k_bytes)
ecc = stab._bch.encode(key_data)

key_bits = stab._bytes_to_bits(key_data, k_bytes * 8)
ecc_bits_arr = stab._bytes_to_bits(ecc, stab._bch.ecc_bits)
codeword = np.zeros(n_blocks, dtype=np.uint8)
codeword[:k_bytes * 8] = key_bits
codeword[k_bytes * 8 : k_bytes * 8 + stab._bch.ecc_bits] = ecc_bits_arr

# Check if voted matches codeword
errors = (voted != codeword)
n_errors = int(errors.sum())
zero_valid_errs = int(((n_valid == 0) & errors).sum())
nonzero_valid_errs = int(((n_valid > 0) & errors).sum())
print(f"Total errors: {n_errors}/{n_blocks}")
print(f"  From zero-valid blocks: {zero_valid_errs}")
print(f"  From non-zero-valid blocks: {nonzero_valid_errs}")
print(f"Blocks with zero valid: {int((n_valid == 0).sum())}")

# Analysis: In self-match:
# noisy[j] = code[j] ^ helper[j] = code[j] ^ (code[j] ^ interleaved[j]) = interleaved[j]
# interleaved = np.tile(codeword, bs)
# noisy_2d[:, i] should all == codeword[i]
# So ones[i] = n_valid[i] * codeword[i]
# voted[i] = (n_valid[i] * codeword[i] > n_valid[i]/2)
#   if codeword[i]=1 and n_valid[i]>0: True
#   if codeword[i]=0 and n_valid[i]>0: False
#   if n_valid[i]=0: ones=0 > 0 is False → voted=0
#     if codeword[i]=1 → ERROR

zero_and_one = int(((n_valid == 0) & (codeword == 1)).sum())
print(f"\nZero-valid AND codeword=1: {zero_and_one}")
print(f"This should equal total errors: {n_errors}")
print(f"Match: {zero_and_one == n_errors}")

print(f"\n=== ROOT CAUSE CONFIRMED ===")
print(f"Blocks with 0 valid bits default to voted=0.")
print(f"If the BCH codeword bit is 1, that block becomes an error.")
print(f"~50% of zero-valid blocks will have codeword=1 → error.")
print(f"Solution: treat zero-valid blocks as erasures, not errors.")
