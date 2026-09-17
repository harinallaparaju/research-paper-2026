"""
Iris Stabilizer — Repetition Code Fuzzy Commitment with Rotation Compensation
and Bit Interleaving.

Converts noisy iris codes into stable cryptographic keys using blocked
repetition codes (majority voting) with rotation alignment.

Key innovation: BIT INTERLEAVING — instead of each block using consecutive
bits (vulnerable to spatially correlated errors from eyelid occlusion),
each block samples bits uniformly across the entire iris code using strided
indexing. Block i uses bits at positions [i, i+N, i+2N, ...] where N is
the number of blocks. This distributes spatially correlated errors across
all blocks so each block sees the average error rate (~26% FHD).

Protocol:
    Enrollment:
        1. Generate N random info bits (one per block)
        2. Create interleaved codeword: position j gets info_bits[j % N]
        3. Helper data: δ = enrolled_code ⊕ codeword
        4. Key = concatenation of all info bits
        5. Store (δ, key_hash, enrollment_mask)

    Verification:
        1. Find optimal rotation via masked FHD matching
        2. Rotate query code to align with enrollment
        3. Compute noisy_codeword = rotated_query ⊕ δ
        4. For each block i: majority vote on positions [i, i+N, i+2N, ...]
        5. Use mask to weight: only count bits where both masks are valid
        6. Concatenate → candidate key → verify SHA-256 hash

A block of size B with majority voting corrects up to (B-1)/2 errors
= 49.8% for B=255. With interleaving, each block sees the average 
genuine FHD (~26%), giving large margin (49.8% - 26% = 23.8%).

Reference:
    Hao, F., Anderson, R., & Daugman, J. (2006). Combining crypto with
    biometrics effectively. IEEE Trans. Computers, 55(9), 1081-1088.
"""

import numpy as np
import hashlib
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class IrisCommitment:
    """Stored enrollment data for iris fuzzy commitment."""
    helper_data: np.ndarray      # δ = enrolled_code ⊕ codeword (full code length)
    enrollment_mask: np.ndarray   # Enrollment validity mask (same length as code)
    block_size: int              # Repetition code block length B
    n_blocks: int                # Number of blocks
    key_hash: str                # SHA-256 of the info bits (for verification)
    key_length_bits: int         # Number of key bits


@dataclass
class StabilizationResult:
    """Result of iris key recovery."""
    success: bool
    key_bits: Optional[np.ndarray] = None
    key_hash: Optional[str] = None
    best_shift: int = 0
    block_decode_rate: float = 0.0   # Fraction of masked-majority-valid blocks


# Default repetition block size: B=1023 for maximum robustness
# 49152-bit code (2 wavelengths, 24 rows) → 48 blocks of 1023 → 48 key bits
# With ~56% combined mask → ~573 valid bits per block
# At FHD=0.40: d≈229, threshold=286 → Z=4.9, virtually guaranteed correct
DEFAULT_BLOCK_SIZE = 1023
# Width of normalized strip for rotation compensation
NORM_WIDTH = 512


class IrisStabilizer:
    """
    Repetition-code based iris stabilizer with rotation compensation.

    Uses majority voting on blocks of B bits. Each block encodes 1 info bit.
    Error correction capacity: (B-1)/(2*B) ≈ 46.7% for B=15.
    """

    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE):
        """
        Args:
            block_size: Repetition code block length (must be odd).
        """
        if block_size % 2 == 0:
            block_size += 1  # Must be odd for clean majority vote
        self.block_size = block_size
        self.threshold = block_size // 2  # Majority threshold

    def enroll(
        self,
        iris_code: np.ndarray,
        iris_mask: np.ndarray,
        seed: Optional[int] = None,
    ) -> IrisCommitment:
        """
        Enroll an iris code: generate helper data and key.

        Uses the FULL iris code (not just valid bits) so that recovery
        can use the same positions regardless of query mask.
        """
        if seed is not None:
            np.random.seed(seed)

        code_len = len(iris_code)
        n_blocks = code_len // self.block_size
        used_len = n_blocks * self.block_size

        # Generate random info bits (one per block)
        info_bits = np.random.randint(0, 2, n_blocks).astype(np.uint8)

        # Create INTERLEAVED codeword: position j belongs to block (j % n_blocks)
        # So codeword[j] = info_bits[j % n_blocks]
        # np.tile repeats the array B times: [b0,b1,...,bN-1, b0,b1,...,bN-1, ...]
        codeword = np.tile(info_bits, self.block_size)

        # Helper data: delta = enrolled_code[:used_len] XOR codeword
        helper = iris_code[:used_len].astype(np.uint8) ^ codeword

        key_hash = hashlib.sha256(info_bits.tobytes()).hexdigest()

        return IrisCommitment(
            helper_data=helper,
            enrollment_mask=iris_mask[:used_len].copy(),
            block_size=self.block_size,
            n_blocks=n_blocks,
            key_hash=key_hash,
            key_length_bits=n_blocks,
        )

    def recover(
        self,
        iris_code: np.ndarray,
        iris_mask: np.ndarray,
        commitment: IrisCommitment,
        max_shift: int = 32,
    ) -> StabilizationResult:
        """
        Attempt to recover the key from a query iris code.

        Tries multiple rotation shifts, decodes at each, checks key hash.
        Uses mask weighting: only bits valid in BOTH enrollment and query
        contribute to the majority vote.

        Rotation is applied on the FULL code (divisible by NORM_WIDTH),
        then truncated to used_len for decoding against helper data.
        """
        used_len = len(commitment.helper_data)
        full_len = len(iris_code)

        # Use full code length for rotation (must be divisible by NORM_WIDTH)
        if full_len % NORM_WIDTH != 0:
            # Fallback: no rotation compensation
            return self._decode_at_shift(
                iris_code[:used_len], iris_mask[:used_len], commitment, shift=0
            )

        n_segments = full_len // NORM_WIDTH

        # Reshape FULL code for rotation compensation
        query_2d = iris_code.reshape(n_segments, NORM_WIDTH)
        qmask_2d = iris_mask.reshape(n_segments, NORM_WIDTH)

        best_result = None

        for shift in range(-max_shift, max_shift + 1):
            # Rotate full query code, then truncate to used_len
            q_shifted = np.roll(query_2d, shift, axis=1).flatten()[:used_len]
            m_shifted = np.roll(qmask_2d, shift, axis=1).flatten()[:used_len]

            result = self._decode_at_shift(q_shifted, m_shifted, commitment, shift)

            if result.success:
                return result

            if best_result is None or result.block_decode_rate > best_result.block_decode_rate:
                best_result = result

        return best_result if best_result else StabilizationResult(success=False)

    def _decode_at_shift(
        self,
        query_code: np.ndarray,
        query_mask: np.ndarray,
        commitment: IrisCommitment,
        shift: int,
    ) -> StabilizationResult:
        """Decode at a specific rotation shift (vectorized)."""
        n_blocks = commitment.n_blocks
        used_len = n_blocks * commitment.block_size

        # Compute noisy codeword
        noisy = query_code[:used_len].astype(np.uint8) ^ commitment.helper_data[:used_len]

        # Combined mask
        combined_mask = (
            commitment.enrollment_mask[:used_len]
            & query_mask[:used_len]
        )

        # VECTORIZED majority vote over interleaved blocks
        # Reshape to (block_size, n_blocks) — interleaved layout means
        # position j belongs to block (j % n_blocks), so reshape as
        # (n_blocks, block_size) via Fortran order or reshape+transpose
        noisy_2d = noisy.reshape(commitment.block_size, n_blocks)
        mask_2d = combined_mask.reshape(commitment.block_size, n_blocks)

        # Count valid bits and ones per block (column sums)
        n_valid = mask_2d.sum(axis=0)                         # (n_blocks,)
        ones = (noisy_2d * mask_2d).sum(axis=0)               # (n_blocks,)

        # Majority vote: 1 if ones > n_valid/2, else 0
        info_bits = (ones > n_valid / 2).astype(np.uint8)
        # Blocks with no valid bits default to 0 (already 0 since 0 > 0/2 is False)

        valid_blocks = int((n_valid > 0).sum())
        decode_rate = valid_blocks / n_blocks if n_blocks > 0 else 0
        key_hash = hashlib.sha256(info_bits.tobytes()).hexdigest()
        success = (key_hash == commitment.key_hash)

        return StabilizationResult(
            success=success,
            key_bits=info_bits if success else None,
            key_hash=key_hash,
            best_shift=shift,
            block_decode_rate=decode_rate,
        )

    def derive_key(self, key_bits: np.ndarray, length: int = 256) -> bytes:
        """
        Derive a fixed-length cryptographic key from recovered iris key bits.

        Args:
            key_bits: Recovered info bits
            length: Desired key length in bits (default 256)

        Returns:
            Key bytes (length // 8 bytes)
        """
        return hashlib.sha256(key_bits.tobytes()).digest()[:length // 8]


# =========================================================================
# CONCATENATED CODE STABILIZER: Inner Repetition + Outer BCH
# =========================================================================

@dataclass
class IrisCommitmentBCH:
    """Stored enrollment data for concatenated code fuzzy commitment."""
    helper_data: np.ndarray      # δ = enrolled_code ⊕ codeword (full code length)
    enrollment_mask: np.ndarray   # Enrollment validity mask (same length as code)
    block_size: int              # Inner repetition block length B
    n_blocks: int                # Number of inner blocks total
    valid_block_mask: np.ndarray # Boolean mask: which blocks have enough valid bits
    bch_n: int                   # BCH codeword length (number of valid blocks used)
    bch_k: int                   # BCH message length (= actual key bits)
    bch_t: int                   # BCH error correction capability
    min_block_valid: int         # Minimum valid bits threshold per block
    key_hash: str                # SHA-256 of the actual key bits
    key_length_bits: int         # Number of actual key bits (= bch_k)


class IrisStabilizerBCH:
    """
    Concatenated code iris stabilizer: inner repetition + outer BCH.

    Architecture:
        Enrollment:
            1. Compute per-block validity from enrollment mask
            2. Select only blocks with ≥ min_block_valid valid bits
            3. Generate k random key bits → BCH-encode to n codeword bits
            4. Map codeword bits to valid blocks only; invalid blocks get 0
            5. Inner repetition with interleaving → helper data

        Recovery:
            1. Inner decode: majority vote on all blocks
            2. Extract only valid-block votes (using stored valid_block_mask)
            3. Outer BCH decode → k key bits → hash verify

    By only including blocks with enough valid bits in the BCH codeword,
    we eliminate the zero-valid-block problem that causes false errors.

    Uses bchlib (C-based) for fast BCH encode/decode (~0.002ms per decode).
    """

    MIN_BLOCK_VALID_DEFAULT = 5  # Minimum valid bits per block to be included

    def __init__(self, block_size: int = 63, bch_t: int = 3,
                 min_block_valid: int = None):
        if block_size % 2 == 0:
            block_size += 1
        self.block_size = block_size
        self.bch_t = bch_t
        self.min_block_valid = min_block_valid or self.MIN_BLOCK_VALID_DEFAULT
        self._bch = None
        self._bch_n = None
        self._bch_k_bits = None
        self._bch_m = None

    def _init_bch(self, n_valid_blocks: int):
        """Initialize BCH code for the given number of valid blocks."""
        import bchlib
        m = 1
        while (2**(m+1) - 1) <= n_valid_blocks:
            m += 1
        # bchlib supports m in [5, 15] and t up to a limit per m
        m = max(m, 5)
        m = min(m, 15)
        self._bch_m = m
        self._bch_n = 2**m - 1
        # Clamp t to a safe range for this m
        max_t = min(self.bch_t, 64)
        try:
            self._bch = bchlib.BCH(t=max_t, m=m)
        except RuntimeError:
            # Reduce t until it works
            for t_try in range(max_t - 1, 0, -1):
                try:
                    self._bch = bchlib.BCH(t=t_try, m=m)
                    break
                except RuntimeError:
                    continue
            else:
                raise ValueError(f"Cannot create BCH code for m={m}")
        self._bch_k_bits = self._bch_n - self._bch.ecc_bits
        return self._bch

    @staticmethod
    def _bits_to_bytes(bits: np.ndarray) -> bytes:
        n = len(bits)
        pad = (8 - n % 8) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        return np.packbits(bits).tobytes()

    @staticmethod
    def _bytes_to_bits(data: bytes, n_bits: int) -> np.ndarray:
        all_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return all_bits[:n_bits].astype(np.uint8)

    def enroll(
        self,
        iris_code: np.ndarray,
        iris_mask: np.ndarray,
        seed: Optional[int] = None,
    ) -> IrisCommitmentBCH:
        """Enroll an iris code with concatenated code protection."""
        code_len = len(iris_code)
        n_blocks_total = code_len // self.block_size
        used_len = n_blocks_total * self.block_size

        # Compute per-block enrollment-mask validity (interleaved layout)
        mask_2d = iris_mask[:used_len].reshape(self.block_size, n_blocks_total)
        block_valid_count = mask_2d.sum(axis=0)  # (n_blocks_total,)
        valid_block_mask = block_valid_count >= self.min_block_valid

        n_valid_blocks = int(valid_block_mask.sum())
        if n_valid_blocks < 8:
            raise ValueError(f"Too few valid blocks ({n_valid_blocks}) for BCH encoding")

        # Initialize BCH based on number of valid blocks
        self._init_bch(n_valid_blocks)
        bch_n = self._bch_n  # ≤ n_valid_blocks

        if seed is not None:
            np.random.seed(seed)

        k_bytes = self._bch_k_bits // 8
        key_data = np.random.bytes(k_bytes)
        key_bits = self._bytes_to_bits(key_data, k_bytes * 8)

        ecc = self._bch.encode(key_data)
        ecc_bits_arr = self._bytes_to_bits(ecc, self._bch.ecc_bits)

        # BCH codeword (bch_n bits)
        bch_codeword = np.zeros(bch_n, dtype=np.uint8)
        bch_codeword[:k_bytes * 8] = key_bits
        bch_codeword[k_bytes * 8 : k_bytes * 8 + self._bch.ecc_bits] = ecc_bits_arr

        # Map BCH codeword to valid blocks only
        # full_codeword[i] = bch_codeword[valid_index] for valid blocks
        # full_codeword[i] = 0 for invalid blocks (always default to 0)
        full_codeword = np.zeros(n_blocks_total, dtype=np.uint8)
        valid_indices = np.where(valid_block_mask)[0]
        # Use first bch_n valid blocks
        full_codeword[valid_indices[:bch_n]] = bch_codeword

        # Inner repetition with interleaving
        interleaved = np.tile(full_codeword, self.block_size)

        helper = iris_code[:used_len].astype(np.uint8) ^ interleaved
        key_hash = hashlib.sha256(key_data).hexdigest()

        return IrisCommitmentBCH(
            helper_data=helper,
            enrollment_mask=iris_mask[:used_len].copy(),
            block_size=self.block_size,
            n_blocks=n_blocks_total,
            valid_block_mask=valid_block_mask,
            bch_n=bch_n,
            bch_k=k_bytes * 8,
            bch_t=self.bch_t,
            min_block_valid=self.min_block_valid,
            key_hash=key_hash,
            key_length_bits=k_bytes * 8,
        )

    def recover(
        self,
        iris_code: np.ndarray,
        iris_mask: np.ndarray,
        commitment: IrisCommitmentBCH,
        max_shift: int = 32,
    ) -> StabilizationResult:
        """Attempt to recover the key using concatenated decoding."""
        if self._bch is None or self._bch_n != commitment.bch_n:
            n_valid = int(commitment.valid_block_mask.sum())
            self._init_bch(n_valid)

        used_len = len(commitment.helper_data)
        full_len = len(iris_code)

        if full_len % NORM_WIDTH != 0:
            return self._decode_at_shift(iris_code[:used_len], iris_mask[:used_len],
                                          commitment, shift=0)

        n_segments = full_len // NORM_WIDTH
        query_2d = iris_code.reshape(n_segments, NORM_WIDTH)
        qmask_2d = iris_mask.reshape(n_segments, NORM_WIDTH)

        best_result = None

        for shift in range(-max_shift, max_shift + 1):
            q_shifted = np.roll(query_2d, shift, axis=1).flatten()[:used_len]
            m_shifted = np.roll(qmask_2d, shift, axis=1).flatten()[:used_len]

            result = self._decode_at_shift(q_shifted, m_shifted, commitment, shift)

            if result.success:
                return result

            if best_result is None or result.block_decode_rate > best_result.block_decode_rate:
                best_result = result

        return best_result if best_result else StabilizationResult(success=False)

    def _decode_at_shift(
        self,
        query_code: np.ndarray,
        query_mask: np.ndarray,
        commitment: IrisCommitmentBCH,
        shift: int,
    ) -> StabilizationResult:
        """Decode at a specific rotation using concatenated decoding."""
        n_blocks = commitment.n_blocks
        bs = commitment.block_size
        used_len = n_blocks * bs

        # --- Inner decode: majority vote on ALL blocks ---
        noisy = query_code[:used_len].astype(np.uint8) ^ commitment.helper_data[:used_len]
        combined_mask = commitment.enrollment_mask[:used_len] & query_mask[:used_len]

        noisy_2d = noisy.reshape(bs, n_blocks)
        mask_2d = combined_mask.reshape(bs, n_blocks)

        n_valid = mask_2d.sum(axis=0)
        ones = (noisy_2d * mask_2d).sum(axis=0)

        # Majority vote per block
        all_voted = (ones > n_valid / 2).astype(np.uint8)

        # --- Extract only valid-block votes for BCH ---
        valid_indices = np.where(commitment.valid_block_mask)[0]
        bch_n = commitment.bch_n
        noisy_codeword = all_voted[valid_indices[:bch_n]]

        valid_blocks = int((n_valid[valid_indices[:bch_n]] > 0).sum())
        decode_rate = valid_blocks / bch_n if bch_n > 0 else 0

        # --- Outer BCH decode ---
        if self._bch is None:
            n_vb = int(commitment.valid_block_mask.sum())
            self._init_bch(n_vb)

        k_bytes = commitment.bch_k // 8
        ecc_bits = self._bch.ecc_bits

        data_bits = noisy_codeword[:k_bytes * 8]
        ecc_part = noisy_codeword[k_bytes * 8 : k_bytes * 8 + ecc_bits]

        data_bytes = bytearray(self._bits_to_bytes(data_bits))
        ecc_bytes = bytearray(self._bits_to_bytes(ecc_part))

        try:
            bitflips = self._bch.decode(bytes(data_bytes), bytes(ecc_bytes))

            if bitflips >= 0:
                self._bch.correct(data_bytes, ecc_bytes)
                key_data = bytes(data_bytes)
                key_hash = hashlib.sha256(key_data).hexdigest()
                success = (key_hash == commitment.key_hash)
                key_bits = self._bytes_to_bits(key_data, k_bytes * 8) if success else None
            else:
                success = False
                key_bits = None
                key_hash = None
        except Exception:
            success = False
            key_bits = None
            key_hash = None

        return StabilizationResult(
            success=success,
            key_bits=key_bits,
            key_hash=key_hash,
            best_shift=shift,
            block_decode_rate=decode_rate,
        )

    def derive_key(self, key_bits: np.ndarray, length: int = 256) -> bytes:
        """Derive a fixed-length cryptographic key from recovered bits."""
        return hashlib.sha256(key_bits.tobytes()).digest()[:length // 8]


# ---------- Convenience functions ----------

def create_iris_commitment(
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: Optional[int] = None,
) -> Tuple[IrisCommitment, IrisStabilizer]:
    """Create an iris commitment with the specified block size."""
    stabilizer = IrisStabilizer(block_size=block_size)
    commitment = stabilizer.enroll(iris_code, iris_mask, seed=seed)
    return commitment, stabilizer


def recover_iris_key(
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    commitment: IrisCommitment,
    stabilizer: IrisStabilizer,
    max_shift: int = 32,
) -> StabilizationResult:
    """Attempt to recover the iris key."""
    return stabilizer.recover(iris_code, iris_mask, commitment, max_shift=max_shift)


def create_iris_commitment_bch(
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    block_size: int = DEFAULT_BLOCK_SIZE,
    bch_t: int = 3,
    min_block_valid: int = None,
    seed: Optional[int] = None,
) -> Tuple[IrisCommitmentBCH, IrisStabilizerBCH]:
    """Create an iris commitment with concatenated inner-repetition + outer-BCH."""
    stabilizer = IrisStabilizerBCH(block_size=block_size, bch_t=bch_t,
                                    min_block_valid=min_block_valid)
    commitment = stabilizer.enroll(iris_code, iris_mask, seed=seed)
    return commitment, stabilizer


def recover_iris_key_bch(
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    commitment: IrisCommitmentBCH,
    stabilizer: IrisStabilizerBCH,
    max_shift: int = 32,
) -> StabilizationResult:
    """Attempt to recover the iris key with concatenated decoding."""
    return stabilizer.recover(iris_code, iris_mask, commitment, max_shift=max_shift)
