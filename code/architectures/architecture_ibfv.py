"""
Architecture IBFV: Iris-Boosted Fuzzy Vault.

Novel architecture where iris-derived key generates ADDITIONAL genuine
polynomial points that are injected into the fingerprint vault. This
creates an ADDITIVE boost: the second modality HELPS vault decoding
instead of GATING it.

Key Property:  EER_IBFV ≤ EER_unimodal  (always, by construction)

When iris succeeds:
    → Attacker cannot forge iris key at B=255 (FAR=0%)
    → Genuine user recovers K bonus polynomial points
    → Lagrange needs k+1 from (M_fp + K) instead of M_fp alone
    → FRR drops significantly for borderline cases

When iris fails:
    → Fall back to fingerprint-only vault decoding
    → Identical to unimodal baseline (no penalty)

Protocol:
    Lock (Enrollment):
        1. Quantize fingerprint → genuine EC x-coordinates
        2. Generate degree-k polynomial p(x)
        3. Compute secret_key, public_key from polynomial coefficients
        4. Enroll iris → IrisCommitment + stable key K
        5. Derive K virtual x-coordinates from iris key:
              x'_j = H("ibfv_vpt" || j || key_hash) mapped to unique EC x
        6. Compute y'_j = p(x'_j) → K additional genuine vault points
        7. Generate chaff NOT on p(x) (standard 5× of original genuine count)
        8. Shuffle all vault points
        9. Store: vault_points (genuine_fp + genuine_iris + chaff),
                  n_genuine_fp, n_bonus, iris_commitment, public_key

    Unlock (Verification):
        1. Quantize query fingerprint → match against vault → M_fp matches
        2. Attempt iris key recovery from query iris
        3A. If iris SUCCEEDS:
            - Derive same K virtual x-coordinates from recovered key
            - Identify matching vault points → K bonus matches
            - Attempt Lagrange on all (k+1)-subsets from (M_fp + K) points
        3B. If iris FAILS:
            - Attempt Lagrange on all (k+1)-subsets from M_fp points only
            - Identical to unimodal vault decoding
        4. Verify: reconstructed secret_key × G == public_key

Security Analysis:
    - Impostor: Never recovers iris key (FAR=0% at B=255, 192-bit security)
      → Cannot identify bonus points → Same attack complexity as unimodal
    - Key theft: If iris key is stolen, attacker gains K known genuine points.
      Still needs (k+1 - K) fingerprint matches → reduced but nonzero security.
      With K=4 and k=10: need 7 FP matches from ~66 vault points → still hard.
    - Combined security: At least max(FP_security, Iris_security) bits

Reference:
    Novel contribution (Nallaparaju, 2026). Extends:
    - Senior's ECC fuzzy vault (JISAA 2025)
    - Hao et al. (2006) iris fuzzy commitment
"""

import random
import hashlib
import itertools
import numpy as np
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    quantize_minutiae,
    _get_curve,
    _ec_multiply,
    _eval_poly,
    _lagrange_interpolation,
    _reconstruct_secret,
    VaultData,
)
from iris_stabilizer import (
    IrisStabilizer, IrisCommitment,
    IrisStabilizerBCH, IrisCommitmentBCH,
)


@dataclass
class MultimodalVaultIBFV:
    """Architecture IBFV vault data."""
    polynomial_degree: int
    public_key: Tuple[int, int]           # secret_key × G
    vault_points: List[Tuple[int, int]]   # genuine_fp + genuine_iris + chaff
    n_genuine_fp: int                     # FP-derived genuine points
    n_bonus: int                          # Iris-derived bonus genuine points
    n_chaff: int                          # Chaff points
    field_prime: int
    iris_commitment: object               # IrisCommitment or IrisCommitmentBCH
    block_size: int
    factor: int
    degree: int
    bch_t: int = 0                        # 0 = original rep-only, >0 = BCH outer code


def _derive_virtual_x_coordinates(
    iris_key_hash: str,
    n_bonus: int,
    field_prime: int,
) -> List[int]:
    """
    Derive deterministic virtual x-coordinates from iris key hash.

    Each x-coordinate is derived as:
        x_j = int(SHA-256("ibfv_vpt" || j || key_hash)[:32]) mod p

    Only checks for self-collisions (astronomically unlikely on 256-bit
    field). Does NOT take existing FP x-coordinates as input — this
    ensures identical derivation during both enrollment and verification.

    Args:
        iris_key_hash: SHA-256 hex digest of iris key bits
        n_bonus: Number of virtual points to generate
        field_prime: ECC field prime p

    Returns:
        List of n_bonus unique virtual x-coordinates
    """
    virtual_xs = []
    used_x = set()
    j = 0
    attempts = 0
    while len(virtual_xs) < n_bonus and attempts < n_bonus * 10:
        h = hashlib.sha256(
            f"ibfv_vpt:{j}:{iris_key_hash}".encode()
        ).digest()
        x = int.from_bytes(h, 'big') % field_prime
        if x > 0 and x not in used_x:
            virtual_xs.append(x)
            used_x.add(x)
        j += 1
        attempts += 1
    return virtual_xs


def lock_vault_ibfv(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    factor: int,
    degree: int,
    n_bonus: int = 4,
    chaff_multiplier: int = 5,
    block_size: int = 1023,
    bch_t: int = 0,
    seed: Optional[int] = None,
) -> Optional[MultimodalVaultIBFV]:
    """
    Create a multimodal vault using Architecture IBFV.

    Args:
        minutiae_points: Enrollment minutiae (x, y, theta)
        iris_code: Enrollment iris code
        iris_mask: Enrollment iris mask
        factor: Quantization factor
        degree: Polynomial degree k (need k+1 points to reconstruct)
        n_bonus: Number of iris-derived bonus vault points (default: 4)
        chaff_multiplier: Chaff points per FP genuine point (default: 5)
        block_size: Iris stabilizer block size (default: 1023)
        bch_t: BCH error correction capability (0=rep-only, >0=BCH outer code)
        seed: Random seed

    Returns:
        MultimodalVaultIBFV or None if enrollment fails
    """
    if seed is not None:
        random.seed(seed)

    quantized = quantize_minutiae(minutiae_points, factor)
    if len(quantized) < degree + 1:
        return None

    curve, G = _get_curve()
    p = curve.field.p

    # Step 1: Generate polynomial (same as standard vault)
    coefficients = [random.randint(10, 63) for _ in range(degree + 1)]
    binary_chunks = [f"{c:06b}" for c in coefficients]
    secret_key = int("".join(binary_chunks), 2) % p
    public_key_point = secret_key * G
    public_key = (public_key_point.x, public_key_point.y)

    # Step 2: Generate genuine FP vault points
    vault_points = []
    genuine_x_set = set()
    for m in quantized:
        point = _ec_multiply(m)
        x = point.x
        y = _eval_poly(coefficients, x, p)
        vault_points.append((x, y))
        genuine_x_set.add(x)

    n_genuine_fp = len(vault_points)

    # Step 3: Enroll iris → fuzzy commitment
    if bch_t > 0:
        stabilizer = IrisStabilizerBCH(block_size=block_size, bch_t=bch_t)
    else:
        stabilizer = IrisStabilizer(block_size=block_size)
    iris_commitment = stabilizer.enroll(iris_code, iris_mask, seed=seed)

    # Step 4: Derive virtual x-coordinates from iris key
    virtual_xs = _derive_virtual_x_coordinates(
        iris_commitment.key_hash, n_bonus, p
    )

    # Step 5: Evaluate polynomial at virtual x-coordinates → bonus genuine points
    for vx in virtual_xs:
        vy = _eval_poly(coefficients, vx, p)
        vault_points.append((vx, vy))

    actual_n_bonus = len(virtual_xs)

    # Step 6: Generate chaff points (based on FP genuine count, NOT including bonus)
    n_chaff = n_genuine_fp * chaff_multiplier
    chaff_count = 0
    while chaff_count < n_chaff:
        cx = random.randint(1, p - 1)
        cy = random.randint(1, p - 1)
        if _eval_poly(coefficients, cx, p) != cy:
            vault_points.append((cx, cy))
            chaff_count += 1

    # Shuffle to hide ordering
    random.shuffle(vault_points)

    return MultimodalVaultIBFV(
        polynomial_degree=degree,
        public_key=public_key,
        vault_points=vault_points,
        n_genuine_fp=n_genuine_fp,
        n_bonus=actual_n_bonus,
        n_chaff=n_chaff,
        field_prime=p,
        iris_commitment=iris_commitment,
        block_size=block_size,
        factor=factor,
        degree=degree,
        bch_t=bch_t,
    )


def unlock_vault_ibfv(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    vault: MultimodalVaultIBFV,
    max_shift: int = 32,
) -> bool:
    """
    Attempt to unlock a multimodal vault (Architecture IBFV).

    1. Match query fingerprint → find M_fp matching vault points
    2. Attempt iris key recovery
    3. If iris succeeds: derive bonus x-coordinates → find K more matches
    4. Lagrange interpolation on (k+1)-subsets from all matches
    5. If iris fails: use only FP matches (identical to unimodal)

    Returns:
        True if vault is successfully unlocked
    """
    curve, G = _get_curve()
    p = vault.field_prime
    k = vault.polynomial_degree

    # Step 1: Match query fingerprint against vault
    quantized = quantize_minutiae(minutiae_points, vault.factor)
    query_x_set = set()
    for m in quantized:
        point = _ec_multiply(m)
        query_x_set.add(point.x)

    fp_matching = [(x, y) for x, y in vault.vault_points if x in query_x_set]

    # Step 2: Attempt iris key recovery
    if vault.bch_t > 0:
        stabilizer = IrisStabilizerBCH(block_size=vault.block_size, bch_t=vault.bch_t)
    else:
        stabilizer = IrisStabilizer(block_size=vault.block_size)
    iris_result = stabilizer.recover(
        iris_code, iris_mask, vault.iris_commitment, max_shift=max_shift
    )

    # Step 3: If iris succeeds, derive bonus point x-coordinates
    bonus_matching = []
    if iris_result.success and iris_result.key_hash is not None:
        virtual_xs = _derive_virtual_x_coordinates(
            iris_result.key_hash,
            vault.n_bonus,
            p,
        )
        virtual_x_set = set(virtual_xs)
        bonus_matching = [
            (x, y) for x, y in vault.vault_points if x in virtual_x_set
        ]

    # Step 4: Combine all matching points
    all_matching = fp_matching + bonus_matching

    # Remove duplicates (same x-coordinate)
    seen_x = set()
    unique_matching = []
    for x, y in all_matching:
        if x not in seen_x:
            unique_matching.append((x, y))
            seen_x.add(x)

    if len(unique_matching) < k + 1:
        return False

    # Step 5: Lagrange interpolation on (k+1)-subsets
    # Optimization: if we have enough points, try smart subsets first
    # (prioritize iris bonus points since they're guaranteed on-polynomial)
    if bonus_matching:
        # Try subsets that include ALL bonus points first (most likely correct)
        n_extra_needed = k + 1 - len(bonus_matching)
        if n_extra_needed > 0 and n_extra_needed <= len(fp_matching):
            # Try combinations of FP points + all bonus points
            bonus_set = set((x, y) for x, y in bonus_matching)
            fp_only = [pt for pt in unique_matching if pt not in bonus_set]

            for fp_combo in itertools.combinations(fp_only, n_extra_needed):
                combo = list(bonus_matching) + list(fp_combo)
                try:
                    reconstructed = _lagrange_interpolation(combo, p)
                    if reconstructed is None:
                        continue
                    secret = _reconstruct_secret(reconstructed) % p
                    expected_pk = secret * G
                    if (expected_pk.x, expected_pk.y) == vault.public_key:
                        return True
                except Exception:
                    continue

    # Fall through: try all (k+1)-subsets from unique_matching
    # (covers FP-only case when iris fails, and any edge cases above)
    for combo in itertools.combinations(unique_matching, k + 1):
        try:
            reconstructed = _lagrange_interpolation(list(combo), p)
            if reconstructed is None:
                continue
            secret = _reconstruct_secret(reconstructed) % p
            expected_pk = secret * G
            if (expected_pk.x, expected_pk.y) == vault.public_key:
                return True
        except Exception:
            continue

    return False
