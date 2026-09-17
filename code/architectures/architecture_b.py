"""
Architecture B: Polynomial Blinding (Iris-Keyed Coefficient Masking).

The iris-derived stable key blinds the fingerprint vault polynomial
coefficients using modular addition. This is the STRONGEST security
model: both modalities are independently required.

Protocol:
    Lock (Enrollment):
        1. Quantize fingerprint → genuine x-values
        2. Generate polynomial p(x) with 6-bit coefficients
        3. Compute secret_key and public_key from ORIGINAL coefficients
        4. Enroll iris → get stable key K via fuzzy commitment
        5. Derive 6-bit blinding values B_i from K for each coefficient
        6. Blind coefficients: c'_i = (c_i + B_i) mod 64
        7. Evaluate BLINDED polynomial p'(x) at genuine x-values
        8. Generate chaff not on p'(x)
        9. Store: vault_points, iris_commitment, public_key

    Unlock (Verification):
        1. Match query fingerprint against vault → find genuine points
        2. Lagrange interpolation → reconstruct BLINDED polynomial p'
        3. Recover iris key K' via fuzzy commitment
        4. Derive blinding values B'_i from K'
        5. Unblind: c_i = (c'_i - B'_i) mod 64
        6. Reconstruct secret key → verify public key

Security:
    - Without iris: can reconstruct p' but not unblind → wrong secret key
    - Without fingerprint: can compute blinding values but can't find
      genuine points → can't reconstruct polynomial
    - Requires BOTH modalities independently
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
from iris_stabilizer import IrisStabilizer, IrisCommitment


@dataclass
class MultimodalVaultB:
    """Architecture B vault data."""
    polynomial_degree: int
    public_key: Tuple[int, int]           # From ORIGINAL (unblinded) polynomial
    vault_points: List[Tuple[int, int]]   # On BLINDED polynomial
    n_genuine: int
    n_chaff: int
    field_prime: int
    iris_commitment: IrisCommitment
    block_size: int
    factor: int


def _derive_blinding_values(iris_key_hash: str, n_coefficients: int) -> List[int]:
    """
    Derive deterministic 6-bit blinding values from iris key hash.

    Uses HKDF-like derivation: SHA-256(key_hash || index) mod 64 for each
    coefficient index. This produces values in [0, 63].
    """
    values = []
    for i in range(n_coefficients):
        h = hashlib.sha256(
            (iris_key_hash + f":blind:{i}").encode()
        ).digest()
        values.append(int.from_bytes(h[:2], 'big') % 64)
    return values


def lock_vault_b(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    factor: int,
    degree: int,
    chaff_multiplier: int = 5,
    block_size: int = 1023,
    seed: Optional[int] = None,
) -> Optional[MultimodalVaultB]:
    """
    Create a multimodal vault using Architecture B.

    Args:
        minutiae_points: Enrollment minutiae (x, y, theta)
        iris_code: Enrollment iris code
        iris_mask: Enrollment iris mask
        factor: Quantization factor
        degree: Polynomial degree
        chaff_multiplier: Chaff points per genuine
        block_size: Iris stabilizer block size
        seed: Random seed for polynomial generation

    Returns:
        MultimodalVaultB or None if enrollment fails
    """
    if seed is not None:
        random.seed(seed)

    quantized = quantize_minutiae(minutiae_points, factor)
    if len(quantized) < degree + 1:
        return None

    curve, G = _get_curve()
    p = curve.field.p

    # Step 1: Generate ORIGINAL polynomial (same as standard vault)
    original_coefficients = [random.randint(10, 63) for _ in range(degree + 1)]
    binary_chunks = [f"{c:06b}" for c in original_coefficients]
    secret_key = int("".join(binary_chunks), 2) % p
    public_key_point = secret_key * G
    public_key = (public_key_point.x, public_key_point.y)

    # Step 2: Enroll iris → get stable key
    stabilizer = IrisStabilizer(block_size=block_size)
    iris_commitment = stabilizer.enroll(iris_code, iris_mask, seed=seed)

    # Step 3: Derive blinding values and blind coefficients
    blinding = _derive_blinding_values(iris_commitment.key_hash, degree + 1)
    blinded_coefficients = [
        (c + b) % 64 for c, b in zip(original_coefficients, blinding)
    ]

    # Step 4: Evaluate BLINDED polynomial at genuine x-values
    vault_points = []
    for m in quantized:
        point = _ec_multiply(m)
        x = point.x
        y = _eval_poly(blinded_coefficients, x, p)
        vault_points.append((x, y))

    n_genuine = len(vault_points)

    # Step 5: Generate chaff points not on the blinded polynomial
    n_chaff = n_genuine * chaff_multiplier
    chaff_count = 0
    while chaff_count < n_chaff:
        cx = random.randint(1, p - 1)
        cy = random.randint(1, p - 1)
        if _eval_poly(blinded_coefficients, cx, p) != cy:
            vault_points.append((cx, cy))
            chaff_count += 1

    random.shuffle(vault_points)

    return MultimodalVaultB(
        polynomial_degree=degree,
        public_key=public_key,
        vault_points=vault_points,
        n_genuine=n_genuine,
        n_chaff=n_chaff,
        field_prime=p,
        iris_commitment=iris_commitment,
        block_size=block_size,
        factor=factor,
    )


def unlock_vault_b(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    vault: MultimodalVaultB,
    max_shift: int = 32,
) -> bool:
    """
    Attempt to unlock a multimodal vault (Architecture B).

    1. Match query fingerprint → find genuine vault points
    2. Lagrange interpolation → reconstruct BLINDED polynomial
    3. Recover iris key → derive blinding values → unblind coefficients
    4. Reconstruct secret key → verify public key

    Returns:
        True if all steps succeed
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

    matching = [(x, y) for x, y in vault.vault_points if x in query_x_set]

    if len(matching) < k + 1:
        return False

    # Step 2: Recover iris key
    stabilizer = IrisStabilizer(block_size=vault.block_size)
    result = stabilizer.recover(
        iris_code, iris_mask, vault.iris_commitment, max_shift=max_shift
    )

    if not result.success:
        return False  # Can't unblind without iris key

    # Step 3: Derive blinding values from recovered key
    blinding = _derive_blinding_values(result.key_hash, k + 1)

    # Step 4: Try all (k+1)-subsets of matching points
    for combo in itertools.combinations(matching, k + 1):
        try:
            # Reconstruct BLINDED polynomial via Lagrange
            blinded_coeffs = _lagrange_interpolation(list(combo), p)
            if blinded_coeffs is None:
                continue

            # Unblind coefficients: c_i = (c'_i - B_i) mod 64
            original_coeffs = [
                (int(c) - b) % 64
                for c, b in zip(blinded_coeffs, blinding)
            ]

            # Reconstruct secret key from original coefficients
            secret = _reconstruct_secret(original_coeffs) % p
            expected_pk = secret * G

            if (expected_pk.x, expected_pk.y) == vault.public_key:
                return True
        except Exception:
            continue

    return False
