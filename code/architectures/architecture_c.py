"""
Architecture C: Iris-Seeded Dense Chaff.

The iris-derived stable key (via fuzzy commitment) seeds the PRNG for
chaff point generation, enabling a much denser chaff set. The vault
becomes practically infeasible to crack without the iris. With the iris
key, chaff points are regenerated and removed, reducing the problem to
standard fingerprint vault security.

Protocol:
    Lock (Enrollment):
        1. Quantize fingerprint → genuine x-values
        2. Generate polynomial, compute secret key, public key (standard)
        3. Evaluate legitimate vault points
        4. Enroll iris → get stable key K via fuzzy commitment
        5. Use K as PRNG seed → generate dense chaff (20× multiplier)
        6. Store: vault_points, iris_commitment, public_key, n_genuine

    Unlock (Verification):
        1. Recover iris key K' via fuzzy commitment
        2. Regenerate chaff using K' as PRNG seed
        3. Remove regenerated chaff from vault → reduced point set
        4. Match query fingerprint against reduced set
        5. Lagrange interpolation → recover polynomial → verify public key

Security Model:
    - Without iris: Attacker faces vault with 20× chaff (huge combinatorial
      security). Standard chaff multiplier is 5×; at 20×, the number of
      subsets to try is astronomically larger.
    - With iris (stolen) but not fingerprint: Can remove chaff, but all
      genuine points are exposed → polynomial is recoverable from k+1 points.
      This architecture does NOT provide independent iris+FP security.
    - With both: Remove chaff, match fingerprint, reconstruct polynomial.

Compared to Architecture B (Polynomial Blinding), this provides stronger
resistance against attackers without iris, but weaker against iris theft.
"""

import random
import hashlib
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
class MultimodalVaultC:
    """Architecture C vault data."""
    polynomial_degree: int
    public_key: Tuple[int, int]
    vault_points: List[Tuple[int, int]]  # genuine + dense chaff
    n_genuine: int
    n_chaff: int
    field_prime: int
    iris_commitment: IrisCommitment
    chaff_multiplier: int
    block_size: int
    factor: int  # Quantization factor (needed for unlock)


def _derive_chaff_seed(iris_key_hash: str) -> int:
    """Derive a deterministic PRNG seed from iris key hash."""
    return int(hashlib.sha256(
        ("chaff_seed:" + iris_key_hash).encode()
    ).hexdigest()[:16], 16)


def lock_vault_c(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    factor: int,
    degree: int,
    chaff_multiplier: int = 20,
    block_size: int = 1023,
    seed: Optional[int] = None,
) -> Optional[MultimodalVaultC]:
    """
    Create a multimodal vault using Architecture C.

    Args:
        minutiae_points: Enrollment minutiae (x, y, theta)
        iris_code: Enrollment iris code
        iris_mask: Enrollment iris mask
        factor: Quantization factor
        degree: Polynomial degree
        chaff_multiplier: Chaff points per genuine (default 20)
        block_size: Iris stabilizer block size
        seed: Random seed for polynomial generation

    Returns:
        MultimodalVaultC or None if enrollment fails
    """
    if seed is not None:
        random.seed(seed)

    quantized = quantize_minutiae(minutiae_points, factor)
    if len(quantized) < degree + 1:
        return None

    curve, G = _get_curve()
    p = curve.field.p

    # Generate polynomial (standard process)
    coefficients = [random.randint(10, 63) for _ in range(degree + 1)]
    binary_chunks = [f"{c:06b}" for c in coefficients]
    secret_key = int("".join(binary_chunks), 2) % p
    public_key_point = secret_key * G
    public_key = (public_key_point.x, public_key_point.y)

    # Generate genuine vault points
    vault_points = []
    for m in quantized:
        point = _ec_multiply(m)
        x = point.x
        y = _eval_poly(coefficients, x, p)
        vault_points.append((x, y))

    n_genuine = len(vault_points)

    # Enroll iris → stable key
    stabilizer = IrisStabilizer(block_size=block_size)
    iris_commitment = stabilizer.enroll(iris_code, iris_mask, seed=seed)

    # Use iris key hash as PRNG seed for chaff generation
    chaff_seed = _derive_chaff_seed(iris_commitment.key_hash)
    chaff_rng = random.Random(chaff_seed)

    # Generate dense chaff using iris-seeded PRNG
    n_chaff = n_genuine * chaff_multiplier
    chaff_count = 0
    while chaff_count < n_chaff:
        cx = chaff_rng.randint(1, p - 1)
        cy = chaff_rng.randint(1, p - 1)
        if _eval_poly(coefficients, cx, p) != cy:
            vault_points.append((cx, cy))
            chaff_count += 1

    # Shuffle vault points (using a separate iris-independent RNG)
    random.shuffle(vault_points)

    return MultimodalVaultC(
        polynomial_degree=degree,
        public_key=public_key,
        vault_points=vault_points,
        n_genuine=n_genuine,
        n_chaff=n_chaff,
        field_prime=p,
        iris_commitment=iris_commitment,
        chaff_multiplier=chaff_multiplier,
        block_size=block_size,
        factor=factor,
    )


def unlock_vault_c(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    vault: MultimodalVaultC,
    max_shift: int = 32,
) -> bool:
    """
    Attempt to unlock a multimodal vault (Architecture C).

    1. Recover iris key → regenerate chaff points → remove from vault
    2. Match query fingerprint against reduced vault
    3. Lagrange interpolation → verify secret key

    Returns:
        True if all steps succeed
    """
    import itertools

    # Step 1: Recover iris key
    stabilizer = IrisStabilizer(block_size=vault.block_size)
    result = stabilizer.recover(
        iris_code, iris_mask, vault.iris_commitment, max_shift=max_shift
    )

    if not result.success:
        return False  # Iris key recovery failed

    # Step 2: Regenerate chaff points using recovered key
    # In a 256-bit field, P(random point on degree-k poly) ≈ 0, so
    # all candidates were accepted during creation. Regenerate exactly.
    chaff_seed = _derive_chaff_seed(result.key_hash)
    chaff_rng = random.Random(chaff_seed)

    p = vault.field_prime
    n_chaff_expected = vault.n_genuine * vault.chaff_multiplier

    chaff_set = set()
    for _ in range(n_chaff_expected):
        cx = chaff_rng.randint(1, p - 1)
        cy = chaff_rng.randint(1, p - 1)
        chaff_set.add((cx, cy))

    # Step 3: Remove chaff from vault → only genuine points remain
    reduced_points = [(x, y) for x, y in vault.vault_points
                      if (x, y) not in chaff_set]

    # Step 4: Standard fingerprint vault unlock on reduced set
    curve, G = _get_curve()
    k = vault.polynomial_degree

    quantized = quantize_minutiae(minutiae_points, vault.factor)
    query_x_set = set()
    for m in quantized:
        point = _ec_multiply(m)
        query_x_set.add(point.x)

    matching = [(x, y) for x, y in reduced_points if x in query_x_set]

    if len(matching) < k + 1:
        return False

    for combo in itertools.combinations(matching, k + 1):
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
