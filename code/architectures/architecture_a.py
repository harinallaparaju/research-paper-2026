"""
Architecture A: Decision-Level AND Fusion.

Multimodal vault that requires BOTH fingerprint vault unlock AND iris
FHD below threshold for successful authentication.

Protocol:
    Lock (Enrollment):
        1. Create standard fingerprint fuzzy vault from impression 1
        2. Compute and store enrollment iris code + mask
        3. Store iris FHD threshold τ

    Unlock (Verification):
        1. Check iris FHD(enrolled, query) < τ
        2. If iris passes: attempt fingerprint vault unlock
        3. If both pass: accept. Otherwise: reject.

Security: An attacker must compromise BOTH modalities.
    - If they have a stolen fingerprint but wrong iris: rejected at step 1
    - If they have a stolen iris but wrong fingerprint: rejected at step 2

This is the most straightforward multimodal fusion and should provide
the best security-FRR tradeoff since iris matching is more reliable
than iris key extraction.
"""

import numpy as np
from typing import Set, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    quantize_minutiae,
    create_vault,
    unlock_vault,
    VaultData,
)
from iris_extraction.matching import fractional_hamming_distance


@dataclass
class MultimodalVaultA:
    """Architecture A vault data."""
    fp_vault: VaultData                   # Standard fingerprint fuzzy vault
    iris_code: np.ndarray                 # Enrollment iris code
    iris_mask: np.ndarray                 # Enrollment iris mask
    iris_threshold: float                 # FHD threshold τ
    factor: int                           # Quantization factor
    degree: int                           # Polynomial degree


def lock_vault_a(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    factor: int,
    degree: int,
    iris_threshold: float = 0.35,
    seed: Optional[int] = None,
) -> Optional[MultimodalVaultA]:
    """
    Create a multimodal vault using Architecture A.

    Args:
        minutiae_points: List of (x, y, theta) tuples
        iris_code: Enrollment iris code (1D binary array)
        iris_mask: Enrollment mask (1D binary array)
        factor: Minutiae quantization factor
        degree: Polynomial degree
        iris_threshold: FHD threshold for iris acceptance
        seed: Random seed

    Returns:
        MultimodalVaultA or None if enrollment fails
    """
    quantized = quantize_minutiae(minutiae_points, factor)
    if len(quantized) < degree + 1:
        return None

    fp_vault = create_vault(quantized, degree, seed=seed)

    return MultimodalVaultA(
        fp_vault=fp_vault,
        iris_code=iris_code.copy(),
        iris_mask=iris_mask.copy(),
        iris_threshold=iris_threshold,
        factor=factor,
        degree=degree,
    )


def unlock_vault_a(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    vault: MultimodalVaultA,
    max_shift: int = 32,
) -> bool:
    """
    Attempt to unlock a multimodal vault (Architecture A).

    1. Check iris FHD < threshold
    2. If iris passes, attempt fingerprint vault unlock
    3. Accept only if BOTH pass

    Args:
        minutiae_points: Query minutiae
        iris_code: Query iris code
        iris_mask: Query iris mask
        vault: Enrolled multimodal vault
        max_shift: Rotation compensation range

    Returns:
        True if both iris AND fingerprint pass
    """
    # Step 1: Iris check
    fhd = fractional_hamming_distance(
        vault.iris_code, vault.iris_mask,
        iris_code, iris_mask,
        max_shift=max_shift
    )

    if fhd >= vault.iris_threshold:
        return False  # Iris rejected

    # Step 2: Fingerprint vault check
    quantized = quantize_minutiae(minutiae_points, vault.factor)
    return unlock_vault(quantized, vault.fp_vault)
