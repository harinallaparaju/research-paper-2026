"""
Architecture D: AES-Encrypted Vault (Iris Key).

The iris-derived stable key (via fuzzy commitment) encrypts the entire
fingerprint fuzzy vault using AES-256. To unlock, the user must first
recover the iris key, then decrypt the vault, then unlock it normally.

Protocol:
    Lock (Enrollment):
        1. Create standard fingerprint fuzzy vault
        2. Serialize the vault to bytes
        3. Create iris commitment → derive AES-256 key
        4. Encrypt serialized vault with AES-256-GCM
        5. Store: (encrypted_vault, iris_commitment, GCM_nonce, GCM_tag)

    Unlock (Verification):
        1. Recover iris key from query iris code via fuzzy commitment
        2. Derive AES-256 key from recovered iris key
        3. Decrypt the vault
        4. Unlock the decrypted fingerprint vault with query minutiae

Security: Vault is protected by AES-256 encryption. Without the correct
iris (which produces the decryption key), the vault contents are hidden.
An attacker cannot attempt fingerprint matching without first breaking
the iris protection.

Note: Due to iris code noise, this has higher FRR than Architecture A.
"""

import json
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    quantize_minutiae,
    create_vault,
    unlock_vault,
    VaultData,
)
from iris_stabilizer import IrisStabilizer, IrisCommitment


@dataclass
class MultimodalVaultD:
    """Architecture D vault data."""
    encrypted_vault: bytes                # AES-256-GCM encrypted vault
    nonce: bytes                          # GCM nonce
    tag: bytes                            # GCM authentication tag
    iris_commitment: IrisCommitment       # Stored iris commitment
    factor: int
    degree: int
    block_size: int                       # Iris stabilizer block size


def _serialize_vault(vault: VaultData) -> bytes:
    """Serialize VaultData to bytes for encryption."""
    data = {
        "degree": vault.polynomial_degree,
        "coefficients": vault.coefficients,
        "public_key": list(vault.public_key),
        "vault_points": vault.vault_points,
        "n_genuine": vault.n_genuine,
        "n_chaff": vault.n_chaff,
        "field_prime": vault.field_prime,
    }
    return json.dumps(data).encode("utf-8")


def _deserialize_vault(data: bytes) -> VaultData:
    """Deserialize bytes back to VaultData."""
    d = json.loads(data.decode("utf-8"))
    return VaultData(
        polynomial_degree=d["degree"],
        coefficients=d["coefficients"],
        secret_key=0,
        public_key=tuple(d["public_key"]),
        vault_points=[(x, y) for x, y in d["vault_points"]],
        n_genuine=d["n_genuine"],
        n_chaff=d["n_chaff"],
        field_prime=d["field_prime"],
    )


def lock_vault_d(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    factor: int,
    degree: int,
    block_size: int = 1023,
    seed: Optional[int] = None,
) -> Optional[MultimodalVaultD]:
    """
    Create a multimodal vault using Architecture D.

    Args:
        minutiae_points: Enrollment minutiae (x, y, theta)
        iris_code: Enrollment iris code
        iris_mask: Enrollment iris mask
        factor: Quantization factor
        degree: Polynomial degree
        block_size: Iris stabilizer block size
        seed: Random seed

    Returns:
        MultimodalVaultD or None if enrollment fails
    """
    # Create fingerprint vault
    quantized = quantize_minutiae(minutiae_points, factor)
    if len(quantized) < degree + 1:
        return None

    fp_vault = create_vault(quantized, degree, seed=seed)

    # Create iris commitment and derive key
    stabilizer = IrisStabilizer(block_size=block_size)
    iris_commitment = stabilizer.enroll(iris_code, iris_mask, seed=seed)

    # Generate enrollment key (deterministic from commitment)
    # Use random info bits from enrollment
    enroll_key = stabilizer.derive_key(
        np.frombuffer(
            bytes.fromhex(iris_commitment.key_hash[:64]), dtype=np.uint8
        )[:32]
    )

    # Encrypt the vault
    vault_bytes = _serialize_vault(fp_vault)
    aesgcm = AESGCM(enroll_key)
    nonce = os.urandom(12)
    encrypted = aesgcm.encrypt(nonce, vault_bytes, None)

    # GCM appends the tag to ciphertext
    ciphertext = encrypted[:-16]
    tag = encrypted[-16:]

    return MultimodalVaultD(
        encrypted_vault=encrypted,  # Full ciphertext + tag
        nonce=nonce,
        tag=tag,
        iris_commitment=iris_commitment,
        factor=factor,
        degree=degree,
        block_size=block_size,
    )


def unlock_vault_d(
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    vault: MultimodalVaultD,
    max_shift: int = 32,
) -> bool:
    """
    Attempt to unlock a multimodal vault (Architecture D).

    1. Recover iris key via fuzzy commitment
    2. Decrypt the vault with AES-256
    3. Unlock the fingerprint vault

    Returns:
        True if all steps succeed
    """
    # Step 1: Recover iris key
    stabilizer = IrisStabilizer(block_size=vault.block_size)
    result = stabilizer.recover(iris_code, iris_mask, vault.iris_commitment, max_shift=max_shift)

    if not result.success:
        return False

    # Step 2: Decrypt vault
    try:
        decrypt_key = stabilizer.derive_key(
            np.frombuffer(
                bytes.fromhex(result.key_hash[:64]), dtype=np.uint8
            )[:32]
        )
        aesgcm = AESGCM(decrypt_key)
        vault_bytes = aesgcm.decrypt(vault.nonce, vault.encrypted_vault, None)
    except Exception:
        return False  # Decryption failed (wrong key)

    # Step 3: Unlock fingerprint vault
    fp_vault = _deserialize_vault(vault_bytes)
    quantized = quantize_minutiae(minutiae_points, vault.factor)
    return unlock_vault(quantized, fp_vault)
