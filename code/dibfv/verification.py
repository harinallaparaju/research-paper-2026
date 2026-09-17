"""
D-IBFV Distributed Verification Pipeline.

Full pipeline:
    1. Blockchain lookup → retrieve CIDs + Shamir params
    2. IPFS download → retrieve t shares
    3. Shamir reconstruct → recover serialized vault
    4. Integrity check → verify SHA-256 trailer
    5. Deserialize → MultimodalVaultIBFV
    6. IBFV verify → biometric accept/reject
    7. Secure erase all local secrets

Every phase is timed for Table 8 (Verification Latency) in the paper.
"""

import time
from typing import Optional, List
from dataclasses import dataclass

import numpy as np

from . import shamir
from . import vault_serial
from .ipfs_client import IPFSClient
from .blockchain_client import EVMClient


@dataclass
class VerificationResult:
    """Result of a D-IBFV distributed verification."""
    success: bool                         # Biometric match result
    user_id: str
    match: Optional[bool] = None          # True=accept, False=reject, None=error

    # Per-phase timing (milliseconds)
    time_blockchain_lookup_ms: float = 0.0
    time_ipfs_download_ms: float = 0.0
    time_shamir_reconstruct_ms: float = 0.0
    time_integrity_check_ms: float = 0.0
    time_deserialize_ms: float = 0.0
    time_ibfv_verify_ms: float = 0.0
    time_total_ms: float = 0.0

    error: Optional[str] = None


def verify(
    user_id: str,
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    ipfs: IPFSClient,
    blockchain: EVMClient,
    max_shift: int = 16,
) -> VerificationResult:
    """
    Execute the full D-IBFV distributed verification pipeline.

    Args:
        user_id:         The enrolled user's identifier.
        minutiae_points: Query fingerprint minutiae (x, y, theta).
        iris_code:       Query iris code (binary array).
        iris_mask:       Query iris mask (binary array).
        ipfs:            Connected IPFSClient instance.
        blockchain:      Connected EVMClient with deployed VaultRegistry.
        max_shift:       Maximum iris rotation shift for matching.

    Returns:
        VerificationResult with match decision and detailed timing.
    """
    result = VerificationResult(success=False, user_id=user_id)
    t_total_start = time.perf_counter()

    try:
        # Phase 1: Blockchain lookup
        t0 = time.perf_counter()
        cids, t_threshold, n_shares = blockchain.lookup_vault(user_id)
        result.time_blockchain_lookup_ms = (time.perf_counter() - t0) * 1000

        # Phase 2: IPFS download (only t shares needed)
        t0 = time.perf_counter()
        share_cids = list(zip(range(1, t_threshold + 1), cids[:t_threshold]))
        downloaded = ipfs.download_shares(share_cids)
        result.time_ipfs_download_ms = (time.perf_counter() - t0) * 1000

        # Phase 3: Shamir reconstruct
        t0 = time.perf_counter()
        blob = shamir.reconstruct(downloaded, t_threshold)
        result.time_shamir_reconstruct_ms = (time.perf_counter() - t0) * 1000

        # Phase 4: Integrity check + deserialize
        t0 = time.perf_counter()
        vault = vault_serial.deserialize(blob)  # raises ValueError on SHA-256 mismatch
        result.time_deserialize_ms = (time.perf_counter() - t0) * 1000

        # Phase 5: IBFV biometric verification
        import sys
        from pathlib import Path
        code_dir = str(Path(__file__).resolve().parent.parent)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from architectures.architecture_ibfv import unlock_vault_ibfv

        t0 = time.perf_counter()
        match = unlock_vault_ibfv(
            vault, minutiae_points, iris_code, iris_mask,
            max_shift=max_shift,
        )
        result.time_ibfv_verify_ms = (time.perf_counter() - t0) * 1000

        result.match = match
        result.success = True

        # Secure erase
        _secure_erase(blob)
        for _, share_data in downloaded:
            _secure_erase(share_data)

    except Exception as e:
        result.error = str(e)

    result.time_total_ms = (time.perf_counter() - t_total_start) * 1000
    return result


def verify_from_vault(
    vault,
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    max_shift: int = 16,
) -> bool:
    """
    Direct IBFV verification without distributed infrastructure.
    Used for regression testing — results must be identical to verify().
    """
    import sys
    from pathlib import Path
    code_dir = str(Path(__file__).resolve().parent.parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from architectures.architecture_ibfv import unlock_vault_ibfv

    return unlock_vault_ibfv(vault, minutiae_points, iris_code, iris_mask,
                             max_shift=max_shift)


def _secure_erase(data):
    """Best-effort zeroing of bytes/bytearray."""
    if isinstance(data, bytearray):
        for i in range(len(data)):
            data[i] = 0
