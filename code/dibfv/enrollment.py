"""
D-IBFV Distributed Enrollment Pipeline.

Full pipeline:
    1. IBFV vault creation (biometric enrollment)
    2. Serialize vault to deterministic bytes
    3. Shamir split into n shares
    4. Upload each share to IPFS
    5. Store IPFS CIDs on blockchain
    6. Securely erase local vault bytes

Every phase is timed for Table 7 (Enrollment Latency) in the paper.
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import numpy as np

from . import shamir
from . import vault_serial
from .ipfs_client import IPFSClient
from .blockchain_client import EVMClient


@dataclass
class EnrollmentResult:
    """Result of a D-IBFV distributed enrollment."""
    success: bool
    user_id: str
    tx_hash: Optional[str] = None
    cids: List[str] = field(default_factory=list)
    threshold: int = 0
    share_count: int = 0
    vault_size_bytes: int = 0
    share_size_bytes: int = 0

    # Per-phase timing (milliseconds)
    time_vault_creation_ms: float = 0.0
    time_serialize_ms: float = 0.0
    time_shamir_split_ms: float = 0.0
    time_ipfs_upload_ms: float = 0.0
    time_blockchain_tx_ms: float = 0.0
    time_total_ms: float = 0.0
    gas_used: int = 0

    error: Optional[str] = None


def enroll(
    user_id: str,
    vault,
    t: int,
    n: int,
    ipfs: IPFSClient,
    blockchain: EVMClient,
) -> EnrollmentResult:
    """
    Execute the full D-IBFV distributed enrollment pipeline.

    Args:
        user_id:    Unique user identifier.
        vault:      A MultimodalVaultIBFV object (already created by IBFV enrollment).
        t:          Shamir reconstruction threshold.
        n:          Total number of Shamir shares.
        ipfs:       Connected IPFSClient instance.
        blockchain: Connected EVMClient with deployed VaultRegistry contract.

    Returns:
        EnrollmentResult with detailed timing and status.
    """
    result = EnrollmentResult(success=False, user_id=user_id, threshold=t, share_count=n)
    t_total_start = time.perf_counter()

    try:
        # Phase 1: Serialize vault
        t0 = time.perf_counter()
        blob = vault_serial.serialize(vault)
        result.time_serialize_ms = (time.perf_counter() - t0) * 1000
        result.vault_size_bytes = len(blob)

        # Phase 2: Shamir split
        t0 = time.perf_counter()
        shares = shamir.split(blob, t, n)
        result.time_shamir_split_ms = (time.perf_counter() - t0) * 1000
        result.share_size_bytes = len(shares[0][1])

        # Phase 3: Upload shares to IPFS
        t0 = time.perf_counter()
        cid_map = ipfs.upload_shares(shares)
        result.time_ipfs_upload_ms = (time.perf_counter() - t0) * 1000
        result.cids = [cid for _, cid in cid_map]

        # Phase 4: Store CIDs on blockchain
        t0 = time.perf_counter()
        tx_result = blockchain.store_vault(user_id, result.cids, t, n)
        result.time_blockchain_tx_ms = (time.perf_counter() - t0) * 1000
        result.tx_hash = tx_result['tx_hash']
        result.gas_used = tx_result['gas_used']

        # Secure erasure of local secrets
        _secure_erase(blob)
        for _, share_data in shares:
            _secure_erase(share_data)

        result.success = True

    except Exception as e:
        result.error = str(e)

    result.time_total_ms = (time.perf_counter() - t_total_start) * 1000
    return result


def enroll_with_vault_creation(
    user_id: str,
    minutiae_points: list,
    iris_code: np.ndarray,
    iris_mask: np.ndarray,
    factor: int,
    degree: int,
    t: int,
    n: int,
    ipfs: IPFSClient,
    blockchain: EVMClient,
    n_bonus: int = 4,
    block_size: int = 1023,
    bch_t: int = 0,
    seed: Optional[int] = None,
) -> EnrollmentResult:
    """
    Full enrollment including IBFV vault creation.

    This wraps the existing lock_vault_ibfv() followed by distributed enrollment.
    """
    import sys
    from pathlib import Path
    code_dir = str(Path(__file__).resolve().parent.parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from architectures.architecture_ibfv import lock_vault_ibfv

    result = EnrollmentResult(success=False, user_id=user_id, threshold=t, share_count=n)
    t_total_start = time.perf_counter()

    # Phase 0: Vault creation
    t0 = time.perf_counter()
    vault = lock_vault_ibfv(
        minutiae_points, iris_code, iris_mask,
        factor=factor, degree=degree,
        n_bonus=n_bonus, block_size=block_size,
        bch_t=bch_t, seed=seed,
    )
    result.time_vault_creation_ms = (time.perf_counter() - t0) * 1000

    if vault is None:
        result.error = "Vault creation failed (insufficient minutiae)"
        result.time_total_ms = (time.perf_counter() - t_total_start) * 1000
        return result

    # Delegate to the main enroll function
    sub = enroll(user_id, vault, t, n, ipfs, blockchain)

    # Merge timings
    result.success = sub.success
    result.tx_hash = sub.tx_hash
    result.cids = sub.cids
    result.vault_size_bytes = sub.vault_size_bytes
    result.share_size_bytes = sub.share_size_bytes
    result.time_serialize_ms = sub.time_serialize_ms
    result.time_shamir_split_ms = sub.time_shamir_split_ms
    result.time_ipfs_upload_ms = sub.time_ipfs_upload_ms
    result.time_blockchain_tx_ms = sub.time_blockchain_tx_ms
    result.gas_used = sub.gas_used
    result.error = sub.error
    result.time_total_ms = (time.perf_counter() - t_total_start) * 1000

    return result


def _secure_erase(data):
    """Best-effort zeroing of bytes/bytearray in memory."""
    if isinstance(data, bytearray):
        for i in range(len(data)):
            data[i] = 0
    # For immutable bytes, we can't erase — Python GC will collect.
    # In production, use ctypes.memset on the buffer address.
