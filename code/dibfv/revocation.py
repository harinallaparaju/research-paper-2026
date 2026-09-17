"""
D-IBFV Vault Revocation and Re-enrollment.

Pipeline:
    1. Blockchain: mark vault as revoked (immutable record)
    2. IPFS: unpin old shares (optional, allows GC)
    3. Re-enroll: create new vault with fresh parameters

Supports cancelability — if re-enrollment fails, the old vault
remains revoked and user must re-attempt.
"""

import time
from typing import Optional, List
from dataclasses import dataclass

from .ipfs_client import IPFSClient
from .blockchain_client import EVMClient
from . import enrollment


@dataclass
class RevocationResult:
    """Result of vault revocation."""
    success: bool
    user_id: str
    tx_hash: Optional[str] = None
    gas_used: int = 0
    time_revoke_ms: float = 0.0
    time_unpin_ms: float = 0.0
    time_total_ms: float = 0.0
    error: Optional[str] = None


def revoke(
    user_id: str,
    blockchain: EVMClient,
    ipfs: Optional[IPFSClient] = None,
    old_cids: Optional[List[str]] = None,
) -> RevocationResult:
    """
    Revoke a vault on-chain and optionally unpin IPFS shares.

    Args:
        user_id:    The user whose vault to revoke.
        blockchain: Connected EVMClient with deployed VaultRegistry.
        ipfs:       Optional IPFSClient for unpinning old shares.
        old_cids:   Optional list of old CIDs to unpin. If None and ipfs is
                    provided, will look up CIDs from blockchain before revoking.

    Returns:
        RevocationResult with timing and status.
    """
    result = RevocationResult(success=False, user_id=user_id)
    t_total = time.perf_counter()

    try:
        # Retrieve old CIDs before revoking (if needed for unpinning)
        if ipfs and old_cids is None:
            try:
                cids, _, _ = blockchain.lookup_vault(user_id)
                old_cids = cids
            except Exception:
                old_cids = []

        # Phase 1: Blockchain revocation
        t0 = time.perf_counter()
        tx_result = blockchain.revoke_vault(user_id)
        result.time_revoke_ms = (time.perf_counter() - t0) * 1000
        result.tx_hash = tx_result['tx_hash']
        result.gas_used = tx_result['gas_used']

        # Phase 2: IPFS unpin (best-effort, non-critical)
        if ipfs and old_cids:
            t0 = time.perf_counter()
            for cid in old_cids:
                try:
                    ipfs.unpin(cid)
                except Exception:
                    pass  # Non-critical
            result.time_unpin_ms = (time.perf_counter() - t0) * 1000

        result.success = True

    except Exception as e:
        result.error = str(e)

    result.time_total_ms = (time.perf_counter() - t_total) * 1000
    return result


def revoke_and_reenroll(
    user_id: str,
    new_vault,
    t: int,
    n: int,
    ipfs: IPFSClient,
    blockchain: EVMClient,
) -> tuple:
    """
    Atomic revoke-then-reenroll. Returns (RevocationResult, EnrollmentResult).
    """
    rev = revoke(user_id, blockchain, ipfs)
    if not rev.success:
        return rev, None

    enr = enrollment.enroll(user_id, new_vault, t, n, ipfs, blockchain)
    return rev, enr
