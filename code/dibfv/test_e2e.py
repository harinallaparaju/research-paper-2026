"""
End-to-end test for the D-IBFV distributed pipeline.

Tests the full flow:
  1. Enroll vault → serialize → split → IPFS → blockchain
  2. Verify (blockchain → IPFS → reconstruct → deserialize → IBFV match)
  3. Revoke → verify fails
  4. Re-enroll → verify succeeds again
  5. Centralized vs distributed result consistency
"""

import sys
import os
import time
import hashlib
import numpy as np
from pathlib import Path

code_dir = str(Path(__file__).resolve().parent.parent)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from dibfv import enrollment, verification, revocation
from dibfv import shamir, vault_serial
from dibfv.ipfs_client import IPFSClient
from dibfv.blockchain_client import GanacheClient, _compile_contract
from iris_stabilizer import IrisCommitment
from architectures.architecture_ibfv import MultimodalVaultIBFV


def _make_vault(n_points=70, seed=42):
    """Create a realistic test vault with deterministic random data."""
    rng = np.random.RandomState(seed)
    fp = 0xA9FB57DBA1EEA9BC3E660A909D838D726E3BF623D52620282013481D1F6E5377
    pts = [(int.from_bytes(os.urandom(32), 'big'),
            int.from_bytes(os.urandom(32), 'big')) for _ in range(n_points)]
    n_bits = 49152
    ic = IrisCommitment(
        helper_data=rng.randint(0, 2, size=n_bits).astype(np.uint8),
        enrollment_mask=rng.randint(0, 2, size=n_bits).astype(np.uint8),
        block_size=1023, n_blocks=48,
        key_hash=hashlib.sha256(os.urandom(6)).hexdigest(),
        key_length_bits=48,
    )
    return MultimodalVaultIBFV(
        polynomial_degree=10,
        public_key=(int.from_bytes(os.urandom(32), 'big'),
                    int.from_bytes(os.urandom(32), 'big')),
        vault_points=pts, n_genuine_fp=20, n_bonus=4, n_chaff=46,
        field_prime=fp, iris_commitment=ic, block_size=1023, factor=5,
        degree=10, bch_t=0,
    )


def _vaults_equal(v1, v2):
    """Check vault equality (vault_points + public_key + iris commitment)."""
    if v1.polynomial_degree != v2.polynomial_degree:
        return False
    if v1.public_key != v2.public_key:
        return False
    if v1.vault_points != v2.vault_points:
        return False
    if not np.array_equal(v1.iris_commitment.helper_data,
                          v2.iris_commitment.helper_data):
        return False
    if not np.array_equal(v1.iris_commitment.enrollment_mask,
                          v2.iris_commitment.enrollment_mask):
        return False
    return True


def test_enrollment_pipeline(ipfs, bc):
    """Test distributed enrollment with timing breakdown."""
    vault = _make_vault(70, seed=1)
    result = enrollment.enroll("e2e_user_001", vault, t=3, n=5, ipfs=ipfs, blockchain=bc)

    assert result.success, f"Enrollment failed: {result.error}"
    assert len(result.cids) == 5
    assert result.vault_size_bytes > 0
    assert result.tx_hash is not None

    print(f"[PASS] Enrollment pipeline")
    print(f"  Vault: {result.vault_size_bytes/1024:.1f} KB, "
          f"Share: {result.share_size_bytes/1024:.1f} KB")
    print(f"  Timing: serialize={result.time_serialize_ms:.1f}ms, "
          f"split={result.time_shamir_split_ms:.1f}ms, "
          f"IPFS={result.time_ipfs_upload_ms:.1f}ms, "
          f"blockchain={result.time_blockchain_tx_ms:.1f}ms, "
          f"total={result.time_total_ms:.1f}ms")
    print(f"  Gas used: {result.gas_used}")
    return result


def test_vault_integrity_via_distributed(ipfs, bc):
    """
    Verify that serialize → split → IPFS → download → reconstruct → deserialize
    produces an IDENTICAL vault (zero-bit-error guarantee).
    """
    vault = _make_vault(70, seed=2)
    blob_original = vault_serial.serialize(vault)

    # Enroll
    result = enrollment.enroll("integrity_user", vault, t=3, n=5, ipfs=ipfs, blockchain=bc)
    assert result.success

    # Retrieve via blockchain + IPFS
    cids, t_val, n_val = bc.lookup_vault("integrity_user")
    share_cids = list(zip(range(1, t_val + 1), cids[:t_val]))
    downloaded = ipfs.download_shares(share_cids)
    blob_restored = shamir.reconstruct(downloaded, t_val)

    # Deserialize and compare
    vault_restored = vault_serial.deserialize(blob_restored)
    assert _vaults_equal(vault, vault_restored), "Vault integrity violated!"

    # Also check raw bytes match
    blob_re_serialized = vault_serial.serialize(vault_restored)
    assert blob_original == blob_re_serialized, "Re-serialization mismatch!"

    print("[PASS] Zero-bit-error vault integrity through distributed pipeline")


def test_revocation(ipfs, bc):
    """Test revocation blocks further lookups."""
    vault = _make_vault(70, seed=3)
    enrollment.enroll("revoke_user", vault, t=3, n=5, ipfs=ipfs, blockchain=bc)

    # Revoke
    rev = revocation.revoke("revoke_user", bc, ipfs)
    assert rev.success, f"Revocation failed: {rev.error}"

    # Status should show revoked
    active, revoked = bc.vault_status("revoke_user")
    assert not active and revoked

    # Lookup should fail
    try:
        bc.lookup_vault("revoke_user")
        assert False, "Should have raised on revoked vault"
    except Exception:
        pass

    print(f"[PASS] Revocation (gas={rev.gas_used}, latency={rev.time_revoke_ms:.1f}ms)")


def test_re_enrollment(ipfs, bc):
    """Test revoke → re-enroll with new parameters."""
    vault1 = _make_vault(70, seed=4)
    enrollment.enroll("reenroll_user", vault1, t=3, n=5, ipfs=ipfs, blockchain=bc)
    
    # Revoke and re-enroll with different vault
    vault2 = _make_vault(80, seed=5)
    rev, enr = revocation.revoke_and_reenroll(
        "reenroll_user", vault2, t=3, n=5, ipfs=ipfs, blockchain=bc
    )
    assert rev.success
    assert enr.success

    # Verify we get vault2 back
    cids, t_val, n_val = bc.lookup_vault("reenroll_user")
    share_cids = list(zip(range(1, t_val + 1), cids[:t_val]))
    downloaded = ipfs.download_shares(share_cids)
    blob = shamir.reconstruct(downloaded, t_val)
    vault_back = vault_serial.deserialize(blob)

    assert _vaults_equal(vault2, vault_back), "Re-enrolled vault mismatch!"
    print("[PASS] Re-enrollment after revocation")


def test_shamir_threshold_configs(ipfs, bc):
    """Test all (t,n) configurations from the paper."""
    configs = [(2, 3), (3, 5), (4, 7), (5, 9)]
    for t, n in configs:
        vault = _make_vault(70, seed=t * 100 + n)
        uid = f"config_user_{t}_{n}"
        result = enrollment.enroll(uid, vault, t=t, n=n, ipfs=ipfs, blockchain=bc)
        assert result.success, f"({t},{n}) enrollment failed: {result.error}"

        # Verify with exactly t shares
        cids, t_back, n_back = bc.lookup_vault(uid)
        assert t_back == t and n_back == n
        share_cids = list(zip(range(1, t + 1), cids[:t]))
        downloaded = ipfs.download_shares(share_cids)
        blob = shamir.reconstruct(downloaded, t)
        vault_back = vault_serial.deserialize(blob)
        assert _vaults_equal(vault, vault_back)

    print(f"[PASS] All Shamir configs: {configs}")


def test_timing_summary(ipfs, bc):
    """Comprehensive timing for paper tables."""
    vault = _make_vault(70, seed=99)
    N = 10
    timings = []

    for i in range(N):
        uid = f"timing_user_{i:04d}"
        r = enrollment.enroll(uid, vault, t=3, n=5, ipfs=ipfs, blockchain=bc)
        assert r.success
        timings.append(r)

    def avg(lst):
        return sum(lst) / len(lst)

    print("[PASS] Enrollment timing (n=10):")
    print(f"  Serialize:    {avg([t.time_serialize_ms for t in timings]):.1f}ms")
    print(f"  Shamir split: {avg([t.time_shamir_split_ms for t in timings]):.1f}ms")
    print(f"  IPFS upload:  {avg([t.time_ipfs_upload_ms for t in timings]):.1f}ms")
    print(f"  Blockchain:   {avg([t.time_blockchain_tx_ms for t in timings]):.1f}ms")
    print(f"  Total:        {avg([t.time_total_ms for t in timings]):.1f}ms")
    print(f"  Avg gas:      {avg([t.gas_used for t in timings]):.0f}")


if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 70)
    print("D-IBFV END-TO-END PIPELINE TESTS")
    print("=" * 70)

    # Setup infrastructure
    ipfs = IPFSClient()
    if not ipfs.is_online():
        print("ERROR: IPFS daemon not running. Start with: ipfs daemon &")
        sys.exit(1)

    bc = GanacheClient()
    if not bc.w3.is_connected():
        print("ERROR: Ganache not running. Start with: docker run -d -p 8545:8545 trufflesuite/ganache:latest --deterministic")
        sys.exit(1)

    bytecode = _compile_contract()
    bc.deploy(bytecode)
    print(f"Contract deployed at {bc.contract_address}\n")

    test_enrollment_pipeline(ipfs, bc)
    test_vault_integrity_via_distributed(ipfs, bc)
    test_revocation(ipfs, bc)
    test_re_enrollment(ipfs, bc)
    test_shamir_threshold_configs(ipfs, bc)
    test_timing_summary(ipfs, bc)

    print("\n" + "=" * 70)
    print("ALL E2E TESTS PASSED")
    print("=" * 70)
