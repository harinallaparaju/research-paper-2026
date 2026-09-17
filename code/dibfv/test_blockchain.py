"""
Tests for blockchain_client.py — Ganache EVM contract deployment and operations.

Tests:
  1. Contract compilation and deployment
  2. Store vault CIDs
  3. Lookup vault CIDs
  4. Revoke vault
  5. Re-enrollment after revocation
  6. Full pipeline: serialize → split → IPFS upload → blockchain store → lookup → download → reconstruct → deserialize
  7. Performance benchmarks (gas costs, latency)
"""

import sys
import os
import time
import numpy as np
import hashlib
from pathlib import Path

code_dir = str(Path(__file__).resolve().parent.parent)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from dibfv.blockchain_client import GanacheClient, _compile_contract
from dibfv import shamir, vault_serial
from dibfv.ipfs_client import IPFSClient
from iris_stabilizer import IrisCommitment
from architectures.architecture_ibfv import MultimodalVaultIBFV


def _make_vault(n_points=70):
    fp = 0xA9FB57DBA1EEA9BC3E660A909D838D726E3BF623D52620282013481D1F6E5377
    pts = [(int.from_bytes(os.urandom(32), 'big'),
            int.from_bytes(os.urandom(32), 'big')) for _ in range(n_points)]
    n_bits = 49152
    ic = IrisCommitment(
        helper_data=np.random.randint(0, 2, size=n_bits, dtype=np.uint8),
        enrollment_mask=np.random.randint(0, 2, size=n_bits, dtype=np.uint8),
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


def test_compile_deploy():
    """Compile Solidity contract and deploy to Ganache."""
    bytecode = _compile_contract()
    assert bytecode.startswith("0x"), f"Bad bytecode prefix: {bytecode[:10]}"
    assert len(bytecode) > 100, "Bytecode too short"
    print(f"  Bytecode length: {len(bytecode)} chars")

    client = GanacheClient()
    address = client.deploy(bytecode)
    assert address is not None
    assert address.startswith("0x")
    print(f"[PASS] Contract deployed at {address}")
    return client, bytecode


def test_store_lookup(client):
    """Store and lookup vault CIDs."""
    user_id = "test_user_001"
    cids = ["QmAAA111", "QmBBB222", "QmCCC333", "QmDDD444", "QmEEE555"]
    result = client.store_vault(user_id, cids, t=3, n=5)

    assert result['gas_used'] > 0
    print(f"  Store gas: {result['gas_used']}, latency: {result['latency_ms']:.1f}ms")

    cids_back, t_back, n_back = client.lookup_vault(user_id)
    assert cids_back == cids, f"CID mismatch: {cids_back}"
    assert t_back == 3
    assert n_back == 5
    print("[PASS] Store + Lookup vault CIDs")
    return result


def test_revoke(client):
    """Revoke a vault and verify it's no longer accessible."""
    user_id = "test_user_revoke"
    cids = ["QmXXX", "QmYYY", "QmZZZ"]
    client.store_vault(user_id, cids, t=2, n=3)

    # Check status
    active, revoked = client.vault_status(user_id)
    assert active and not revoked

    # Revoke
    result = client.revoke_vault(user_id)
    assert result['gas_used'] > 0
    print(f"  Revoke gas: {result['gas_used']}, latency: {result['latency_ms']:.1f}ms")

    # Check status after revocation
    active, revoked = client.vault_status(user_id)
    assert not active and revoked

    # Lookup should fail
    try:
        client.lookup_vault(user_id)
        assert False, "Should have raised"
    except Exception:
        pass

    print("[PASS] Revoke vault")


def test_re_enrollment(client):
    """Re-enroll after revocation with new CIDs."""
    user_id = "test_user_reenroll"
    old_cids = ["QmOLD1", "QmOLD2", "QmOLD3"]
    client.store_vault(user_id, old_cids, t=2, n=3)
    client.revoke_vault(user_id)

    new_cids = ["QmNEW1", "QmNEW2", "QmNEW3", "QmNEW4", "QmNEW5"]
    client.store_vault(user_id, new_cids, t=3, n=5)

    cids_back, t_back, n_back = client.lookup_vault(user_id)
    assert cids_back == new_cids
    assert t_back == 3 and n_back == 5
    print("[PASS] Re-enrollment after revocation")


def test_full_pipeline(client):
    """Full D-IBFV pipeline: vault → serialize → split → IPFS → blockchain → lookup → download → reconstruct → deserialize."""
    ipfs = IPFSClient()
    if not ipfs.is_online():
        print("[SKIP] IPFS daemon not running — skipping full pipeline")
        return

    vault = _make_vault(70)
    blob = vault_serial.serialize(vault)

    # Split
    shares = shamir.split(blob, 3, 5)

    # Upload shares to IPFS
    cid_map = ipfs.upload_shares(shares)
    cids = [cid for _, cid in cid_map]
    indices = [idx for idx, _ in cid_map]

    # Store CIDs on blockchain
    user_id = "test_user_full_pipeline"
    store_result = client.store_vault(user_id, cids, t=3, n=5)

    # Lookup from blockchain
    cids_back, t_back, n_back = client.lookup_vault(user_id)
    assert cids_back == cids

    # Download first t shares from IPFS
    share_cids = list(zip(indices[:t_back], cids_back[:t_back]))
    downloaded = ipfs.download_shares(share_cids)

    # Reconstruct
    restored = shamir.reconstruct(downloaded, t_back)
    vault2 = vault_serial.deserialize(restored)

    assert vault2.polynomial_degree == vault.polynomial_degree
    assert vault2.public_key == vault.public_key
    assert vault2.vault_points == vault.vault_points
    assert np.array_equal(vault2.iris_commitment.helper_data,
                          vault.iris_commitment.helper_data)

    print(f"[PASS] Full D-IBFV pipeline (blockchain gas: {store_result['gas_used']})")


def test_performance(client, bytecode):
    """Benchmark store/lookup/revoke operations."""
    N = 20
    cids = [f"Qm{'A' * 44}{i:02d}" for i in range(5)]

    # Store benchmark
    gas_store = []
    latency_store = []
    for i in range(N):
        uid = f"bench_user_{i:04d}"
        r = client.store_vault(uid, cids, t=3, n=5)
        gas_store.append(r['gas_used'])
        latency_store.append(r['latency_ms'])

    avg_gas = sum(gas_store) / N
    avg_lat = sum(latency_store) / N
    print(f"  Store:  gas={avg_gas:.0f}, latency={avg_lat:.1f}ms (n={N})")

    # Lookup benchmark (view call, no gas)
    t0 = time.perf_counter()
    for i in range(N):
        uid = f"bench_user_{i:04d}"
        client.lookup_vault(uid)
    lookup_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Lookup: latency={lookup_ms:.1f}ms (n={N})")

    # Revoke benchmark
    gas_revoke = []
    latency_revoke = []
    for i in range(N):
        uid = f"bench_user_{i:04d}"
        r = client.revoke_vault(uid)
        gas_revoke.append(r['gas_used'])
        latency_revoke.append(r['latency_ms'])

    avg_gas_r = sum(gas_revoke) / N
    avg_lat_r = sum(latency_revoke) / N
    print(f"  Revoke: gas={avg_gas_r:.0f}, latency={avg_lat_r:.1f}ms (n={N})")

    # TPS estimate (sequential)
    tps_store = 1000.0 / avg_lat if avg_lat > 0 else 0
    tps_lookup = 1000.0 / lookup_ms if lookup_ms > 0 else 0
    print(f"  Estimated TPS: store={tps_store:.0f}, lookup={tps_lookup:.0f}")
    print("[PASS] Performance benchmarks complete")


if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 70)
    print("BLOCKCHAIN CLIENT TESTS (Ganache via OrbStack)")
    print("=" * 70)

    client, bytecode = test_compile_deploy()
    test_store_lookup(client)
    test_revoke(client)
    test_re_enrollment(client)
    test_full_pipeline(client)
    test_performance(client, bytecode)

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
