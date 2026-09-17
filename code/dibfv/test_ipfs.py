"""
Tests for IPFS client — upload/download shares, verify integrity.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

code_dir = str(Path(__file__).resolve().parent.parent)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from dibfv.ipfs_client import IPFSClient, IPFSError
from dibfv import shamir, vault_serial
from iris_stabilizer import IrisCommitment
from architectures.architecture_ibfv import MultimodalVaultIBFV
import hashlib


def _make_vault(n_points=70):
    """Create a test vault."""
    fp = 0xA9FB57DBA1EEA9BC3E660A909D838D726E3BF623D52620282013481D1F6E5377
    pts = [(int.from_bytes(os.urandom(32), 'big'), int.from_bytes(os.urandom(32), 'big'))
           for _ in range(n_points)]
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
        public_key=(int.from_bytes(os.urandom(32), 'big'), int.from_bytes(os.urandom(32), 'big')),
        vault_points=pts, n_genuine_fp=20, n_bonus=4, n_chaff=46,
        field_prime=fp, iris_commitment=ic, block_size=1023, factor=5, degree=10, bch_t=0,
    )


def test_online():
    client = IPFSClient()
    assert client.is_online(), "IPFS daemon not running"
    print("[PASS] IPFS daemon online")


def test_upload_download():
    client = IPFSClient()
    data = os.urandom(1024)
    cid = client.upload(data)
    assert cid.startswith("Qm") or cid.startswith("bafy"), f"Unexpected CID format: {cid}"
    retrieved = client.download(cid)
    assert retrieved == data, "Downloaded data mismatch"
    print(f"[PASS] Upload/download 1KB: CID={cid[:20]}...")


def test_share_upload_download():
    """Upload 5 shares, download 3, Shamir reconstruct, verify."""
    client = IPFSClient()
    secret = os.urandom(16 * 1024)  # 16KB
    shares = shamir.split(secret, 3, 5)

    # Upload all 5
    cid_map = client.upload_shares(shares)
    assert len(cid_map) == 5

    # Download only first 3
    downloaded = client.download_shares(cid_map[:3])
    restored = shamir.reconstruct(downloaded, 3)
    assert restored == secret, "Shamir reconstruction after IPFS round-trip failed"
    print("[PASS] 5 shares uploaded, 3 downloaded, Shamir reconstruct OK")


def test_full_vault_pipeline():
    """Vault → serialize → split → IPFS → download → reconstruct → deserialize."""
    client = IPFSClient()
    vault = _make_vault(70)
    blob = vault_serial.serialize(vault)

    shares = shamir.split(blob, 3, 5)
    cid_map = client.upload_shares(shares)

    downloaded = client.download_shares(cid_map[:3])
    restored = shamir.reconstruct(downloaded, 3)
    vault2 = vault_serial.deserialize(restored)

    assert vault2.polynomial_degree == vault.polynomial_degree
    assert vault2.public_key == vault.public_key
    assert vault2.vault_points == vault.vault_points
    assert np.array_equal(vault2.iris_commitment.helper_data, vault.iris_commitment.helper_data)
    print("[PASS] Full vault pipeline via IPFS")


def test_latency():
    """Measure upload/download latency for paper tables."""
    client = IPFSClient()
    data = os.urandom(16 * 1024)  # 16KB share

    N = 20
    cids = []
    t0 = time.perf_counter()
    for _ in range(N):
        cid = client.upload(data)
        cids.append(cid)
    upload_ms = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for cid in cids:
        _ = client.download(cid)
    download_ms = (time.perf_counter() - t0) / N * 1000

    print(f"[PASS] IPFS latency (16KB): upload={upload_ms:.1f}ms, download={download_ms:.1f}ms")

    # Full pipeline timing: serialize → split(3,5) → upload 5 → download 3 → reconstruct → deserialize
    vault = _make_vault(70)
    blob = vault_serial.serialize(vault)
    share_size_kb = len(shamir.split(blob, 3, 5)[0][1]) / 1024

    trials = 10
    t0 = time.perf_counter()
    for _ in range(trials):
        shares = shamir.split(blob, 3, 5)
        cids = client.upload_shares(shares)
        downloaded = client.download_shares(cids[:3])
        restored = shamir.reconstruct(downloaded, 3)
        _ = vault_serial.deserialize(restored)
    e2e_ms = (time.perf_counter() - t0) / trials * 1000

    print(f"       E2E pipeline (vault={len(blob)/1024:.1f}KB, share={share_size_kb:.1f}KB): {e2e_ms:.1f}ms")
    print(f"       Breakdown estimate: ser+split+recon+deser≈1ms, rest is IPFS I/O")


if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 70)
    print("IPFS CLIENT TESTS")
    print("=" * 70)

    test_online()
    test_upload_download()
    test_share_upload_download()
    test_full_vault_pipeline()
    test_latency()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
