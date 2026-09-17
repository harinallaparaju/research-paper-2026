"""
Tests for vault_serial.py — deterministic binary serialization of MultimodalVaultIBFV.

Tests:
  1. Round-trip: serialize → deserialize → identical vault
  2. Round-trip with BCH iris commitment
  3. Determinism: same vault → identical bytes
  4. Integrity detection: flip one bit → SHA-256 mismatch
  5. Bad magic rejected
  6. Large vault (400 points) — size and round-trip
  7. Serialize → Shamir split → reconstruct → deserialize → identical
  8. Bit-packing correctness for edge-case arrays
  9. Performance benchmark
"""

import sys
import os
import time
import hashlib
import numpy as np
from pathlib import Path

# Add code/ to path
code_dir = str(Path(__file__).resolve().parent.parent)
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from dibfv import vault_serial, shamir
from iris_stabilizer import IrisCommitment, IrisCommitmentBCH
from architectures.architecture_ibfv import MultimodalVaultIBFV

# --- Helpers ---

def _random_256bit():
    """Random 256-bit integer."""
    return int.from_bytes(os.urandom(32), 'big')


def _make_iris_commitment(n_bits=49152, block_size=1023):
    """Create a realistic IrisCommitment with random data."""
    n_blocks = n_bits // block_size
    return IrisCommitment(
        helper_data=np.random.randint(0, 2, size=n_bits, dtype=np.uint8),
        enrollment_mask=np.random.randint(0, 2, size=n_bits, dtype=np.uint8),
        block_size=block_size,
        n_blocks=n_blocks,
        key_hash=hashlib.sha256(os.urandom(6)).hexdigest(),
        key_length_bits=n_blocks,
    )


def _make_iris_commitment_bch(n_bits=49152, block_size=1023):
    """Create a realistic IrisCommitmentBCH with random data."""
    n_blocks = n_bits // block_size
    valid_mask = np.random.randint(0, 2, size=n_blocks, dtype=np.uint8)
    bch_n = int(valid_mask.sum())
    bch_k = max(1, bch_n // 2)
    return IrisCommitmentBCH(
        helper_data=np.random.randint(0, 2, size=n_bits, dtype=np.uint8),
        enrollment_mask=np.random.randint(0, 2, size=n_bits, dtype=np.uint8),
        block_size=block_size,
        n_blocks=n_blocks,
        valid_block_mask=valid_mask,
        bch_n=bch_n,
        bch_k=bch_k,
        bch_t=3,
        min_block_valid=400,
        key_hash=hashlib.sha256(os.urandom(6)).hexdigest(),
        key_length_bits=bch_k,
    )


def _make_vault(n_points=70, bch=False, n_bits=49152, block_size=1023):
    """Create a realistic MultimodalVaultIBFV with random data."""
    fp = 0x00000000000000000000000000000000A9FB57DBA1EEA9BC3E660A909D838D726E3BF623D52620282013481D1F6E5377
    vault_points = [(_random_256bit(), _random_256bit()) for _ in range(n_points)]
    ic = _make_iris_commitment_bch(n_bits, block_size) if bch else _make_iris_commitment(n_bits, block_size)
    return MultimodalVaultIBFV(
        polynomial_degree=10,
        public_key=(_random_256bit(), _random_256bit()),
        vault_points=vault_points,
        n_genuine_fp=20,
        n_bonus=4,
        n_chaff=n_points - 24,
        field_prime=fp,
        iris_commitment=ic,
        block_size=block_size,
        factor=5,
        degree=10,
        bch_t=3 if bch else 0,
    )


def _vaults_equal(v1, v2) -> bool:
    """Deep equality check for two MultimodalVaultIBFV instances."""
    if v1.polynomial_degree != v2.polynomial_degree:
        return False
    if v1.public_key != v2.public_key:
        return False
    if v1.vault_points != v2.vault_points:
        return False
    if v1.n_genuine_fp != v2.n_genuine_fp:
        return False
    if v1.n_bonus != v2.n_bonus:
        return False
    if v1.n_chaff != v2.n_chaff:
        return False
    if v1.field_prime != v2.field_prime:
        return False
    if v1.block_size != v2.block_size:
        return False
    if v1.factor != v2.factor:
        return False
    if v1.degree != v2.degree:
        return False
    if v1.bch_t != v2.bch_t:
        return False

    ic1, ic2 = v1.iris_commitment, v2.iris_commitment
    if type(ic1) != type(ic2):
        return False
    if not np.array_equal(ic1.helper_data, ic2.helper_data):
        return False
    if not np.array_equal(ic1.enrollment_mask, ic2.enrollment_mask):
        return False
    if ic1.block_size != ic2.block_size:
        return False
    if ic1.n_blocks != ic2.n_blocks:
        return False
    if ic1.key_hash != ic2.key_hash:
        return False
    if ic1.key_length_bits != ic2.key_length_bits:
        return False

    if isinstance(ic1, IrisCommitmentBCH):
        if not np.array_equal(ic1.valid_block_mask, ic2.valid_block_mask):
            return False
        if ic1.bch_n != ic2.bch_n:
            return False
        if ic1.bch_k != ic2.bch_k:
            return False
        if ic1.bch_t != ic2.bch_t:
            return False
        if ic1.min_block_valid != ic2.min_block_valid:
            return False

    return True


# --- Tests ---

def test_roundtrip_basic():
    """Serialize → deserialize → identical vault (IrisCommitment)."""
    vault = _make_vault(n_points=70, bch=False)
    data = vault_serial.serialize(vault)
    vault2 = vault_serial.deserialize(data)
    assert _vaults_equal(vault, vault2), "Basic round-trip failed"
    print("[PASS] Basic round-trip (IrisCommitment)")


def test_roundtrip_bch():
    """Serialize → deserialize → identical vault (IrisCommitmentBCH)."""
    vault = _make_vault(n_points=70, bch=True)
    data = vault_serial.serialize(vault)
    vault2 = vault_serial.deserialize(data)
    assert _vaults_equal(vault, vault2), "BCH round-trip failed"
    print("[PASS] BCH round-trip (IrisCommitmentBCH)")


def test_determinism():
    """Same vault serialized twice → identical bytes."""
    vault = _make_vault(n_points=50)
    d1 = vault_serial.serialize(vault)
    d2 = vault_serial.serialize(vault)
    assert d1 == d2, "Determinism failed"
    print("[PASS] Deterministic serialization")


def test_integrity_detection():
    """Corrupting one byte triggers SHA-256 mismatch."""
    vault = _make_vault(n_points=50)
    data = bytearray(vault_serial.serialize(vault))
    # Flip a byte in the middle of the body
    mid = len(data) // 2
    data[mid] ^= 0x01
    try:
        vault_serial.deserialize(bytes(data))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "SHA-256" in str(e) or "Integrity" in str(e)
    print("[PASS] Integrity detection (corrupted byte)")


def test_bad_magic():
    """Bad magic bytes → rejected."""
    vault = _make_vault(n_points=30)
    data = bytearray(vault_serial.serialize(vault))
    data[0:4] = b'XXXX'
    # Recompute trailer so integrity passes but magic fails
    body = bytes(data[:-32])
    trailer = hashlib.sha256(body).digest()
    data[-32:] = trailer
    try:
        vault_serial.deserialize(bytes(data))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "magic" in str(e).lower() or "Bad" in str(e)
    print("[PASS] Bad magic rejected")


def test_large_vault():
    """400-point vault round-trips correctly, check size."""
    vault = _make_vault(n_points=400, bch=True)
    data = vault_serial.serialize(vault)
    vault2 = vault_serial.deserialize(data)
    assert _vaults_equal(vault, vault2), "Large vault round-trip failed"
    size_kb = len(data) / 1024
    print(f"[PASS] Large vault (400 points): {len(data)} bytes ({size_kb:.1f} KB)")


def test_shamir_integration():
    """serialize → Shamir split → reconstruct → deserialize → identical."""
    vault = _make_vault(n_points=70, bch=False)
    blob = vault_serial.serialize(vault)

    for t, n in [(2, 3), (3, 5)]:
        shares = shamir.split(blob, t, n)
        restored = shamir.reconstruct(shares[:t], t)
        vault2 = vault_serial.deserialize(restored)
        assert _vaults_equal(vault, vault2), f"Shamir integration failed for ({t},{n})"

    print("[PASS] Shamir split → reconstruct → deserialize round-trip")


def test_bit_packing_edge_cases():
    """Verify bit packing for all-zero, all-one, and non-byte-aligned lengths."""
    for n in [1, 7, 8, 9, 15, 16, 49152]:
        arr = np.random.randint(0, 2, size=n, dtype=np.uint8)
        packed = vault_serial._pack_bits(arr)
        unpacked = vault_serial._unpack_bits(packed, n)
        assert np.array_equal(arr, unpacked), f"Bit packing failed for n={n}"

    # All zeros
    arr = np.zeros(100, dtype=np.uint8)
    assert np.array_equal(arr, vault_serial._unpack_bits(vault_serial._pack_bits(arr), 100))

    # All ones
    arr = np.ones(100, dtype=np.uint8)
    assert np.array_equal(arr, vault_serial._unpack_bits(vault_serial._pack_bits(arr), 100))

    print("[PASS] Bit packing edge cases")


def test_performance():
    """Benchmark serialize/deserialize speed."""
    vault = _make_vault(n_points=70, bch=False)

    # Warm up
    vault_serial.serialize(vault)

    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        data = vault_serial.serialize(vault)
    t_ser = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        vault_serial.deserialize(data)
    t_deser = (time.perf_counter() - t0) / N * 1000

    size_kb = len(data) / 1024
    print(f"[PASS] Performance ({size_kb:.1f} KB): serialize={t_ser:.1f}ms, deserialize={t_deser:.1f}ms")

    # With Shamir end-to-end
    t0 = time.perf_counter()
    for _ in range(N):
        blob = vault_serial.serialize(vault)
        shares = shamir.split(blob, 3, 5)
        restored = shamir.reconstruct(shares[:3], 3)
        _ = vault_serial.deserialize(restored)
    t_e2e = (time.perf_counter() - t0) / N * 1000
    print(f"       E2E (ser → split(3,5) → recon → deser): {t_e2e:.1f}ms")


def test_empty_vault_points():
    """Edge case: vault with zero points (degenerate but valid)."""
    vault = _make_vault(n_points=0)
    vault.n_genuine_fp = 0
    vault.n_bonus = 0
    vault.n_chaff = 0
    data = vault_serial.serialize(vault)
    vault2 = vault_serial.deserialize(data)
    assert _vaults_equal(vault, vault2)
    print("[PASS] Empty vault points edge case")


def test_field_prime_sizes():
    """Vault with different field prime sizes round-trips correctly."""
    for prime_bits in [128, 192, 256, 384]:
        vault = _make_vault(n_points=30)
        # Create a prime-like value of the specified bit size
        vault.field_prime = (1 << (prime_bits - 1)) + int.from_bytes(os.urandom(prime_bits // 8 - 1), 'big')
        data = vault_serial.serialize(vault)
        vault2 = vault_serial.deserialize(data)
        assert vault2.field_prime == vault.field_prime, f"Field prime mismatch for {prime_bits}-bit"
    print("[PASS] Variable field prime sizes")


if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 70)
    print("VAULT SERIALIZATION TESTS")
    print("=" * 70)

    test_roundtrip_basic()
    test_roundtrip_bch()
    test_determinism()
    test_integrity_detection()
    test_bad_magic()
    test_large_vault()
    test_shamir_integration()
    test_bit_packing_edge_cases()
    test_performance()
    test_empty_vault_points()
    test_field_prime_sizes()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
