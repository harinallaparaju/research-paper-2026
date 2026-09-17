"""
Deterministic binary serialization for MultimodalVaultIBFV.

Encodes the complete IBFV vault (fingerprint vault points, ECC public key,
iris fuzzy commitment, all parameters) into a compact byte string suitable
for Shamir splitting and distributed storage.

Binary Format (all multi-byte integers little-endian unless noted):

    HEADER (16 bytes):
        magic           4B    b'IBFV'
        version         2B    uint16 (0x0001)
        flags           2B    bit0 = BCH mode
        poly_degree     2B    uint16
        n_genuine_fp    2B    uint16
        n_bonus         2B    uint16
        n_chaff         2B    uint16

    PARAMS (16 bytes):
        block_size      2B    uint16
        factor          2B    uint16
        degree          2B    uint16
        bch_t           2B    uint16
        n_vault_points  4B    uint32
        field_prime_len 2B    uint16 (bytes, typically 32)
        reserved        2B    zero

    FIELD_PRIME (field_prime_len bytes):  big-endian

    PUBLIC_KEY (64 bytes):   pk_x (32B BE) + pk_y (32B BE)

    VAULT_POINTS (n_vault_points × 64 bytes):
        each: x (32B BE) + y (32B BE)

    IRIS_COMMITMENT:
        helper_len      4B    uint32 (number of bits)
        helper_bits     ⌈helper_len/8⌉ bytes (packed, MSB-first per byte)
        mask_bits       ⌈helper_len/8⌉ bytes
        iris_bs         2B    uint16
        iris_n_blocks   2B    uint16
        key_hash        32B   raw SHA-256 bytes (decoded from hex string)
        key_len_bits    2B    uint16

        IF BCH (flags bit0 = 1):
            vbm_len         2B   uint16 (= n_blocks)
            vbm_bits        ⌈vbm_len/8⌉ bytes
            bch_n           2B
            bch_k           2B
            bch_t_iris      2B
            min_block_valid 2B

    TRAILER (32 bytes):  SHA-256 of all preceding bytes

Properties:
    - Deterministic: same vault → same bytes (no padding randomness)
    - Compact: ~8-40 KB depending on vault size
    - Self-validating: SHA-256 integrity trailer
    - Lossless: deserialize(serialize(v)) recreates the exact vault
"""

import struct
import hashlib
from typing import Tuple

import numpy as np

# Lazy imports to avoid circular dependency
_MultimodalVaultIBFV = None
_IrisCommitment = None
_IrisCommitmentBCH = None

MAGIC = b'IBFV'
VERSION = 1
FLAG_BCH = 0x0001


def _import_types():
    global _MultimodalVaultIBFV, _IrisCommitment, _IrisCommitmentBCH
    if _MultimodalVaultIBFV is None:
        import sys
        from pathlib import Path
        code_dir = str(Path(__file__).resolve().parent.parent)
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        from architectures.architecture_ibfv import MultimodalVaultIBFV
        from iris_stabilizer import IrisCommitment, IrisCommitmentBCH
        _MultimodalVaultIBFV = MultimodalVaultIBFV
        _IrisCommitment = IrisCommitment
        _IrisCommitmentBCH = IrisCommitmentBCH


def _pack_bits(arr: np.ndarray) -> bytes:
    """Pack a binary (0/1) numpy array into bytes, MSB-first per byte."""
    n = len(arr)
    # Pad to multiple of 8
    padded = np.zeros(((n + 7) // 8) * 8, dtype=np.uint8)
    padded[:n] = arr.astype(np.uint8)
    return np.packbits(padded, bitorder='big').tobytes()


def _unpack_bits(data: bytes, n_bits: int) -> np.ndarray:
    """Unpack bytes into a binary numpy array of exactly n_bits elements."""
    all_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='big')
    return all_bits[:n_bits].astype(np.uint8)


def _int_to_be(val: int, length: int) -> bytes:
    """Encode a non-negative integer as big-endian bytes of fixed length."""
    return val.to_bytes(length, byteorder='big')


def _be_to_int(data: bytes) -> int:
    """Decode big-endian bytes to a non-negative integer."""
    return int.from_bytes(data, byteorder='big')


def serialize(vault) -> bytes:
    """
    Serialize a MultimodalVaultIBFV to deterministic bytes.

    Args:
        vault: A MultimodalVaultIBFV instance.

    Returns:
        Byte string with SHA-256 integrity trailer.
    """
    _import_types()
    ic = vault.iris_commitment
    is_bch = isinstance(ic, _IrisCommitmentBCH)
    flags = FLAG_BCH if is_bch else 0

    # --- HEADER (16 bytes) ---
    header = struct.pack('<4sHHHHHH',
                         MAGIC, VERSION, flags,
                         vault.polynomial_degree,
                         vault.n_genuine_fp,
                         vault.n_bonus,
                         vault.n_chaff)

    # --- PARAMS (16 bytes) ---
    n_vp = len(vault.vault_points)
    fp_len = (vault.field_prime.bit_length() + 7) // 8
    params = struct.pack('<HHHHIHH',
                         vault.block_size,
                         vault.factor,
                         vault.degree,
                         vault.bch_t,
                         n_vp,
                         fp_len,
                         0)  # reserved

    # --- FIELD_PRIME ---
    fp_bytes = _int_to_be(vault.field_prime, fp_len)

    # --- PUBLIC_KEY (64 bytes) ---
    pk_x, pk_y = vault.public_key
    pk_bytes = _int_to_be(pk_x, 32) + _int_to_be(pk_y, 32)

    # --- VAULT_POINTS ---
    vp_parts = []
    for x, y in vault.vault_points:
        vp_parts.append(_int_to_be(x, 32))
        vp_parts.append(_int_to_be(y, 32))
    vp_bytes = b''.join(vp_parts)

    # --- IRIS COMMITMENT ---
    helper = ic.helper_data.ravel()
    mask = ic.enrollment_mask.ravel()
    helper_len = len(helper)
    helper_packed = _pack_bits(helper)
    mask_packed = _pack_bits(mask)
    key_hash_raw = bytes.fromhex(ic.key_hash)

    iris_base = struct.pack('<I', helper_len)
    iris_base += helper_packed
    iris_base += mask_packed
    iris_base += struct.pack('<HH', ic.block_size, ic.n_blocks)
    iris_base += key_hash_raw
    iris_base += struct.pack('<H', ic.key_length_bits)

    iris_bch_ext = b''
    if is_bch:
        vbm = ic.valid_block_mask.ravel().astype(np.uint8)
        vbm_len = len(vbm)
        vbm_packed = _pack_bits(vbm)
        iris_bch_ext = struct.pack('<H', vbm_len) + vbm_packed
        iris_bch_ext += struct.pack('<HHHH',
                                    ic.bch_n, ic.bch_k,
                                    ic.bch_t, ic.min_block_valid)

    # Assemble all sections (before trailer)
    body = header + params + fp_bytes + pk_bytes + vp_bytes + iris_base + iris_bch_ext

    # --- TRAILER (32 bytes) ---
    digest = hashlib.sha256(body).digest()
    return body + digest


def deserialize(data: bytes):
    """
    Deserialize bytes into a MultimodalVaultIBFV.

    Args:
        data: Byte string produced by serialize().

    Returns:
        A MultimodalVaultIBFV instance identical to the original.

    Raises:
        ValueError: If magic, version, or integrity check fails.
    """
    _import_types()

    if len(data) < 16 + 16 + 64 + 32:
        raise ValueError(f"Data too short: {len(data)} bytes")

    # Verify integrity trailer
    body = data[:-32]
    expected_hash = data[-32:]
    actual_hash = hashlib.sha256(body).digest()
    if actual_hash != expected_hash:
        raise ValueError("Integrity check failed: SHA-256 mismatch")

    pos = 0

    # --- HEADER ---
    magic, version, flags, poly_deg, n_gfp, n_bonus, n_chaff = \
        struct.unpack_from('<4sHHHHHH', body, pos)
    pos += 16

    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")
    is_bch = bool(flags & FLAG_BCH)

    # --- PARAMS ---
    blk_sz, factor, degree, bch_t, n_vp, fp_len, _ = \
        struct.unpack_from('<HHHHIHH', body, pos)
    pos += 16

    # --- FIELD_PRIME ---
    field_prime = _be_to_int(body[pos:pos + fp_len])
    pos += fp_len

    # --- PUBLIC_KEY ---
    pk_x = _be_to_int(body[pos:pos + 32])
    pk_y = _be_to_int(body[pos + 32:pos + 64])
    pos += 64

    # --- VAULT_POINTS ---
    vault_points = []
    for _ in range(n_vp):
        vx = _be_to_int(body[pos:pos + 32])
        vy = _be_to_int(body[pos + 32:pos + 64])
        vault_points.append((vx, vy))
        pos += 64

    # --- IRIS_COMMITMENT ---
    helper_len, = struct.unpack_from('<I', body, pos)
    pos += 4
    packed_len = (helper_len + 7) // 8
    helper_data = _unpack_bits(body[pos:pos + packed_len], helper_len)
    pos += packed_len
    enrollment_mask = _unpack_bits(body[pos:pos + packed_len], helper_len)
    pos += packed_len
    iris_bs, iris_n_blocks = struct.unpack_from('<HH', body, pos)
    pos += 4
    key_hash_raw = body[pos:pos + 32]
    key_hash = key_hash_raw.hex()
    pos += 32
    key_len_bits, = struct.unpack_from('<H', body, pos)
    pos += 2

    if is_bch:
        vbm_len, = struct.unpack_from('<H', body, pos)
        pos += 2
        vbm_packed_len = (vbm_len + 7) // 8
        valid_block_mask = _unpack_bits(body[pos:pos + vbm_packed_len], vbm_len)
        pos += vbm_packed_len
        bch_n_iris, bch_k_iris, bch_t_iris, min_block_valid = \
            struct.unpack_from('<HHHH', body, pos)
        pos += 8

        iris_commitment = _IrisCommitmentBCH(
            helper_data=helper_data,
            enrollment_mask=enrollment_mask,
            block_size=iris_bs,
            n_blocks=iris_n_blocks,
            valid_block_mask=valid_block_mask,
            bch_n=bch_n_iris,
            bch_k=bch_k_iris,
            bch_t=bch_t_iris,
            min_block_valid=min_block_valid,
            key_hash=key_hash,
            key_length_bits=key_len_bits,
        )
    else:
        iris_commitment = _IrisCommitment(
            helper_data=helper_data,
            enrollment_mask=enrollment_mask,
            block_size=iris_bs,
            n_blocks=iris_n_blocks,
            key_hash=key_hash,
            key_length_bits=key_len_bits,
        )

    return _MultimodalVaultIBFV(
        polynomial_degree=poly_deg,
        public_key=(pk_x, pk_y),
        vault_points=vault_points,
        n_genuine_fp=n_gfp,
        n_bonus=n_bonus,
        n_chaff=n_chaff,
        field_prime=field_prime,
        iris_commitment=iris_commitment,
        block_size=blk_sz,
        factor=factor,
        degree=degree,
        bch_t=bch_t,
    )
