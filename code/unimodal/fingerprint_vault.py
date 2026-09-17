"""
Fingerprint Fuzzy Vault — Clean Implementation.

Wraps the core ECC-based fuzzy vault algorithm into a clean, modular API.
The underlying algorithm is from the published JISAA 2025 paper.

Pipeline:
    1. Minutiae (x, y, θ) → Quantize by factor f → 18-bit integers
    2. Each integer → scalar multiply on brainpoolP256r1 → EC point
    3. Random degree-k polynomial with 6-bit coefficients
    4. Secret key = binary concatenation of coefficients → int mod p
    5. Vault = {(P.x, poly(P.x)) for each genuine P} ∪ {chaff points}
    6. Public key = secret_key × G (for verification)

Unlocking:
    1. Query minutiae → same quantization + EC transform
    2. Find matching x-coordinates in vault
    3. Lagrange interpolation on k+1 points → recover polynomial
    4. Reconstruct secret key → check public_key == secret_key × G
"""

import random
import itertools
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field

import tinyec.ec as ec
import tinyec.registry as reg


# ---------- ECC setup ----------

_CURVE = None
_BASE_POINT = None
_EC_CACHE = {}  # scalar → EC point


def _get_curve():
    global _CURVE, _BASE_POINT
    if _CURVE is None:
        _CURVE = reg.get_curve("brainpoolP256r1")
        _BASE_POINT = _CURVE.g
    return _CURVE, _BASE_POINT


def _ec_multiply(scalar: int) -> ec.Point:
    """Cached scalar multiplication: scalar × G."""
    if scalar not in _EC_CACHE:
        _, G = _get_curve()
        _EC_CACHE[scalar] = scalar * G
    return _EC_CACHE[scalar]


# ---------- Quantization ----------

def quantize_minutiae(
    minutiae: List[Tuple[int, int, int]],
    factor: int,
) -> Set[int]:
    """
    Quantize minutiae (x, y, θ) → set of 18-bit integers.

    Each component is floor-divided by factor, clipped to 6 bits,
    and concatenated: binary(x//f, 6) || binary(y//f, 6) || binary(θ//f, 6).

    Args:
        minutiae: List of (x, y, theta) tuples
        factor: Quantization divisor

    Returns:
        Set of unique quantized integer values
    """
    result = set()
    for x, y, t in minutiae:
        if x < 0 or y < 0 or t < 0:
            continue
        qx = min(x // factor, 63)  # Clip to 6 bits
        qy = min(y // factor, 63)
        qt = min(t // factor, 63)
        binary_str = f"{qx:06b}{qy:06b}{qt:06b}"
        result.add(int(binary_str, 2))
    return result


# ---------- Vault data ----------

@dataclass
class VaultData:
    """Complete vault state for storage/transmission."""
    polynomial_degree: int
    coefficients: List[int]         # k+1 coefficients (6-bit each)
    secret_key: int                 # secret mod p
    public_key: Tuple[int, int]     # (x, y) of secret_key × G
    vault_points: List[Tuple[int, int]]  # genuine + chaff (x, y) pairs
    n_genuine: int
    n_chaff: int
    field_prime: int


# ---------- Vault creation ----------

def create_vault(
    quantized_minutiae: Set[int],
    degree: int,
    chaff_multiplier: int = 5,
    seed: Optional[int] = None,
) -> VaultData:
    """
    Create a fuzzy vault from quantized minutiae.

    Args:
        quantized_minutiae: Set of quantized integer values
        degree: Polynomial degree k (need k+1 points to reconstruct)
        chaff_multiplier: Number of chaff points per genuine point
        seed: Optional random seed for reproducibility

    Returns:
        VaultData containing the complete vault
    """
    if seed is not None:
        random.seed(seed)

    curve, G = _get_curve()
    p = curve.field.p

    # Generate random polynomial coefficients (6-bit each, [10, 63])
    coefficients = [random.randint(10, 63) for _ in range(degree + 1)]

    # Construct secret key from coefficient bits
    binary_chunks = [f"{c:06b}" for c in coefficients]
    secret_key = int("".join(binary_chunks), 2) % p

    # Public key for verification
    public_key_point = secret_key * G
    public_key = (public_key_point.x, public_key_point.y)

    # Generate genuine vault points
    vault_points = []
    for m in quantized_minutiae:
        point = _ec_multiply(m)
        x = point.x
        y = _eval_poly(coefficients, x, p)
        vault_points.append((x, y))

    n_genuine = len(vault_points)

    # Generate chaff points
    n_chaff = n_genuine * chaff_multiplier
    chaff_count = 0
    while chaff_count < n_chaff:
        cx = random.randint(1, p - 1)
        cy = random.randint(1, p - 1)
        if _eval_poly(coefficients, cx, p) != cy:
            vault_points.append((cx, cy))
            chaff_count += 1

    # Shuffle to hide genuine/chaff order
    random.shuffle(vault_points)

    return VaultData(
        polynomial_degree=degree,
        coefficients=coefficients,
        secret_key=secret_key,
        public_key=public_key,
        vault_points=vault_points,
        n_genuine=n_genuine,
        n_chaff=n_chaff,
        field_prime=p,
    )


# ---------- Vault unlocking ----------

def unlock_vault(
    quantized_minutiae: Set[int],
    vault: VaultData,
) -> bool:
    """
    Attempt to unlock a vault using query minutiae.

    Finds matching x-coordinates, tries all (k+1)-subsets via
    Lagrange interpolation, verifies secret key against public key.

    Args:
        quantized_minutiae: Set of quantized integer values from query
        vault: The vault to unlock

    Returns:
        True if vault was successfully unlocked
    """
    curve, G = _get_curve()
    p = vault.field_prime
    k = vault.polynomial_degree

    # Transform query minutiae to EC x-coordinates
    query_x_set = set()
    for m in quantized_minutiae:
        point = _ec_multiply(m)
        query_x_set.add(point.x)

    # Find matching vault points
    matching = [(x, y) for x, y in vault.vault_points if x in query_x_set]

    if len(matching) < k + 1:
        return False

    # Try all (k+1)-subsets of matching points
    for combo in itertools.combinations(matching, k + 1):
        try:
            reconstructed = _lagrange_interpolation(list(combo), p)
            if reconstructed is None:
                continue

            # Verify: reconstruct secret key and check public key
            secret = _reconstruct_secret(reconstructed) % p
            expected_pk = secret * G

            if (expected_pk.x, expected_pk.y) == vault.public_key:
                return True
        except Exception:
            continue

    return False


# ---------- Helper functions ----------

def _eval_poly(coefficients: List[int], x: int, p: int) -> int:
    """Evaluate polynomial at x mod p. coefficients[i] = coeff of x^i."""
    result = 0
    for i, c in enumerate(coefficients):
        result = (result + c * pow(x, i, p)) % p
    return result


def _lagrange_interpolation(
    points: List[Tuple[int, int]], p: int
) -> Optional[List[int]]:
    """
    Lagrange interpolation over GF(p).

    Args:
        points: List of (x, y) pairs
        p: Field prime

    Returns:
        List of polynomial coefficients [c0, c1, ..., ck] or None on failure
    """
    n = len(points)
    if len(set(x for x, _ in points)) != n:
        return None  # Duplicate x-coordinates

    coeffs = [0] * n
    for i, (xi, yi) in enumerate(points):
        # Compute Lagrange basis polynomial L_i
        li_coeffs = [1]
        denom = 1
        for j, (xj, _) in enumerate(points):
            if i == j:
                continue
            denom = (denom * (xi - xj)) % p

            # Multiply by (x - xj)
            new_li = [0] * (len(li_coeffs) + 1)
            for k_idx in range(len(li_coeffs)):
                new_li[k_idx] = (new_li[k_idx] + li_coeffs[k_idx] * (-xj)) % p
                new_li[k_idx + 1] = (new_li[k_idx + 1] + li_coeffs[k_idx]) % p
            li_coeffs = new_li

        denom_inv = pow(denom, -1, p)
        for k_idx in range(len(coeffs)):
            coeffs[k_idx] = (coeffs[k_idx] + li_coeffs[k_idx] * denom_inv * yi) % p

    return coeffs


def _reconstruct_secret(coefficients: List[int]) -> int:
    """Reconstruct secret key from polynomial coefficients."""
    binary_chunks = [f"{c % 64:06b}" for c in coefficients]
    return int("".join(binary_chunks), 2)


# ---------- I/O for legacy vault files ----------

def save_vault_file(vault: VaultData, filepath: str):
    """Save vault in legacy text format (compatible with senior's code)."""
    curve, G = _get_curve()
    with open(filepath, "w") as f:
        f.write(f"Elliptic Curve: {curve.name}\n")
        f.write(f"Field Prime: {vault.field_prime}\n")
        f.write(f"Base Point: ({G.x}, {G.y})\n")
        f.write(f"Polynomial Degree: {vault.polynomial_degree}\n")
        f.write(f"Polynomial Coefficients: {vault.coefficients}\n")
        f.write(f"Public Key: ({vault.public_key[0]}, {vault.public_key[1]})\n")
        for x, y in vault.vault_points:
            f.write(f"{x}, {y}\n")


def load_vault_file(filepath: str) -> VaultData:
    """Load vault from legacy text format."""
    with open(filepath) as f:
        lines = f.readlines()

    field_prime = int(lines[1].strip().split(": ")[1])
    degree = int(lines[3].strip().split(": ")[1])
    coefficients = eval(lines[4].strip().split(": ")[1])
    pk_str = lines[5].strip().split(": ")[1].strip("()")
    public_key = tuple(map(int, pk_str.split(", ")))

    vault_points = []
    for line in lines[6:]:
        parts = line.strip().split(",")
        if len(parts) == 2:
            vault_points.append((int(parts[0].strip()), int(parts[1].strip())))

    return VaultData(
        polynomial_degree=degree,
        coefficients=coefficients,
        secret_key=0,  # Not stored in file
        public_key=public_key,
        vault_points=vault_points,
        n_genuine=0,   # Unknown from file
        n_chaff=0,
        field_prime=field_prime,
    )


# ---------- Read minutiae file ----------

def read_minutiae_file(filepath: str) -> List[Tuple[int, int, int]]:
    """Read minutiae from a CSV file (x, y, theta per line)."""
    minutiae = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                x, y, t = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
                if x >= 0 and y >= 0 and t >= 0:
                    minutiae.append((x, y, t))
    return minutiae
