"""
GF(257) Arithmetic — the smallest prime field containing all byte values {0..255}.

Every byte value b ∈ {0, ..., 255} is a valid field element in GF(257).
The field has 257 elements: {0, 1, ..., 256}, with arithmetic mod 257.

Why 257?
    - 257 is prime, so GF(257) is a field (every nonzero element has a
      multiplicative inverse).
    - 257 > 255, so every byte value maps injectively into the field.
    - Byte values 0–255 round-trip without truncation: split a byte as a
      GF(257) element, reconstruct → get the same byte back.
    - 256 is NOT prime (2^8), so GF(256) is a different algebraic structure
      (extension field F_{2^8}) that requires polynomial arithmetic — much
      more complex and slower for our purpose.

Reference:
    Shamir, A. (1979). "How to share a secret." Communications of the ACM.
"""

P = 257  # The field prime


def add(a: int, b: int) -> int:
    """Addition in GF(257)."""
    return (a + b) % P


def sub(a: int, b: int) -> int:
    """Subtraction in GF(257)."""
    return (a - b) % P


def mul(a: int, b: int) -> int:
    """Multiplication in GF(257)."""
    return (a * b) % P


def inv(a: int) -> int:
    """Multiplicative inverse in GF(257) via Fermat's little theorem: a^{-1} = a^{p-2} mod p."""
    if a == 0:
        raise ZeroDivisionError("No inverse for 0 in GF(257)")
    return pow(a, P - 2, P)


def div(a: int, b: int) -> int:
    """Division in GF(257): a / b = a * b^{-1}."""
    return mul(a, inv(b))


def neg(a: int) -> int:
    """Additive inverse in GF(257)."""
    return (P - a) % P


def eval_poly(coeffs: list, x: int) -> int:
    """
    Evaluate polynomial at x using Horner's method in GF(257).

    coeffs[0] is the constant term (the secret), coeffs[-1] is the
    highest-degree coefficient.

    p(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
    """
    result = 0
    for c in reversed(coeffs):
        result = add(mul(result, x), c)
    return result


def lagrange_interpolate_at_zero(points: list) -> int:
    """
    Lagrange interpolation at x = 0 in GF(257).

    Given t points [(x_1, y_1), ..., (x_t, y_t)], returns f(0) where
    f is the unique polynomial of degree < t passing through all points.

    This is the core Shamir reconstruction operation:
        f(0) = Σ_i y_i * Π_{j≠i} (0 - x_j) / (x_i - x_j)
             = Σ_i y_i * Π_{j≠i} (-x_j) / (x_i - x_j)
    """
    t = len(points)
    result = 0
    for i in range(t):
        xi, yi = points[i]
        # Compute Lagrange basis polynomial L_i(0)
        numerator = 1
        denominator = 1
        for j in range(t):
            if j == i:
                continue
            xj = points[j][0]
            numerator = mul(numerator, neg(xj))          # (0 - x_j) = -x_j
            denominator = mul(denominator, sub(xi, xj))  # (x_i - x_j)
        basis = div(numerator, denominator)
        result = add(result, mul(yi, basis))
    return result
