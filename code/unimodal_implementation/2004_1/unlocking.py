import tinyec.ec as ec
import tinyec.registry as reg
import itertools
import os

# Caches to avoid redundant computation
vault_cache = {}
transformed_points_cache = {}
curve_cache= {}


def find_count():
    print(len(vault_cache))
    print(vault_cache.keys())

# Step 1: Parse a vault file to extract curve parameters and vault points
def parse_vault_file(vault_file):
    with open(vault_file, "r") as vf:
        lines = vf.readlines()

    curve_name = lines[0].strip().split(": ")[1]
    field_prime = int(lines[1].strip().split(": ")[1])
    base_point_coords = tuple(map(int, lines[2].strip().split(": ")[1].strip("()").split(", ")))
    polynomial_degree = int(lines[3].strip().split(": ")[1])
    public_key_coords = tuple(map(int, lines[5].strip().split(": ")[1].strip("()").split(", ")))

    vault_points = []
    for line in lines[6:]:
        x, y = map(int, line.strip().split(","))
        vault_points.append((x, y))

    return curve_name, field_prime, base_point_coords, polynomial_degree, public_key_coords, vault_points

# Step 2: Initialize elliptic curve and base point
def initialize_curve(curve_name, field_prime, base_point_coords):
    curve = reg.get_curve(curve_name)
    base_point = ec.Point(curve, *base_point_coords)

    if (base_point.y**2 - base_point.x**3 - curve.a * base_point.x - curve.b) % curve.field.p != 0:
        raise ValueError("Base point is not on the curve.")

    return curve, base_point

# Step 3: Transform minutiae points into EC points
def transform_points(points, curve, base_point):
    transformed_points = []
    for integer in points:
        point = None
        if integer < 0 or integer >= curve.field.p:
            raise ValueError(f"Integer {integer} is out of range for the curve field.")
        if integer not in curve_cache:
            point = integer * base_point
            curve_cache[integer] = point
        else:
            point = curve_cache[integer]
        transformed_points.append((point.x, point.y))
    return transformed_points

# Step 4: Polynomial reconstruction
def mod_inverse(a, p):
    return pow(a, -1, p)

def lagrange_interpolation_numeric(points, prime):
    n = len(points)
    if len(set(x for x, _ in points)) != n:
        raise ValueError("Duplicate x-coordinates in points.")
    coeffs = [0] * n
    for i, (xi, yi) in enumerate(points):
        li_coeffs = [1]
        denominator = 1
        for j, (xj, _) in enumerate(points):
            if i != j:
                denominator = (denominator * (xi - xj)) % prime
                li_coeffs.append(0)
                li_coeffs2 = [(-xj * coeff) % prime for coeff in li_coeffs[:-1]]
                for k in range(len(li_coeffs) - 1):
                    li_coeffs[k+1] = (li_coeffs2[k] + li_coeffs[k+1]) % prime
        denominator_inv = mod_inverse(denominator, prime)
        li_coeffs = [(coeff * denominator_inv * yi) % prime for coeff in li_coeffs]
        for k in range(len(coeffs)):
            coeffs[k] = (coeffs[k] + li_coeffs[k]) % prime
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs[::-1]

def evaluate_polynomial(coefficients, x, p):
    return sum(coeff * pow(x, i, p) for i, coeff in enumerate(coefficients)) % p

def reconstruct_secret_key(coefficients):
    binary_chunks = [format(i, "06b") for i in coefficients]
    return int("".join(binary_chunks), 2)

# Step 5: Vault Unlock Attempt
def unlock_vault(transformed_points, vault_points, polynomial_degree, public_key_coords, curve, base_point):
    matching_points = [(x, y) for x, y in vault_points if x in {tp[0] for tp in transformed_points}]
    for combination in itertools.combinations(matching_points, polynomial_degree + 1):
        try:
            reconstructed_polynomial = lagrange_interpolation_numeric(combination, curve.field.p)
            genuine_points = [(x, evaluate_polynomial(reconstructed_polynomial, x, curve.field.p)) for x, _ in combination]
            if all(point in vault_points for point in genuine_points):
                secret_key = reconstruct_secret_key(reconstructed_polynomial) % curve.field.p
                expected_public_key = secret_key * base_point
                if (expected_public_key.x, expected_public_key.y) == public_key_coords:
                    return True, secret_key
        except Exception:
            continue
    return False, None

# Step 6: Vault Checking
def check_vault_parallel(input_file, vault_files):
    input_key = tuple(input_file)

    def process_vault(vault_file):
        if vault_file not in vault_cache:
            try:
                curve_name, field_prime, base_point_coords, polynomial_degree, public_key_coords, vault_points = parse_vault_file(vault_file)
                curve, base_point = initialize_curve(curve_name, field_prime, base_point_coords)
                vault_cache[vault_file] = {
                    "curve": curve,
                    "base_point": base_point,
                    "degree": polynomial_degree,
                    "public_key": public_key_coords,
                    "vault_points": vault_points
                }
            except Exception as e:
                print(f"[ERROR] Failed to cache vault {vault_file}: {e}")
                return None, None

        data = vault_cache[vault_file]
        try:
            if input_key not in transformed_points_cache:
                transformed_points_cache[input_key] = transform_points(input_file, data["curve"], data["base_point"])

            transformed_points = transformed_points_cache[input_key]
            success, secret_key = unlock_vault(transformed_points, data["vault_points"], data["degree"], data["public_key"], data["curve"], data["base_point"])
            if success:
                return vault_file, secret_key
        except Exception as e:
            print(f"[ERROR] Processing {vault_file}: {e}")
        return None, None

    results = [process_vault(vf) for vf in vault_files]
    matching_vaults = [vault_file for vault_file, _ in results if vault_file]
    return matching_vaults

# Step 7: Main Interface for External Use
def main(input_file, vault_dir):
    vault_files = [os.path.join(vault_dir, f) for f in os.listdir(vault_dir) if f.endswith(".txt")]
    return check_vault_parallel(input_file, vault_files)
