import tinyec.ec as ec
import tinyec.registry as reg
import random
import os


curve_cache = {}
# Step 1: Define the elliptic curve
def get_elliptic_curve():
    curve = reg.get_curve("brainpoolP256r1")
    return curve

# Step 2: Select a base point G of large order
def get_base_point(curve):
    return curve.g

# Step 3: Generate a random 6-bit integer
def random_6_bit_number():
    return random.randint(10, 63)  # 6-bit number is in the range [0, 63]

# Step 4: Generate coefficients and construct the secret key
def construct_coefficients_and_secret_key(degree):
    coefficients = []
    binary_chunks = []
    
    # Generate random 6-bit integers for each coefficient
    for _ in range(degree + 1):
        coeff = random_6_bit_number()
        coefficients.append(coeff)
        binary_chunks.append(format(coeff, "06b"))  # Convert to 6-bit binary string
    
    # Concatenate all 6-bit chunks to form the secret key
    secret_key_binary = "".join(binary_chunks)
    secret_key = int(secret_key_binary, 2)  # Convert binary string to integer
    return coefficients, secret_key

# Step 5: Evaluate the polynomial at a given x
def evaluate_polynomial(coefficients, x, p):
    return sum(coeff * pow(x, i, p) for i, coeff in enumerate(coefficients)) % p

# Step 6: Create the fuzzy vault
def create_fuzzy_vault(points, output_file, curve, base_point, coefficients, secret_key, num_chaff=200):
    vault = []
    public_key = secret_key * base_point
    # Write the curve details, polynomial degree, and public key to the output file
    with open(output_file, "w") as outfile:
        outfile.write(f"Elliptic Curve: {curve.name}\n")
        outfile.write(f"Field Prime: {curve.field.p}\n")
        outfile.write(f"Base Point: ({base_point.x}, {base_point.y})\n")
        degree = len(coefficients) - 1
        outfile.write(f"Polynomial Degree: {degree}\n")
        outfile.write(f"Polynomial Coefficients: {coefficients}\n")
        outfile.write(f"Public Key: ({public_key.x}, {public_key.y})\n")
        # Process input points and create genuine points
        
        for line in points:
            integer = int(line)
            if integer < 0 or integer >= curve.field.p:
                raise ValueError(f"Integer {integer} is out of range for the curve field.")

            # Transform integer using elliptic curve (scalar multiplication)
            point= None
            if integer not in curve_cache:
                point = integer * base_point
                curve_cache[integer] = point
            else:
                point = curve_cache[integer]
                #transformed_points.append((point.x, point.y))
            #return transformed_points

            #point = integer * base_point
            x = point.x
            y = evaluate_polynomial(coefficients, x, curve.field.p)  # Evaluate polynomial

            # Write genuine point to the vault
            outfile.write(f"{x}, {y}\n")
            vault.append((x, y))

        # Add chaff points
        chaff_points = 0
        while chaff_points < num_chaff:
            x = random.randint(1, curve.field.p - 1)
            y = random.randint(1, curve.field.p - 1)

            # Ensure the point is not on the polynomial
            if evaluate_polynomial(coefficients, x, curve.field.p) != y:
                outfile.write(f"{x}, {y}\n")
                vault.append((x, y))
                chaff_points += 1

    #print(f"Fuzzy vault created and saved to {output_file}\n\n\n")

# Main execution
def main(points,file_name,factor,per):
    #input_file = os.path.join(file_path, file_name)
    base_name = os.path.splitext(file_name)[0]
    new_name = file_name + "_vault.txt"
    #print(new_name)
    # Get the elliptic curve and base point
    curve = get_elliptic_curve()
    base_point = get_base_point(curve)

    # Read the total number of points from the input file
    
    total_points = len(points)

    for percent in range(per[0],per[1],per[2]):

        #percent = i*2
        directory = "vaults" + "/" + factor + "/" +"vaults{}".format(percent)
        if not os.path.exists(directory):
            os.makedirs(directory)  # Creates the directory (and parent directories if needed)
    
        output_file =  directory+ "/" + new_name



        # Calculate the degree of the polynomial (70% of total points)
        #degree = int((percent/100) * total_points)
        degree = percent
        #print(degree)

        # Construct the coefficients and the secret key
        coefficients, secret_key = construct_coefficients_and_secret_key(degree)
        secret_key= secret_key % curve.field.p

        # Display the generated secret key
        #print(f"Generated Secret Key: {secret_key}")
        #print(f"Coefficients: {coefficients}")

        # Create the fuzzy vault
        create_fuzzy_vault(points, output_file, curve, base_point, coefficients,secret_key,num_chaff= total_points*5)

if __name__ == "__main__":
    file_path = "transformed2"  # Path to the input file
    file_name = "sample_input.txt"  # Input file containing integers
    main(file_path, file_name)
