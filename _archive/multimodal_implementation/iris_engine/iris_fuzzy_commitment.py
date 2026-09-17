import os
import hashlib
import secrets
from reedsolo import RSCodec, ReedSolomonError

class IrisFuzzyCommitment:
    def __init__(self, key_size=16, ecc_symbols=16):
        """
        Initializes the Reed-Solomon Fuzzy Commitment scheme.
        
        Args:
            key_size: Number of bytes for the cryptographic Secret (S). 16 bytes = 128-bit AES key.
            ecc_symbols: Number of Error Correcting Code (ECC) symbols. 
                         RS can correct (ecc_symbols // 2) byte errors.
                         With 16 ECC bytes, it can mathematically correct up to 8 byte-level errors.
        """
        # Ensure the total size fits within our 256-bit (32 bytes) DL Iris extracted codes
        if key_size + ecc_symbols != 32:
            raise ValueError("Key size + ECC symbols must equal 32 bytes (256 bits) to match the Iris vector.")
            
        self.key_size = key_size
        self.ecc_symbols = ecc_symbols
        # Initialize Reed-Solomon codec
        self.rsc = RSCodec(self.ecc_symbols)
        
    def _bits_to_bytes(self, bit_string):
        """Convert a 256-length string of '1's and '0's into exactly 32 bytes."""
        if len(bit_string) != 256:
            raise ValueError(f"Expected exactly 256 bits, got {len(bit_string)}")
        return int(bit_string, 2).to_bytes(32, byteorder='big')

    def _xor_bytes(self, a, b):
        """Compute bitwise XOR of two byte arrays."""
        return bytes(x ^ y for x, y in zip(a, b))
        
    def create_vault(self, iris_binary_string):
        """
        Creates the Iris Fuzzy Commitment Vault.
        
        Outputs:
            W: The obscured codeword (XOR of Iris and Reed-Solomon codeword).
            hash_S: SHA-256 hash of the generated secret, used for verification.
        """
        iris_bytes = self._bits_to_bytes(iris_binary_string)
        
        # 1. Generate a purely random 16-byte cryptographic key (Secret S)
        secret_key = secrets.token_bytes(self.key_size)
        
        # 2. Encode the Secret S with Reed-Solomon to get a 32-byte Codeword (C)
        codeword = self.rsc.encode(secret_key)
        
        # 3. Create the protected template W = Codeword XOR Iris
        W = self._xor_bytes(codeword, iris_bytes)
        
        # 4. Hash the secret H(S)
        hash_S = hashlib.sha256(secret_key).digest()
        
        # The stored vault consists of (W, hash_S)
        return W, hash_S
        
    def unlock_vault(self, query_iris_binary_string, W, stored_hash):
        """
        Attempts to unlock the vault using a new Iris sample.
        
        Returns:
            Success (bool), Recovered Secret (bytes or None)
        """
        query_bytes = self._bits_to_bytes(query_iris_binary_string)
        
        # 1. Reconstruct the noisy codeword: C' = query_iris XOR W
        # If query_iris is close to the original iris, C' will be very close to original Codeword C.
        noisy_codeword = self._xor_bytes(query_bytes, W)
        
        try:
            # 2. Attempt Reed-Solomon error correction
            decoded_secret_key, decoded_full, _ = self.rsc.decode(noisy_codeword)
            
            # 3. Verify the hash
            if hashlib.sha256(decoded_secret_key).digest() == stored_hash:
                return True, decoded_secret_key
            else:
                return False, None
        except ReedSolomonError:
            # Too many errors for Reed-Solomon to mathematically fix. Authentication fails.
            return False, None

if __name__ == "__main__":
    print("Testing Iris Fuzzy Commitment (Reed-Solomon 32-byte framework)...")
    
    # Simulate a perfect 256-bit extraction from our Deep Learning model
    template_iris = "10101010" * 32
    
    fc = IrisFuzzyCommitment(key_size=16, ecc_symbols=16)
    
    # User Enroll:
    W, hash_S = fc.create_vault(template_iris)
    print("\n--- ENROLLMENT ---")
    print("Vault created successfully.")
    print(f"Obscured Template (W) length: {len(W)} bytes")
    print(f"Secret Hash length: {len(hash_S)} bytes")
    
    print("\n--- AUTHENTICATION ---")
    # Test 1: Exact Match (0 errors)
    success, key = fc.unlock_vault(template_iris, W, hash_S)
    print(f"Testing Exact Match -> Success: {success}")
    
    # Test 2: User variation (e.g. 25% errors - up to 8 full bytes corrupted)
    # Let's completely flip the first 8 bytes
    error_iris = "01010101" * 8 + "10101010" * 24
    success_noisy, key_noisy = fc.unlock_vault(error_iris, W, hash_S)
    print(f"Testing Match with ~25% max allowed errors -> Success: {success_noisy}")
    if success_noisy:
         print("  -> Reed-Solomon successfully erased the biometric noise!")
    
    # Test 3: Attacker (too many errors - 9 bytes corrupted)
    error_iris_bad = "01010101" * 9 + "10101010" * 23
    success_bad, _ = fc.unlock_vault(error_iris_bad, W, hash_S)
    print(f"Testing Hacker/Mismatch (Too many errors) -> Success: {success_bad}")
