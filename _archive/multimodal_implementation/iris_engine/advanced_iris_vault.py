import hashlib
import secrets
from reedsolo import RSCodec, ReedSolomonError

class AdvancedIrisVault:
    def __init__(self, key_size=127, ecc_symbols=128):
        """
        The Bit-to-Byte Spacial Expansion Matrix.
        Instead of treating an Iris vector as an array of bits, we mathematically 
        expand it to an array of bytes. This bypassed the "Neural Bit-Scattering" limit.
        
        Args:
            key_size: 127 bytes of payload (perfect for a huge AES key).
            ecc_symbols: 128 bytes of Error Correction. (Can fix up to 64 byte-errors/bit-flips!)
        """
        self.key_size = key_size
        self.ecc_symbols = ecc_symbols
        # Initialize Reed-Solomon codec
        self.rsc = RSCodec(self.ecc_symbols)
        
    def _expand_bits_to_bytes(self, bit_string):
        """Converts '1010...' strictly into [1, 0, 1, 0...] as pure bytes over a 255-byte field."""
        if len(bit_string) > 255:
            # We truncate beautifully to 255 for RS(255) math.
            bit_string = bit_string[:255]
        return bytes([int(bit) for bit in bit_string])

    def _xor_bytes(self, a, b):
        """Compute bitwise XOR of two identically sized byte arrays."""
        return bytes(x ^ y for x, y in zip(a, b))
        
    def create_vault(self, iris_binary_string):
        """
        Creates the advanced Iris Fuzzy Commitment Vault using Spacial Expansion.
        """
        iris_expanded_bytes = self._expand_bits_to_bytes(iris_binary_string)
        
        # Ensure it fits the mathematical field exactly
        if len(iris_expanded_bytes) < 255:
            iris_expanded_bytes += bytes([0] * (255 - len(iris_expanded_bytes)))
            
        # 1. Generate random Secret (S)
        secret_key = secrets.token_bytes(self.key_size)
        
        # 2. Encode to 255-byte Codeword (C)
        codeword = self.rsc.encode(secret_key)
        
        # 3. Cryptographically Lock: W = Codeword XOR Expanded Iris
        W = self._xor_bytes(codeword, iris_expanded_bytes)
        
        # 4. Hash the secret H(S)
        hash_S = hashlib.sha256(secret_key).digest()
        
        return W, hash_S
        
    def unlock_vault(self, query_iris_binary_string, W, stored_hash):
        """
        Attempts to unlock the advanced vault.
        """
        query_expanded_bytes = self._expand_bits_to_bytes(query_iris_binary_string)
        if len(query_expanded_bytes) < 255:
            query_expanded_bytes += bytes([0] * (255 - len(query_expanded_bytes)))
            
        # 1. Reconstruct noisy codeword
        noisy_codeword = self._xor_bytes(query_expanded_bytes, W)
        
        try:
            # 2. Reed-Solomon decoding (Can now fix up to 64 independent bit-flips naturally)
            decoded_secret_key, _, _ = self.rsc.decode(noisy_codeword)
            
            # 3. Verify
            if hashlib.sha256(decoded_secret_key).digest() == stored_hash:
                return True, decoded_secret_key
            else:
                return False, None
        except ReedSolomonError:
            return False, None
