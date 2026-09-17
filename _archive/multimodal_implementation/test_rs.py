from reedsolo import RSCodec
import os
import random

rsc = RSCodec(200) # Can correct 100 corrupted bytes!
secret = os.urandom(55) # 255 - 200 = 55 bytes
codeword = rsc.encode(secret)
print(f"Codeword len: {len(codeword)}")

# Corrupt exactly 80 bytes
corrupted = bytearray(codeword)
for i in random.sample(range(255), 80):
    corrupted[i] = (corrupted[i] + 1) % 256

decoded = rsc.decode(corrupted)[0]
print(f"Decoded successfully! Match: {decoded == secret}")
