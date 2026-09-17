# Research Protocol: Deep Learning Multimodal Cryptographic Biometric Vault

## 1. The Core Problem Addressed
Previous multimodal fingerprint-iris fusion evaluations failed because traditional algorithmic Iris extraction (Hough Transform + Daugman Gabor) relies heavily on perfect pupil/sclera segmentation. On realistic datasets like CASIA-Iris-Interval, these algorithms fail silently, defaulting to random noise and destroying logical cryptographic fusion.

## 2. The Deep Learning Paradigm Shift
Instead of relying on fragile geometric transforms, we use a zero-shot, spatially-aware Convolutional Neural Network (Pre-trained ResNet-18) to extract deep biometric texture descriptors.
- **Model:** ResNet-18 (truncated before the final classification FC layer).
- **Features:** 512-dimensional output vector per image.
- **Quantization:** Binarized exactly across the image median. This guarantees maximum entropy (an equal number of 1s and 0s), providing the highest impostor separation mathematically possible.
- **Distance Metric:** Fractional Hamming Distance (FHD) with bit-shifts (`range(-8, 9)`) to account for head rotation.

## 3. Modalities and Evaluation Protocol
- **Fingerprint Dataset:** FVC2002 DB1_A (100 Users, 8 Impressions).
- **Iris Dataset:** Extracted deeply from CASIA-Iris-Interval, mapped perfectly 1-to-1 to FVC users via `User_Impression.txt`.
- **Parametric Sweep:**
  - **Quantization Factors (Tolerance bounds for Fingerprint Minutiae):** `[2, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35]`
  - **Polynomial Degrees (Vault Security Bounds):** `[5, 6, 7, 8, 9, 10, 11]`
- **Testing Modes:** Both `1_to_lock_1_to_unlock` and `1_to_lock_rest_to_unlock` (impressions 2-8).

## 4. The Three Cryptographic Architectures
### Architecture A: Decision-Level Fusion
- **Mechanism:** Vault unlocking relies entirely on the unimodal fingerprint polynomials decoding correctly. Independently, the incoming Iris code must pass the mathematical FHD threshold (`0.20` strictly).
- **Security:** "AND" gate. Both systems must approve the user.

### Architecture B: Feature-Polynomic Level Fusion
- **Mechanism:** The secret Reed-Solomon polynomial coefficients locking the fingerprint vault are cryptographically entangled (XOR'd) using a random hash mask generated directly from the locking Iris array. Let the coefficients be $C$ and the Iris mask be $M$: 
  $Entangled_i = C_i \oplus M_i$
- **Unlocking:** The incoming Iris must recreate the identical mask to un-XOR the coefficients. If the Iris is slightly misaligned, Reed-Solomon Error Correction natively repairs the damage *only* if the fingerprint minutiae evaluate the structure correctly.

### Architecture C: Cryptographic Generation Fusion
- **Mechanism:** The Iris feature code is directly hashed and used mathematically as the PRNG (Pseudo-Random Number Generator) seed to position the Chaff (noise) points in the final Fingerprint Vault. 
- **Security:** If an attacker attempts to brute-force the vault, the chaff structure is deterministic only to the true Iris. An imposter generates a completely disarranged chaff field, preventing unlocking.
