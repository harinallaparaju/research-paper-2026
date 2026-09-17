# Multi-Biometric Fingerprint and Iris Fuzzy Vault Project Log

## Project Objective
The central objective is to extend an existing, published unimodal ECC-based Fingerprint Fuzzy Vault system (developed by a senior researcher) into a novel multimodal biometric framework incorporating Iris recognition. The goal is to design, implement, rigorously test, and formally document multiple architectural fusions (Decision-Level, Feature-Level, and Cryptographic) using the FVC2002 fingerprint database and the CASIA-Iris-Interval database.

## System Components

### 1. Unimodal Baseline (The Senior's Implementation)
- **Methodology:** Reed-Solomon polynomial fuzzy vault using quantized fingerprint minutiae.
- **Execution Flow:**
  - `quantize2.py`: Maps spatial minutiae coordinates (x, y, theta) into 18-bit binary strings based on a quantization factor (e.g., 18).
  - `main2.4.py` / `vault_create2.py`: Creates the fuzzy vault. Uses FVC 2002 impression `1_1` (first impression) to lock the vault. Generates a secret polynomial, binds the minutiae points to it, and adds random chaff points.
  - `main3.py` / `unlocking.py`: Attempts to unlock the vault. Uses subsequent impressions (e.g., `1_2` to `1_8`) as probes. It identifies candidate points and uses Reed-Solomon decoding (`check_vault_parallel`) to recover the polynomial if enough genuine points are found.
- **Current Status:** The baseline code exists in `/code/unimodal_implementation/2002_1/`. The immediate uncompleted task is to execute this unimodal code in isolation to replicate the senior's exact FAR, FRR, and EER metrics from the published paper.

### 2. Iris Extraction Engine
- **Methodology:** Extract binary feature vectors from Iris images.
- **Historical Context:** The project originally utilized a pre-extracted dataset of 1,054 highly accurate Iris codes located in `/data/Iris/Extracted_Codes`.
- **Recent Development (and Failure):** An attempt was made to implement a custom Daugman-based extractor (`iris_extractor_v3.py`). However, this extractor failed segmentation on the specific contrast of the CASIA datasets, outputting essentially random noise (Fractional Hamming Distance ~0.49 for genuine matches).
- **Corrective Action Taken:** The system was reverted to use the original, reliable `/Extracted_Codes` dataset.
- **Future Plan:** Return to first principles. Re-evaluate the Iris extraction pipeline to ensure we can reliably and accurately generate feature vectors from raw images if needed, or formally validate the existing 1,054 codes.

### 3. Multimodal Architectures (The Novel Extension)

We designed three distinct fusion architectures, heavily relying on the unimodal Reed-Solomon physical check (`unlocking.check_vault_parallel`).

#### Architecture A: Decision-Level Fusion
- **Mechanism:** Both modalities are evaluated completely independently. The Fingerprint Vault must be unlocked via Reed-Solomon decoding, AND the Iris code must match the enrolled code within a specified Fractional Hamming Distance (FHD) threshold.
- **Status:** Implemented and mathematically verified.

#### Architecture B: Feature-Level / Polynomial Modification Fusion
- **Mechanism (Intended):** The Iris code generates a cryptographic mask that is XOR'd directly against the raw coefficients of the secret fingerprint polynomial *before* chaff generation. Unlocking requires applying the probe Iris mask to un-XOR the coefficients before Reed-Solomon decoding can even begin.
- **Mechanism (Current Interim State):** Due to implementation complexities and logic shortcuts that bypassed the Reed-Solomon check, the current script (`Architecture_B.py`) simulates this by enforcing strict physical vault unlocking *alongside* strict Iris bounds.
- **Future Plan:** True code-level entanglement must be implemented where the Iris bits physically alter the mathematical structure of the vault.

#### Architecture C: Cryptographic Seed Chaffing
- **Mechanism (Intended):** The pseudorandom number generator (PRNG) that creates the chaff points for the Fingerprint Vault is seeded directly by the Iris binary vector. Without the correct Iris code, the attacker cannot reproduce the chaff distribution sequence.
- **Future Plan:** This requires deeply modifying `vault_create2.py` to accept the Iris vector as the definitive entropy source for chaff generation.

### 4. Evaluation Framework
- **Script:** `3_Run_Parametric_Evaluation_Parallel.py`
- **Purpose:** A multiprocessed, parallel parametric sweep that tests all combinations of quantization `factor` (e.g., 16, 18, 20) and polynomial `degree` (e.g., 5, 6, 7) across all architectures to generate comprehensive False Acceptance Rate (FAR), False Rejection Rate (FRR), and Equal Error Rate (EER) data.
- **Status:** Functional, but relies on the architectural files being mathematically sound.

## Critical Incident & Resolution

- **The Issue:** The project workspace became cluttered with legacy evaluation logic. A cleanup was performed, but Architectures B and C were running corrupted logic. They were bypassing the physical Reed-Solomon check (`unlocking.check_vault`) and instead simply counting minutiae points. When paired with the broken Daugman Iris noise, this caused the FAR to spike anomalously (e.g., to 32%).
- **The Fix:** The metrics and logic were heavily analyzed. The architectures were strictly rewritten to force the execution of `unlocking.check_vault_parallel`. When tested locally, this successfully forced the FAR back down to 0%, proving the physical mathematics hold under strict testing.

## Immediate Action Plan (The "First Principles" Reset)

To ensure brutal accuracy moving forward, the following sequence was established:

1. **Isolate and Replicate Unimodal Baseline:** Entirely ignore multimodal fusion temporarily. Run `main2.4.py` and `main3.py` on FVC2002 Db1_a. Document the exact metrics. Compare these against the senior's published paper to ensure the foundation is unshakeable.
2. **Reconstruct Iris Integration:** Thoroughly validate the Iris extraction (`Extracted_Codes`). Determine exactly how many users map cleanly to the 100 FVC users.
3. **Engineer True Cryptographic Fusion:** Move away from logic-gate simulations (like Architecture A) and physically write the Python code that allows Iris arrays to structurally mutate the Fingerprint polynomial equations (Architectures B and C).
4. **Brainstorm Novel Combinations:** Explore further intersections (e.g., utilizing feature points from Iris to dynamically alter the quantization factor of the Fingerprint).
5. **Final Evaluation:** Finally, re-run the massive parallel sweep to generate publication-ready tables.