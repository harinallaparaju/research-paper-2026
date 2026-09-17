import os
import sys
import numpy as np

# --- Architecture A: Decision-Level Fusion Baseline ---
# Mathematically pure independent scoring. Both unimodal systems
# (Fingerprint Vault polynomial interpolation & DL Iris code fractional
# Hamming mapping) must independently satisfy their respective authentications.

# Establish secure path mapping to the senior's unimodal fingerprint extraction
unimodal_path = "/Users/suryanallaparaju/Desktop/Surya's Project/code/unimodal_implementation/2002_1"
sys.path.insert(0, unimodal_path)

try:
    import unlocking
    import quantize2
except ImportError:
    print("[-] Fatal: Could not link to unimodal fingerprint vault mechanism.")
    sys.exit(1)

def fhd(c1, c2):
    """
    Fractional Hamming Distance with cyclical shift correction for planar head tilt.
    """
    if c1 is None or c2 is None: return 1.0
    md = 1.0
    # Search within standard offset deviation (-8 to 8 bits)
    for s in range(-8, 9): 
        md = min(md, np.sum(c1 != np.roll(c2, s)) / len(c1))
    return md

def evaluate_arch_a(db_fvc, iris_dir, factor, degree, mode="1_to_lock_rest_to_unlock", iris_threshold=0.20, max_users=100):
    """"""
    # 1. Path Definition
    vault_dir = os.path.join(unimodal_path, f"vaults/{factor}/vaults{degree}")
    fp_targets = []
    if os.path.exists(db_fvc):
        fp_targets = sorted([f for f in os.listdir(db_fvc) if f.endswith(".txt")], 
                            key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1].split(".")[0])))
        
        # BRUTAL TEST FILTER
        if max_users < 100:
            print(f"[!] Brutal Test Active: Limiting to {max_users} users.")
            fp_targets = [f for f in fp_targets if int(f.split("_")[0]) <= max_users]
                            
    # 2. Tracking Setup
    genuine_attempts = 0
    genuine_accepts = 0
    impostor_attempts = 0
    impostor_accepts = 0

    # 3. Modality Load
    iris_codes = {}
    for user_idx in range(1, 101):
        for imp_idx in range(1, 9):
            k = f"{user_idx}_{imp_idx}.txt"
            p = os.path.join(iris_dir, k)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    iris_codes[k] = np.array([int(x) for x in f.read().strip().split(',')])

    # 4. Matrix Evaluation
    for target in fp_targets:
        target_subject = int(target.split("_")[0])
        target_impression = int(target.split("_")[1].split(".")[0])
        
        # Only impression 1 locks the vault.
        if target_impression != 1:
            continue
            
        locked_iris = iris_codes.get(f"{target_subject}_{target_impression}.txt")
        # Build the exact string expected by Unimodal Vaults (stripping .jpg if present)
        clean_target_name = f"{target_subject}_{target_impression}.txt"
        target_vault_path = os.path.join(vault_dir, f"{clean_target_name}_vault.txt")
        
        if locked_iris is None or not os.path.exists(target_vault_path):
            continue

        for probe in fp_targets:
            probe_subject = int(probe.split("_")[0])
            probe_impression = int(probe.split("_")[1].split(".")[0])
            
            # Select mode
            if mode == "1_to_lock_1_to_unlock" and probe_impression != 2:
                continue
            if mode == "1_to_lock_rest_to_unlock" and probe_impression == 1:
                continue

            probe_fvc = os.path.join(db_fvc, probe)
            # Use cleanly mapped name to load the iris code!
            clean_probe_name = f"{probe_subject}_{probe_impression}.txt"
            probe_iris = iris_codes.get(clean_probe_name)
            if not os.path.exists(probe_fvc) or probe_iris is None:
                continue

            is_genuine = (target_subject == probe_subject)
            if is_genuine:
                genuine_attempts += 1
            else:
                impostor_attempts += 1

            # A: FINGERPRINT MODALITY CHECK
            try:
                # The unimodal mechanism reads integer coordinates first
                with open(probe_fvc, 'r') as file:
                    minutiae = [
                        list(map(int, line.strip().split(',')))
                        for line in file if len(line.strip().split(',')) == 3
                    ]
                filtered = [m for m in minutiae if all(i >= 0 for i in m)]
                
                # `quantize2.read_minutiae` expects the mapped list, not file path
                probe_fp_points = list(set(quantize2.read_minutiae(filtered, factor)))
                
                # Check against target vault locally
                fingerprint_matches = unlocking.check_vault_parallel(probe_fp_points, [target_vault_path])
                fingerprint_match = len(fingerprint_matches) > 0
            except Exception as e:
                fingerprint_match = False
            
            # B: IRIS MODALITY CHECK
            dist = fhd(locked_iris, probe_iris)
            iris_match = dist < iris_threshold
            
            # DECISION LEVEL FUSION (AND)
            if fingerprint_match and iris_match:
                if is_genuine:
                    genuine_accepts += 1
                else:
                    impostor_accepts += 1

    frr = ((genuine_attempts - genuine_accepts) / genuine_attempts * 100) if genuine_attempts > 0 else 0.0
    far = (impostor_accepts / impostor_attempts * 100) if impostor_attempts > 0 else 0.0
    eer = (frr + far) / 2
    
    return {
        "Architecture": f"Arch A ({mode})",
        "Factor": factor,
        "Degree": degree,
        "GA": genuine_attempts,
        "IA": impostor_attempts,
        "GAcc": genuine_accepts,
        "IAcc": impostor_accepts,
        "FRR": frr,
        "FAR": far,
        "EER": eer
    }
