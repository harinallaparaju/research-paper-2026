"""Quick timing sanity test and fix verification."""
import time, json, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import read_minutiae_file, quantize_minutiae, create_vault, unlock_vault
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv

CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"

# Load DB1 Setup 1 (100 subjects)
with open(CHIMERIC_DIR / "chimeric_fvc2002_1_min2.json") as f:
    mapping = json.load(f)

subjects = []
for entry in mapping["mapping"]:
    subj = {"chimeric_id": entry["chimeric_id"], "minutiae": {}, "iris": {}}
    for pair in entry["pairs"]:
        imp = pair["impression"] - 1
        subj["minutiae"][imp] = read_minutiae_file(pair["fp_file"])
        iris_data = np.load(pair["iris_file"])
        subj["iris"][imp] = {"code": iris_data["code"], "mask": iris_data["mask"]}
    subjects.append(subj)

print(f"Loaded {len(subjects)} subjects")

# Test with f=24, k=10 (the senior's DB1 best params)
factor, degree = 24, 10

# --- Unimodal ---
t0 = time.time()
vaults = {}
for subj in subjects:
    q = quantize_minutiae(subj["minutiae"].get(0, []), factor)
    if len(q) >= degree + 1:
        vaults[subj["chimeric_id"]] = create_vault(q, degree, seed=subj["chimeric_id"])

ga, ia, gacc, iacc = 0, 0, 0, 0
for test_subj in subjects:
    test_q = quantize_minutiae(test_subj["minutiae"].get(1, []), factor)
    for vault_cid, vault in vaults.items():
        success = unlock_vault(test_q, vault)
        if vault_cid == test_subj["chimeric_id"]:
            ga += 1; gacc += int(success)
        else:
            ia += 1; iacc += int(success)
t_uni = time.time() - t0

far = iacc / ia * 100 if ia else 0
frr = (ga - gacc) / ga * 100 if ga else 0
print(f"Unimodal f={factor} k={degree}: FAR={far:.4f}% FRR={frr:.4f}% EER={(far+frr)/2:.4f}% | {t_uni:.1f}s | GA={ga} IA={ia}")

# --- IBFV ---
t0 = time.time()
ibfv_vaults = {}
for subj in subjects:
    mins = subj["minutiae"].get(0, [])
    iris = subj["iris"].get(0)
    if mins and iris:
        v = lock_vault_ibfv(mins, iris["code"], iris["mask"], factor, degree,
                            n_bonus=4, block_size=63, seed=subj["chimeric_id"])
        if v:
            ibfv_vaults[subj["chimeric_id"]] = v

ga, ia, gacc, iacc = 0, 0, 0, 0
for test_subj in subjects:
    test_mins = test_subj["minutiae"].get(1, [])
    test_iris = test_subj["iris"].get(1)
    if not test_mins or test_iris is None:
        continue
    for vault_cid, vault in ibfv_vaults.items():
        success = unlock_vault_ibfv(test_mins, test_iris["code"], test_iris["mask"], vault)
        if vault_cid == test_subj["chimeric_id"]:
            ga += 1; gacc += int(success)
        else:
            ia += 1; iacc += int(success)
t_ibfv = time.time() - t0

far = iacc / ia * 100 if ia else 0
frr = (ga - gacc) / ga * 100 if ga else 0
print(f"IBFV f={factor} k={degree}:     FAR={far:.4f}% FRR={frr:.4f}% EER={(far+frr)/2:.4f}% | {t_ibfv:.1f}s | GA={ga} IA={ia}")

# Also test f=14, k=5 (fast config)
factor, degree = 14, 5
t0 = time.time()
vaults = {}
for subj in subjects:
    q = quantize_minutiae(subj["minutiae"].get(0, []), factor)
    if len(q) >= degree + 1:
        vaults[subj["chimeric_id"]] = create_vault(q, degree, seed=subj["chimeric_id"])

ga, ia, gacc, iacc = 0, 0, 0, 0
for test_subj in subjects:
    test_q = quantize_minutiae(test_subj["minutiae"].get(1, []), factor)
    for vault_cid, vault in vaults.items():
        success = unlock_vault(test_q, vault)
        if vault_cid == test_subj["chimeric_id"]:
            ga += 1; gacc += int(success)
        else:
            ia += 1; iacc += int(success)
t_f14 = time.time() - t0

far = iacc / ia * 100 if ia else 0
frr = (ga - gacc) / ga * 100 if ga else 0
print(f"Unimodal f=14 k=5:    FAR={far:.4f}% FRR={frr:.4f}% EER={(far+frr)/2:.4f}% | {t_f14:.1f}s")

print(f"\nTiming summary: unimodal(f=24,k=10)={t_uni:.1f}s, IBFV(f=24,k=10)={t_ibfv:.1f}s, unimodal(f=14,k=5)={t_f14:.1f}s")
print(f"Estimated full sweep (36 combos × 4 DBs × 1 setup): {36 * 4 * (t_uni + t_ibfv) / 60:.0f} min")
