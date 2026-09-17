import sys, time
from architectures.Architecture_A import evaluate_arch_a
from architectures.Architecture_B import evaluate_arch_b
from architectures.Architecture_C import evaluate_arch_c

print("\n--- FAST BRUTAL VALIDATION (PULLING 20 USERS FOR IMMEDIATE PROOF) ---")
start = time.time()

db_fvc = "/Users/suryanallaparaju/Desktop/Surya's Project/data/minutiae/2002/Db1_a"
iris_dir = "/Users/suryanallaparaju/Desktop/Surya's Project/data/Iris/Extracted_Codes"

factor = 18
degree = 5
mode = "1_to_lock_rest_to_unlock"
max_users = 20

print("[*] Evaluating Architecture A (Decision Level)...")
res_a = evaluate_arch_a(db_fvc, iris_dir, factor, degree, mode, max_users=max_users)
print(f"    -> Arch A | GA:{res_a['GA']} IA:{res_a['IA']} | FRR:{res_a['FRR']:.2f}% FAR:{res_a['FAR']:.2f}% EER:{res_a['EER']:.2f}%")

print("[*] Evaluating Architecture B (Feature Level/Entanglement)...")
res_b = evaluate_arch_b(db_fvc, iris_dir, factor, degree, mode, max_usimport sys, time
from architectures.Architecture_A import evaluate'Ifrom architectu_b['FRR']:.2f}% FAR:{res_b['FAR']:.2f}% EER:{res_b['EER']:.from architectures.Architecture_C import evaluate_arch_ph
print("\n--- FAST BRUTAL VALIDATION (PULLING 20 USERS dirstart = time.time()

db_fvc = "/Users/suryanallaparaju/Desktop/Surya'GA:{res_c['G
db_fvc = "/Users/']}iris_dir = "/Users/suryanallaparaju/Desktop/Surya's Project/data/Iris/Extracted_Cot(f"\n[*] Validation completed in {time.time() - start:.2f} seconds.")
