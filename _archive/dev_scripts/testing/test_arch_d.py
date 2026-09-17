"""Quick diagnostic for Architecture D."""
import sys, json, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import load_chimeric_data
from architectures.architecture_d import lock_vault_d, unlock_vault_d

subjects = load_chimeric_data('fvc2002_1', min_iris=2)
print(f'Loaded {len(subjects)} subjects')

f, k = 24, 10

# --- GENUINE TESTS (all 100 subjects) ---
locked = 0
genuine_ok = 0
genuine_fail = 0
vaults = {}

for i, s in enumerate(subjects):
    mins0 = s['minutiae'].get(0, [])
    iris0 = s['iris'].get(0)
    mins1 = s['minutiae'].get(1, [])
    iris1 = s['iris'].get(1)

    if not mins0 or iris0 is None or not mins1 or iris1 is None:
        continue

    vault = lock_vault_d(mins0, iris0['code'], iris0['mask'], f, k, seed=42)
    if vault is None:
        continue
    locked += 1
    vaults[s['chimeric_id']] = vault

    ok = unlock_vault_d(mins1, iris1['code'], iris1['mask'], vault)
    if ok:
        genuine_ok += 1
    else:
        genuine_fail += 1

print(f'\n=== GENUINE RESULTS ({locked} subjects) ===')
print(f'Accept: {genuine_ok}, Reject: {genuine_fail}')
print(f'GAR: {genuine_ok/locked*100:.1f}%, FRR: {genuine_fail/locked*100:.1f}%')

# --- IMPOSTOR TESTS (first 20 subjects vs first 20 vaults) ---
print(f'\n=== IMPOSTOR TESTS (20×19=380 cross-attempts) ===')
imp_accept = 0
imp_total = 0
n_imp_test = min(20, len(subjects))

for i in range(n_imp_test):
    s = subjects[i]
    mins1 = s['minutiae'].get(1, [])
    iris1 = s['iris'].get(1)
    if not mins1 or iris1 is None:
        continue
    
    for j in range(n_imp_test):
        if i == j:
            continue
        vault_cid = subjects[j]['chimeric_id']
        if vault_cid not in vaults:
            continue
        imp_total += 1
        ok = unlock_vault_d(mins1, iris1['code'], iris1['mask'], vaults[vault_cid])
        if ok:
            imp_accept += 1

print(f'Impostor attempts: {imp_total}')
print(f'Impostor accepts: {imp_accept}')
print(f'FAR: {imp_accept/imp_total*100:.4f}%' if imp_total > 0 else 'N/A')
