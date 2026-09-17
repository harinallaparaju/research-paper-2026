"""Quick test for Architecture B and C."""
import sys, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_evaluation import load_chimeric_data
from architectures.architecture_b import lock_vault_b, unlock_vault_b
from architectures.architecture_c import lock_vault_c, unlock_vault_c

subjects = load_chimeric_data('fvc2002_1', min_iris=2)
print(f'Loaded {len(subjects)} subjects')
f, k = 24, 10
n_test = 10

for arch_name, lock_fn, unlock_fn in [
    ("B (Poly Blinding)", lock_vault_b, unlock_vault_b),
    ("C (Iris-Seeded Chaff)", lock_vault_c, unlock_vault_c),
]:
    print(f'\n=== Architecture {arch_name} ===')
    locked = 0
    gen_ok = gen_fail = 0
    imp_ok = imp_total = 0
    vaults = {}

    # Lock and genuine test
    for i in range(n_test):
        s = subjects[i]
        mins0 = s['minutiae'].get(0, [])
        iris0 = s['iris'].get(0)
        mins1 = s['minutiae'].get(1, [])
        iris1 = s['iris'].get(1)
        if not mins0 or iris0 is None or not mins1 or iris1 is None:
            continue

        t0 = time.time()
        vault = lock_fn(mins0, iris0['code'], iris0['mask'], f, k, seed=42)
        lock_t = time.time() - t0
        if vault is None:
            print(f'  Subject {s["chimeric_id"]}: lock failed')
            continue
        locked += 1
        vaults[i] = vault

        t0 = time.time()
        ok = unlock_fn(mins1, iris1['code'], iris1['mask'], vault)
        unlock_t = time.time() - t0
        if ok:
            gen_ok += 1
        else:
            gen_fail += 1
        status = "ACCEPT" if ok else "REJECT"
        print(f'  Subject {s["chimeric_id"]}: GENUINE {status} (lock {lock_t:.1f}s, unlock {unlock_t:.1f}s)')

    # Impostor test (first 5 × first 5)
    for i in range(min(5, n_test)):
        s = subjects[i]
        mins1 = s['minutiae'].get(1, [])
        iris1 = s['iris'].get(1)
        if not mins1 or iris1 is None:
            continue
        for j in range(min(5, n_test)):
            if i == j or j not in vaults:
                continue
            imp_total += 1
            ok = unlock_fn(mins1, iris1['code'], iris1['mask'], vaults[j])
            if ok:
                imp_ok += 1

    print(f'\n  Genuine: {gen_ok}/{locked} accept ({gen_ok/locked*100:.0f}%)')
    print(f'  Impostor: {imp_ok}/{imp_total} accept (FAR={imp_ok/imp_total*100:.2f}%)' if imp_total > 0 else '  No impostor tests')
