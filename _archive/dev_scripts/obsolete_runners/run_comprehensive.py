"""
Comprehensive multimodal evaluation: all architectures, all parameters.

Produces publication-ready results for:
1. Architecture comparison at f=24, k=10 (best unimodal)
2. Parameter sweep f × k for each architecture
3. Block size / threshold sensitivity analysis
"""

import sys, csv, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_evaluation import load_chimeric_data
from unimodal.fingerprint_vault import (
    read_minutiae_file, quantize_minutiae, create_vault, unlock_vault,
)
from architectures.architecture_a import lock_vault_a, unlock_vault_a
from architectures.architecture_b import lock_vault_b, unlock_vault_b
from architectures.architecture_c import lock_vault_c, unlock_vault_c
from architectures.architecture_d import lock_vault_d, unlock_vault_d


def run_setup1_eval(subjects, lock_fn, unlock_fn, lock_kwargs=None, unlock_kwargs=None):
    """Run Setup 1 evaluation: lock imp 0, test imp 1, all subjects."""
    lock_kwargs = lock_kwargs or {}
    unlock_kwargs = unlock_kwargs or {}

    vaults = {}
    for s in subjects:
        cid = s['chimeric_id']
        mins0 = s['minutiae'].get(0, [])
        iris0 = s['iris'].get(0)
        if not mins0 or iris0 is None:
            continue
        vault = lock_fn(mins0, iris0['code'], iris0['mask'], **lock_kwargs)
        if vault is not None:
            vaults[cid] = vault

    ga = ia = gacc = iacc = gr = ir = 0
    for s in subjects:
        cid = s['chimeric_id']
        mins1 = s['minutiae'].get(1, [])
        iris1 = s['iris'].get(1)
        if not mins1 or iris1 is None:
            continue
        for vault_cid, vault in vaults.items():
            is_genuine = (vault_cid == cid)
            success = unlock_fn(mins1, iris1['code'], iris1['mask'], vault, **unlock_kwargs)
            if is_genuine:
                ga += 1
                if success:
                    gacc += 1
                else:
                    gr += 1
            else:
                ia += 1
                if success:
                    iacc += 1
                else:
                    ir += 1

    far = iacc / ia * 100 if ia > 0 else 0
    frr = gr / ga * 100 if ga > 0 else 0
    return {
        'ga': ga, 'ia': ia, 'gacc': gacc, 'iacc': iacc,
        'gr': gr, 'ir': ir,
        'far': round(far, 6), 'frr': round(frr, 6),
        'eer': round((far + frr) / 2, 6),
        'n_vaults': len(vaults),
    }


def run_unimodal(subjects, factor, degree):
    """Run unimodal fingerprint-only evaluation."""
    vaults = {}
    for s in subjects:
        cid = s['chimeric_id']
        mins0 = s['minutiae'].get(0, [])
        if not mins0:
            continue
        q = quantize_minutiae(mins0, factor)
        if len(q) < degree + 1:
            continue
        vault = create_vault(q, degree)
        if vault is not None:
            vaults[cid] = vault

    ga = ia = gacc = iacc = gr = ir = 0
    for s in subjects:
        cid = s['chimeric_id']
        mins1 = s['minutiae'].get(1, [])
        if not mins1:
            continue
        q = quantize_minutiae(mins1, factor)
        for vault_cid, vault in vaults.items():
            is_genuine = (vault_cid == cid)
            success = unlock_vault(q, vault)
            if is_genuine:
                ga += 1
                if success:
                    gacc += 1
                else:
                    gr += 1
            else:
                ia += 1
                if success:
                    iacc += 1
                else:
                    ir += 1

    far = iacc / ia * 100 if ia > 0 else 0
    frr = gr / ga * 100 if ga > 0 else 0
    return {
        'ga': ga, 'ia': ia, 'gacc': gacc, 'iacc': iacc,
        'gr': gr, 'ir': ir,
        'far': round(far, 6), 'frr': round(frr, 6),
        'eer': round((far + frr) / 2, 6),
        'n_vaults': len(vaults),
    }


def main():
    output_dir = Path(__file__).resolve().parent / 'results' / 'multimodal' / 'fvc2002_1'
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = load_chimeric_data('fvc2002_1', min_iris=2)
    print(f'Loaded {len(subjects)} chimeric subjects\n')

    all_results = []

    # ================================================================
    # 1. FULL PARAMETER SWEEP: f × k for each architecture
    # ================================================================
    factors = [16, 18, 20, 22, 24, 26, 28, 30]
    degrees = [5, 6, 7, 8, 9, 10, 11]

    # -- Unimodal baseline --
    print('=== UNIMODAL BASELINE ===')
    for f in factors:
        for k in degrees:
            t0 = time.time()
            r = run_unimodal(subjects, f, k)
            dt = time.time() - t0
            row = {'arch': 'Unimodal', 'factor': f, 'degree': k,
                   'param': '-', **r}
            all_results.append(row)
            if f == 24:  # Print key results
                print(f'  f={f} k={k}: FAR={r["far"]:.4f}% FRR={r["frr"]:.4f}% '
                      f'EER={r["eer"]:.4f}% ({dt:.1f}s)')

    # -- Architecture A: sweep τ at best (f,k) --
    print('\n=== ARCHITECTURE A (Decision AND) ===')
    thresholds = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46]
    for tau in thresholds:
        for f in [20, 24, 28]:
            for k in [7, 9, 10, 11]:
                t0 = time.time()
                r = run_setup1_eval(
                    subjects, lock_vault_a, unlock_vault_a,
                    lock_kwargs={'factor': f, 'degree': k, 'iris_threshold': tau},
                )
                dt = time.time() - t0
                row = {'arch': 'A', 'factor': f, 'degree': k,
                       'param': f'tau={tau}', **r}
                all_results.append(row)
                if f == 24 and k == 10:
                    print(f'  τ={tau} f={f} k={k}: FAR={r["far"]:.4f}% '
                          f'FRR={r["frr"]:.4f}% ({dt:.1f}s)')

    # -- Architecture B: sweep B at key (f,k) combos --
    print('\n=== ARCHITECTURE B (Polynomial Blinding) ===')
    for B in [255, 511, 1023]:
        for f in [20, 24, 28]:
            for k in [7, 9, 10, 11]:
                t0 = time.time()
                r = run_setup1_eval(
                    subjects, lock_vault_b, unlock_vault_b,
                    lock_kwargs={'factor': f, 'degree': k, 'block_size': B},
                )
                dt = time.time() - t0
                row = {'arch': 'B', 'factor': f, 'degree': k,
                       'param': f'B={B}', **r}
                all_results.append(row)
                if f == 24 and k == 10:
                    print(f'  B={B} f={f} k={k}: FAR={r["far"]:.4f}% '
                          f'FRR={r["frr"]:.4f}% ({dt:.1f}s)')

    # -- Architecture C: sweep B at key (f,k) combos --
    print('\n=== ARCHITECTURE C (Iris-Seeded Dense Chaff) ===')
    for B in [255, 1023]:
        for f, k in [(24, 10)]:  # Only best params (C is slow)
            t0 = time.time()
            r = run_setup1_eval(
                subjects, lock_vault_c, unlock_vault_c,
                lock_kwargs={'factor': f, 'degree': k, 'block_size': B},
            )
            dt = time.time() - t0
            row = {'arch': 'C', 'factor': f, 'degree': k,
                   'param': f'B={B}', **r}
            all_results.append(row)
            print(f'  B={B} f={f} k={k}: FAR={r["far"]:.4f}% '
                  f'FRR={r["frr"]:.4f}% ({dt:.1f}s)')

    # -- Architecture D: sweep B at key (f,k) combos --
    print('\n=== ARCHITECTURE D (AES Encrypt) ===')
    for B in [255, 511, 1023]:
        for f in [20, 24, 28]:
            for k in [7, 9, 10, 11]:
                t0 = time.time()
                r = run_setup1_eval(
                    subjects, lock_vault_d, unlock_vault_d,
                    lock_kwargs={'factor': f, 'degree': k, 'block_size': B},
                )
                dt = time.time() - t0
                row = {'arch': 'D', 'factor': f, 'degree': k,
                       'param': f'B={B}', **r}
                all_results.append(row)
                if f == 24 and k == 10:
                    print(f'  B={B} f={f} k={k}: FAR={r["far"]:.4f}% '
                          f'FRR={r["frr"]:.4f}% ({dt:.1f}s)')

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    csv_path = output_dir / 'comprehensive_results.csv'
    with open(csv_path, 'w', newline='') as fp:
        fields = ['arch', 'factor', 'degree', 'param', 'ga', 'ia',
                  'gacc', 'iacc', 'gr', 'ir', 'far', 'frr', 'eer', 'n_vaults']
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)

    json_path = output_dir / 'comprehensive_results.json'
    with open(json_path, 'w') as fp:
        json.dump(all_results, fp, indent=2)

    print(f'\n=== RESULTS SAVED ===')
    print(f'  CSV: {csv_path}')
    print(f'  JSON: {json_path}')
    print(f'  Total experiments: {len(all_results)}')

    # Print summary table
    print('\n=== SUMMARY: f=24, k=10 ===')
    print(f'{"Architecture":<25} {"FAR":>8} {"FRR":>8} {"EER":>8}')
    print('-' * 55)
    for r in all_results:
        if r['factor'] == 24 and r['degree'] == 10:
            label = f'{r["arch"]} ({r["param"]})'
            print(f'{label:<25} {r["far"]:>7.4f}% {r["frr"]:>7.4f}% {r["eer"]:>7.4f}%')


if __name__ == '__main__':
    main()
