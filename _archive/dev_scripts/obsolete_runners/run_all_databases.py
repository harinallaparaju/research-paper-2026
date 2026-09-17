"""
Critical analysis: compute exactly what multimodal gives us on ALL databases.
Runs fast unimodal + Architecture A evaluation on each FVC database.
This tells us WHERE multimodal helps and WHERE it doesn't.
"""
import json, time, csv, sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_evaluation import load_chimeric_data
from unimodal.fingerprint_vault import quantize_minutiae, create_vault, unlock_vault
from architectures.architecture_a import lock_vault_a, unlock_vault_a
from iris_extraction.matching import fractional_hamming_distance


def run_unimodal_eval(subjects, factor, degree):
    """Quick unimodal eval: Setup 1."""
    vaults = {}
    for s in subjects:
        cid = s['chimeric_id']
        mins = s['minutiae'].get(0, [])
        if not mins:
            continue
        q = quantize_minutiae(mins, factor)
        if len(q) < degree + 1:
            continue
        vault = create_vault(q, degree)
        if vault is not None:
            vaults[cid] = vault

    ga = ia = gacc = iacc = gr = ir = 0
    for s in subjects:
        cid = s['chimeric_id']
        mins = s['minutiae'].get(1, [])
        if not mins:
            continue
        q = quantize_minutiae(mins, factor)
        for vid, v in vaults.items():
            is_gen = (vid == cid)
            ok = unlock_vault(q, v)
            if is_gen:
                ga += 1
                if ok: gacc += 1
                else: gr += 1
            else:
                ia += 1
                if ok: iacc += 1
                else: ir += 1

    far = iacc / ia * 100 if ia > 0 else 0
    frr = gr / ga * 100 if ga > 0 else 0
    return {'ga': ga, 'ia': ia, 'gacc': gacc, 'iacc': iacc, 'gr': gr, 'ir': ir,
            'far': round(far, 6), 'frr': round(frr, 6), 'eer': round((far+frr)/2, 6),
            'n_vaults': len(vaults)}


def run_arch_a_eval(subjects, factor, degree, tau):
    """Quick Architecture A eval: Setup 1."""
    vaults = {}
    for s in subjects:
        cid = s['chimeric_id']
        mins = s['minutiae'].get(0, [])
        iris = s['iris'].get(0)
        if not mins or iris is None:
            continue
        v = lock_vault_a(mins, iris['code'], iris['mask'], factor, degree, tau)
        if v is not None:
            vaults[cid] = v

    ga = ia = gacc = iacc = gr = ir = 0
    for s in subjects:
        cid = s['chimeric_id']
        mins = s['minutiae'].get(1, [])
        iris = s['iris'].get(1)
        if not mins or iris is None:
            continue
        for vid, v in vaults.items():
            is_gen = (vid == cid)
            ok = unlock_vault_a(mins, iris['code'], iris['mask'], v)
            if is_gen:
                ga += 1
                if ok: gacc += 1
                else: gr += 1
            else:
                ia += 1
                if ok: iacc += 1
                else: ir += 1

    far = iacc / ia * 100 if ia > 0 else 0
    frr = gr / ga * 100 if ga > 0 else 0
    return {'ga': ga, 'ia': ia, 'gacc': gacc, 'iacc': iacc, 'gr': gr, 'ir': ir,
            'far': round(far, 6), 'frr': round(frr, 6), 'eer': round((far+frr)/2, 6),
            'n_vaults': len(vaults)}


def main():
    databases = ['fvc2002_1', 'fvc2002_2', 'fvc2002_3', 'fvc2004_1']

    # Senior's best parameters per DB (from published paper)
    # Plus our extended range to find actual best
    param_sets = {
        'fvc2002_1': [(14, 5), (16, 5), (18, 5), (18, 7), (20, 7), (24, 10)],
        'fvc2002_2': [(10, 5), (12, 5), (14, 5), (16, 5), (18, 5)],
        'fvc2002_3': [(10, 5), (12, 5), (14, 5), (16, 5), (18, 5)],
        'fvc2004_1': [(10, 5), (12, 5), (14, 5), (16, 5), (18, 5)],
    }

    thresholds = [0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46]

    all_results = []

    for db in databases:
        print(f"\n{'='*70}")
        print(f"DATABASE: {db}")
        print(f"{'='*70}")

        try:
            subjects = load_chimeric_data(db, min_iris=2)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        print(f"  Loaded {len(subjects)} chimeric subjects")

        # --- Unimodal sweep ---
        print(f"\n  --- UNIMODAL ---")
        best_uni_eer = 999
        best_uni = None
        for f, k in param_sets[db]:
            t0 = time.time()
            r = run_unimodal_eval(subjects, f, k)
            dt = time.time() - t0
            print(f"  f={f:2d} k={k:2d}: FAR={r['far']:8.4f}% FRR={r['frr']:8.4f}% "
                  f"EER={r['eer']:8.4f}% gacc={r['gacc']} iacc={r['iacc']} ({dt:.1f}s)")
            row = {'db': db, 'arch': 'Unimodal', 'factor': f, 'degree': k,
                   'param': '-', **r}
            all_results.append(row)
            if r['eer'] < best_uni_eer:
                best_uni_eer = r['eer']
                best_uni = (f, k, r)

        if best_uni:
            bf, bk, br = best_uni
            print(f"  BEST UNIMODAL: f={bf} k={bk} → FAR={br['far']:.4f}% FRR={br['frr']:.4f}% EER={br['eer']:.4f}%")

        # --- Architecture A sweep at best unimodal params ---
        if best_uni:
            bf, bk, _ = best_uni
            print(f"\n  --- ARCHITECTURE A (f={bf}, k={bk}) ---")
            for tau in thresholds:
                t0 = time.time()
                r = run_arch_a_eval(subjects, bf, bk, tau)
                dt = time.time() - t0
                print(f"  τ={tau:.2f}: FAR={r['far']:8.4f}% FRR={r['frr']:8.4f}% "
                      f"EER={r['eer']:8.4f}% gacc={r['gacc']} iacc={r['iacc']} ({dt:.1f}s)")
                row = {'db': db, 'arch': 'A', 'factor': bf, 'degree': bk,
                       'param': f'tau={tau}', **r}
                all_results.append(row)

        # Also run Architecture A at wider f range for the best threshold
        print(f"\n  --- ARCHITECTURE A (τ=0.42, sweep f,k) ---")
        for f, k in param_sets[db]:
            t0 = time.time()
            r = run_arch_a_eval(subjects, f, k, 0.42)
            dt = time.time() - t0
            print(f"  f={f:2d} k={k:2d} τ=0.42: FAR={r['far']:8.4f}% FRR={r['frr']:8.4f}% "
                  f"gacc={r['gacc']} iacc={r['iacc']} ({dt:.1f}s)")
            row = {'db': db, 'arch': 'A', 'factor': f, 'degree': k,
                   'param': 'tau=0.42', **r}
            all_results.append(row)

    # Save all results
    out_dir = Path(__file__).resolve().parent / 'results' / 'multimodal' / 'all_databases'
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / 'multi_db_analysis.csv'
    with open(csv_path, 'w', newline='') as f:
        fields = ['db', 'arch', 'factor', 'degree', 'param',
                  'ga', 'ia', 'gacc', 'iacc', 'gr', 'ir', 'far', 'frr', 'eer', 'n_vaults']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)

    print(f"\n\nResults saved to: {csv_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: BEST UNIMODAL vs BEST ARCH A per database")
    print(f"{'='*80}")
    print(f"{'Database':<15} {'System':<20} {'FAR':>8} {'FRR':>8} {'EER':>8}")
    print('-' * 65)
    for db in databases:
        db_results = [r for r in all_results if r['db'] == db]
        # Best unimodal by EER
        uni = [r for r in db_results if r['arch'] == 'Unimodal']
        if uni:
            best_u = min(uni, key=lambda x: x['eer'])
            print(f"{db:<15} {'Unimodal':<20} {best_u['far']:>7.4f}% {best_u['frr']:>7.4f}% {best_u['eer']:>7.4f}%")
        # Best Arch A: lowest FAR with FRR < 20%
        arch_a = [r for r in db_results if r['arch'] == 'A' and r['frr'] < 20]
        if arch_a:
            best_a_far = min(arch_a, key=lambda x: x['far'])
            p = best_a_far['param']
            print(f"{'':<15} {f'Arch A ({p})':<20} {best_a_far['far']:>7.4f}% {best_a_far['frr']:>7.4f}% {best_a_far['eer']:>7.4f}%")
        # Best Arch A by EER
        if arch_a:
            best_a_eer = min(arch_a, key=lambda x: x['eer'])
            p = best_a_eer['param']
            print(f"{'':<15} {f'Arch A EER ({p})':<20} {best_a_eer['far']:>7.4f}% {best_a_eer['frr']:>7.4f}% {best_a_eer['eer']:>7.4f}%")
        print()


if __name__ == '__main__':
    main()
