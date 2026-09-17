"""Ground truth test: what does the system produce RIGHT NOW?"""
import json, numpy as np, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from unimodal.fingerprint_vault import read_minutiae_file, quantize_minutiae, create_vault, unlock_vault
from architectures.architecture_ibfv import lock_vault_ibfv, unlock_vault_ibfv

# 1. What iris codes are active?
chimeric = json.load(open('chimeric_db/chimeric_fvc2002_1_min2.json'))
pair0 = chimeric['mapping'][0]['pairs'][0]
N_iris = np.load(pair0['iris_file'])['code'].size
print(f"=== CURRENT IRIS CODES: {N_iris} bits ===")
for Bs in [63, 127, 169, 255, 511]:
    print(f"  Bs={Bs:3d} -> {N_iris//Bs:4d} blocks")
print()

def load_db(db_name):
    with open(f'chimeric_db/chimeric_{db_name}_min2.json') as f:
        mapping = json.load(f)
    subjects = []
    for entry in mapping['mapping']:
        subj = {'cid': entry['chimeric_id'], 'mins': {}, 'iris': {}}
        for pair in entry['pairs']:
            imp = pair['impression'] - 1
            subj['mins'][imp] = read_minutiae_file(pair['fp_file'])
            idata = np.load(pair['iris_file'])
            subj['iris'][imp] = {'code': idata['code'], 'mask': idata['mask']}
        subjects.append(subj)
    return subjects

def eval_setup1(subjects, f_val, k_val, Bs):
    # Build unimodal vaults
    uni_v = {}
    for s in subjects:
        m0 = s['mins'].get(0, [])
        if not m0:
            continue
        try:
            q = quantize_minutiae(m0, f_val)
            v = create_vault(q, k_val, seed=s['cid'])
            if v:
                uni_v[s['cid']] = v
        except Exception:
            pass

    # Build IBFV vaults
    ibfv_v = {}
    for s in subjects:
        m0 = s['mins'].get(0, [])
        ir0 = s['iris'].get(0)
        if not m0 or ir0 is None:
            continue
        try:
            v = lock_vault_ibfv(m0, ir0['code'], ir0['mask'], f_val, k_val,
                                seed=s['cid'], block_size=Bs, n_bonus=4)
            if v:
                ibfv_v[s['cid']] = v
        except Exception:
            pass

    results = {}
    for lbl, vaults, use_iris in [('Uni', uni_v, False), ('IBFV', ibfv_v, True)]:
        ga = gacc = ia = iacc = 0
        for s in subjects:
            m1 = s['mins'].get(1, [])
            ir1 = s['iris'].get(1)
            if not m1:
                continue
            if use_iris and ir1 is None:
                continue
            for vc, v in vaults.items():
                if use_iris:
                    ok = unlock_vault_ibfv(m1, ir1['code'], ir1['mask'], v)
                else:
                    q1 = quantize_minutiae(m1, f_val)
                    ok = unlock_vault(q1, v)
                if vc == s['cid']:
                    ga += 1
                    gacc += int(ok)
                else:
                    ia += 1
                    iacc += int(ok)
        far = iacc / ia * 100 if ia else 0
        frr = (ga - gacc) / ga * 100 if ga else 0
        gar = gacc / ga * 100 if ga else 0
        eer = (far + frr) / 2
        results[lbl] = {'gar': gar, 'far': far, 'frr': frr, 'eer': eer, 'ga': ga, 'gacc': gacc, 'ia': ia, 'iacc': iacc}
    return results

# 2. Test on key databases
dbs = [
    ('fvc2002_1', 'DB1', 22, 7),
    ('fvc2002_2', 'DB2', 22, 7),
    ('fvc2002_3', 'DB3', 16, 5),
    ('fvc2004_1', 'DB4', 22, 5),
]

print(f"{'DB':>3s} {'f':>2s} {'k':>2s} {'Bs':>3s} | {'Uni GAR':>8s} {'Uni EER':>8s} | {'IBFV GAR':>9s} {'IBFV EER':>9s} | {'Delta':>7s} | time")
print("-" * 90)

for db_name, db_label, f_val, k_val in dbs:
    subjects = load_db(db_name)
    for Bs in [63, 169]:
        t0 = time.time()
        r = eval_setup1(subjects, f_val, k_val, Bs)
        dt = time.time() - t0
        u = r['Uni']
        i = r['IBFV']
        delta = u['eer'] - i['eer']
        print(f"{db_label:>3s} {f_val:2d} {k_val:2d} {Bs:3d} | {u['gar']:6.1f}% {u['eer']:6.3f}% | {i['gar']:7.1f}% {i['eer']:7.3f}% | {delta:+6.3f}% | {dt:.0f}s")
    print()

# 3. Also check: iris standalone GAR at each Bs
print("\n=== IRIS STANDALONE GAR ===")
from iris_extraction.iris_stabilizer import recover_iris_key

for db_name, db_label in [('fvc2002_1','DB1'), ('fvc2002_3','DB3'), ('fvc2004_1','DB4')]:
    subjects = load_db(db_name)
    for Bs in [63, 169]:
        ga = gacc = ia = iacc = 0
        for s in subjects:
            ir0 = s['iris'].get(0)
            ir1 = s['iris'].get(1)
            if ir0 is None or ir1 is None:
                continue
            # lock with impression 0
            key0 = recover_iris_key(ir0['code'], ir0['mask'], block_size=Bs)
            if key0 is None:
                continue
            # genuine: unlock with impression 1
            key1 = recover_iris_key(ir1['code'], ir1['mask'], block_size=Bs)
            if key1 is not None and key0 == key1:
                gacc += 1
            ga += 1
        gar = gacc / ga * 100 if ga else 0
        print(f"  {db_label} Bs={Bs:3d}: iris GAR = {gar:.1f}% ({gacc}/{ga})")
    print()
