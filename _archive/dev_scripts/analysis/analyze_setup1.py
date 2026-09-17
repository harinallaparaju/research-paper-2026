"""Analyze Setup 1 results."""
import csv

rows = []
with open("results/complete_experiments/setup1_results_20260405_134407.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["factor"] = int(r["factor"])
        r["degree"] = int(r["degree"])
        r["eer_pct"] = float(r["eer_pct"])
        r["far_pct"] = float(r["far_pct"])
        r["frr_pct"] = float(r["frr_pct"])
        rows.append(r)

print("=" * 80)
print("SETUP 1 COMPLETE RESULTS - BEST EER PER DATABASE")
print("=" * 80)

dbs = ["fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"]
for db in dbs:
    db_rows = [r for r in rows if r["database"] == db]
    uni = [r for r in db_rows if r["architecture"] == "Unimodal"]
    arch_a = [r for r in db_rows if r["architecture"] == "A"]
    ibfv = [r for r in db_rows if r["architecture"] == "IBFV"]

    best_uni = min(uni, key=lambda r: r["eer_pct"])
    best_a = min(arch_a, key=lambda r: r["eer_pct"])
    best_ibfv = min(ibfv, key=lambda r: r["eer_pct"])

    print(f"\n{db}:")
    print(f"  Best Unimodal: EER={best_uni['eer_pct']:.4f}% (f={best_uni['factor']},k={best_uni['degree']})")
    print(f"  Best ArchA:    EER={best_a['eer_pct']:.4f}% (f={best_a['factor']},k={best_a['degree']})")
    print(f"  Best IBFV:     EER={best_ibfv['eer_pct']:.4f}% (f={best_ibfv['factor']},k={best_ibfv['degree']})")

# Count IBFV improvements
print("\n" + "=" * 80)
print("IBFV vs UNIMODAL: IMPROVEMENT COUNT")
print("=" * 80)
better = equal = worse = 0
max_improvement = 0
max_imp_cfg = ""

for db in dbs:
    for f in [14, 16, 18, 20, 22, 24, 26, 28, 30]:
        for k in [5, 7, 9, 11]:
            uni_r = [r for r in rows if r["database"] == db and r["architecture"] == "Unimodal"
                     and r["factor"] == f and r["degree"] == k]
            ibfv_r = [r for r in rows if r["database"] == db and r["architecture"] == "IBFV"
                      and r["factor"] == f and r["degree"] == k]
            if uni_r and ibfv_r:
                delta = ibfv_r[0]["eer_pct"] - uni_r[0]["eer_pct"]
                if delta < -0.001:
                    better += 1
                    if delta < max_improvement:
                        max_improvement = delta
                        max_imp_cfg = f"{db} f={f} k={k}"
                elif delta > 0.001:
                    worse += 1
                else:
                    equal += 1

total = better + equal + worse
print(f"  IBFV < Unimodal (BETTER): {better} / {total}")
print(f"  IBFV = Unimodal (EQUAL):  {equal} / {total}")
print(f"  IBFV > Unimodal (WORSE):  {worse} / {total}")
print(f"  Max improvement: {max_improvement:+.4f}% at {max_imp_cfg}")

# Same-params comparison at best unimodal point
print("\n" + "=" * 80)
print("AT BEST UNIMODAL OPERATING POINT (per DB)")
print("=" * 80)
print(f"{'Database':<14} {'f':>3} {'k':>3}  {'Unimodal':>10} {'ArchA':>10} {'IBFV':>10} {'delta':>8}")
print("-" * 62)
for db in dbs:
    uni = [r for r in rows if r["database"] == db and r["architecture"] == "Unimodal"]
    best_uni = min(uni, key=lambda r: r["eer_pct"])
    f, k = best_uni["factor"], best_uni["degree"]
    ibfv_match = [r for r in rows if r["database"] == db and r["architecture"] == "IBFV"
                  and r["factor"] == f and r["degree"] == k]
    a_match = [r for r in rows if r["database"] == db and r["architecture"] == "A"
               and r["factor"] == f and r["degree"] == k]

    u = best_uni["eer_pct"]
    i = ibfv_match[0]["eer_pct"] if ibfv_match else -1
    a = a_match[0]["eer_pct"] if a_match else -1
    d = i - u
    print(f"{db:<14} {f:3d} {k:3d}  {u:10.4f} {a:10.4f} {i:10.4f} {d:+8.4f}")

# Show biggest improvements on hard databases
print("\n" + "=" * 80)
print("TOP 10 IBFV IMPROVEMENTS (by absolute EER reduction)")
print("=" * 80)
deltas = []
for db in dbs:
    for f in [14, 16, 18, 20, 22, 24, 26, 28, 30]:
        for k in [5, 7, 9, 11]:
            uni_r = [r for r in rows if r["database"] == db and r["architecture"] == "Unimodal"
                     and r["factor"] == f and r["degree"] == k]
            ibfv_r = [r for r in rows if r["database"] == db and r["architecture"] == "IBFV"
                      and r["factor"] == f and r["degree"] == k]
            if uni_r and ibfv_r:
                u = uni_r[0]["eer_pct"]
                i = ibfv_r[0]["eer_pct"]
                deltas.append((i - u, db, f, k, u, i))

deltas.sort()
print(f"{'Database':<14} {'f':>3} {'k':>3}  {'Uni EER':>10} {'IBFV EER':>10} {'delta':>10}")
print("-" * 56)
for d, db, f, k, u, i in deltas[:10]:
    print(f"{db:<14} {f:3d} {k:3d}  {u:10.4f} {i:10.4f} {d:+10.4f}")

# Verify: is IBFV ever worse?
print("\n" + "=" * 80)
print("VERIFICATION: Any case where IBFV FAR > Unimodal FAR?")
print("=" * 80)
far_worse = 0
for db in dbs:
    for f in [14, 16, 18, 20, 22, 24, 26, 28, 30]:
        for k in [5, 7, 9, 11]:
            uni_r = [r for r in rows if r["database"] == db and r["architecture"] == "Unimodal"
                     and r["factor"] == f and r["degree"] == k]
            ibfv_r = [r for r in rows if r["database"] == db and r["architecture"] == "IBFV"
                      and r["factor"] == f and r["degree"] == k]
            if uni_r and ibfv_r:
                u_far = uni_r[0]["far_pct"]
                i_far = ibfv_r[0]["far_pct"]
                if i_far > u_far + 0.001:
                    far_worse += 1
                    print(f"  {db} f={f} k={k}: Uni FAR={u_far:.4f}% IBFV FAR={i_far:.4f}%")
if far_worse == 0:
    print("  NONE! IBFV FAR <= Unimodal FAR for all 144 configurations.")
