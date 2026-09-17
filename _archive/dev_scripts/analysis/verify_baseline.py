"""
Verify our unimodal results match the senior's published JISAA 2025 paper.

Senior's published best results (Setup 1):
  FVC2002 DB1: EER = 0.0051%  (f=24, k=10)
  FVC2002 DB2: EER = 0.25%    (estimated from paper)  
  FVC2002 DB3: EER = 2.85%    (estimated from paper)
  FVC2004 DB1: EER = 3.51%    (estimated from paper)

Note: Senior used factors 8-34 step 2 and degrees 5-11 step 1.
Our sweep uses factors 14-30 step 2 and degrees 5,7,9,11.
"""
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
        r["ga"] = int(r["ga"])
        r["ia"] = int(r["ia"])
        r["gacc"] = int(r["gacc"])
        r["iacc"] = int(r["iacc"])
        rows.append(r)

print("=" * 80)
print("VERIFICATION: Our Results vs Senior's Published Paper (JISAA 2025)")
print("=" * 80)

# Senior's published results (Setup 1, best EER for each DB)
senior = {
    "fvc2002_1": {"eer": 0.0051, "note": "f=24,k=10 from paper"},
    "fvc2002_2": {"eer": 0.25, "note": "estimated from paper figures"},
    "fvc2002_3": {"eer": 2.85, "note": "estimated from paper figures"},
    "fvc2004_1": {"eer": 3.51, "note": "estimated from paper figures"},
}

dbs = ["fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"]

for db in dbs:
    uni = [r for r in rows if r["database"] == db and r["architecture"] == "Unimodal"]
    best = min(uni, key=lambda r: r["eer_pct"])
    pub = senior[db]
    
    print(f"\n{db}:")
    print(f"  Senior published: EER = {pub['eer']:.4f}%  ({pub['note']})")
    print(f"  Our best:         EER = {best['eer_pct']:.4f}%  (f={best['factor']},k={best['degree']})")
    print(f"  GA={best['ga']} IA={best['ia']} GAcc={best['gacc']} IAcc={best['iacc']}")
    print(f"  FAR={best['far_pct']:.4f}% FRR={best['frr_pct']:.4f}%")
    
    # Also show f=24, k=9 or k=11 (closest to senior's k=10)
    for k in [9, 11]:
        match = [r for r in uni if r["factor"] == 24 and r["degree"] == k]
        if match:
            m = match[0]
            print(f"  At f=24,k={k}: EER={m['eer_pct']:.4f}% FAR={m['far_pct']:.4f}% FRR={m['frr_pct']:.4f}%")

# Show senior's f=24 comparison
print("\n" + "=" * 80)
print("f=24 ACROSS ALL DEGREES (Senior's primary factor)")
print("=" * 80)
print(f"{'Database':<14} {'k':>3}  {'Unimodal':>10} {'ArchA':>10} {'IBFV':>10} {'delta':>8}")
print("-" * 60)
for db in dbs:
    for k in [5, 7, 9, 11]:
        u = [r for r in rows if r["database"] == db and r["architecture"] == "Unimodal"
             and r["factor"] == 24 and r["degree"] == k]
        a = [r for r in rows if r["database"] == db and r["architecture"] == "A"
             and r["factor"] == 24 and r["degree"] == k]
        i = [r for r in rows if r["database"] == db and r["architecture"] == "IBFV"
             and r["factor"] == 24 and r["degree"] == k]
        if u:
            ue = u[0]["eer_pct"]
            ae = a[0]["eer_pct"] if a else -1
            ie = i[0]["eer_pct"] if i else -1
            d = ie - ue
            print(f"{db:<14} {k:3d}  {ue:10.4f} {ae:10.4f} {ie:10.4f} {d:+8.4f}")

# Comprehensive table for paper
print("\n" + "=" * 80)
print("COMPLETE SETUP 1 TABLE (FOR PAPER)")
print("=" * 80)
print(f"{'DB':<12} {'f':>3} {'k':>3}  {'Uni_EER':>8} {'A_EER':>8} {'IBFV_EER':>9} {'delta':>8} {'Uni_FAR':>8} {'Uni_FRR':>8} {'IBFV_FAR':>9} {'IBFV_FRR':>9}")
print("-" * 105)
for db in dbs:
    for f in [14, 18, 22, 24, 28]:  # Key factors
        for k in [5, 7, 9, 11]:
            u = [r for r in rows if r["database"] == db and r["architecture"] == "Unimodal"
                 and r["factor"] == f and r["degree"] == k]
            a = [r for r in rows if r["database"] == db and r["architecture"] == "A"
                 and r["factor"] == f and r["degree"] == k]
            iv = [r for r in rows if r["database"] == db and r["architecture"] == "IBFV"
                  and r["factor"] == f and r["degree"] == k]
            if u and iv:
                ue = u[0]["eer_pct"]
                ae = a[0]["eer_pct"] if a else 0
                ie = iv[0]["eer_pct"]
                d = ie - ue
                uf = u[0]["far_pct"]
                ur = u[0]["frr_pct"]
                ivf = iv[0]["far_pct"]
                ivr = iv[0]["frr_pct"]
                marker = " *" if d < -0.01 else ""
                print(f"{db:<12} {f:3d} {k:3d}  {ue:8.4f} {ae:8.4f} {ie:9.4f} {d:+8.4f}{marker}")
