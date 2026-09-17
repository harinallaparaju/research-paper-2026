"""Analyze Setup 1 and Setup 2 results."""
import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "complete_experiments")

def analyze_setup(csv_path, setup_label):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    
    print(f"\n{'='*80}")
    print(f"{setup_label} — {len(rows)} results")
    print(f"{'='*80}")
    
    dbs = sorted(set(r["database"] for r in rows))
    
    for db in dbs:
        uni = [r for r in rows if r["database"] == db and r["architecture"] == "Unimodal"]
        ibfv = [r for r in rows if r["database"] == db and r["architecture"] == "IBFV"]
        archa = [r for r in rows if r["database"] == db and r["architecture"] == "A"]
        
        if not uni or not ibfv:
            continue
        
        bu = min(uni, key=lambda r: float(r["eer_pct"]))
        bi = min(ibfv, key=lambda r: float(r["eer_pct"]))
        ba = min(archa, key=lambda r: float(r["eer_pct"])) if archa else None
        
        delta = float(bi["eer_pct"]) - float(bu["eer_pct"])
        print(f"\n{db}:")
        print(f"  Unimodal BEST: EER={float(bu['eer_pct']):8.4f}%  (f={bu['factor']}, k={bu['degree']})")
        if ba:
            print(f"  Arch A   BEST: EER={float(ba['eer_pct']):8.4f}%  (f={ba['factor']}, k={ba['degree']})")
        print(f"  IBFV     BEST: EER={float(bi['eer_pct']):8.4f}%  (f={bi['factor']}, k={bi['degree']})  ΔEER={delta:+.4f}%")
    
    # Check violations
    violations = 0
    improvements = 0
    total = 0
    for db in dbs:
        ibfv_rows = [r for r in rows if r["database"] == db and r["architecture"] == "IBFV"]
        for ri in ibfv_rows:
            ru_list = [r for r in rows if r["database"] == db 
                       and r["architecture"] == "Unimodal"
                       and r["factor"] == ri["factor"] and r["degree"] == ri["degree"]]
            if not ru_list:
                continue
            ru = ru_list[0]
            total += 1
            d = float(ri["eer_pct"]) - float(ru["eer_pct"])
            if d > 0.001:
                violations += 1
                print(f"  VIOLATION: {db} f={ri['factor']} k={ri['degree']}: "
                      f"Uni={ru['eer_pct']} IBFV={ri['eer_pct']}")
            elif d < -0.001:
                improvements += 1
    
    ties = total - violations - improvements
    print(f"\nIBFV ≤ Unimodal violations: {violations}/{total}")
    print(f"IBFV < Unimodal improvements: {improvements}/{total}")
    print(f"IBFV = Unimodal ties: {ties}/{total}")
    
    return rows


def compare_senior(rows):
    """Compare with senior's published Setup 1 best results."""
    senior = {
        "fvc2002_1": 0.0051,
        "fvc2002_2": 0.25,
        "fvc2002_3": 2.85,
        "fvc2004_1": 3.51,
    }
    print(f"\n{'='*80}")
    print("COMPARISON WITH SENIOR'S PUBLISHED RESULTS (Setup 1)")
    print(f"{'='*80}")
    
    for db, pub_eer in senior.items():
        uni = [r for r in rows if r["database"] == db 
               and r["architecture"] == "Unimodal" and r["setup"] == "1"]
        if not uni:
            continue
        bu = min(uni, key=lambda r: float(r["eer_pct"]))
        our = float(bu["eer_pct"])
        match = "MATCH" if abs(our - pub_eer) < 0.1 else "DIFFERS"
        print(f"  {db}: Published={pub_eer:.4f}%  Ours={our:.4f}%  "
              f"(f={bu['factor']},k={bu['degree']})  [{match}]")


def detailed_table(rows, setup_num):
    """Print detailed comparison table for key params."""
    print(f"\n{'='*80}")
    print(f"DETAILED TABLE — Setup {setup_num}")
    print(f"{'='*80}")
    print(f"{'DB':<12} {'f':>3} {'k':>3}  {'Unimodal':>10} {'ArchA':>10} {'IBFV':>10} {'ΔEER':>8}")
    print("-" * 65)
    
    dbs = sorted(set(r["database"] for r in rows))
    for db in dbs:
        for f in [14, 18, 20, 24, 28]:
            for k in [5, 7, 9, 11]:
                uni_r = [r for r in rows if r["database"] == db 
                         and r["architecture"] == "Unimodal"
                         and r["factor"] == str(f) and r["degree"] == str(k)]
                a_r = [r for r in rows if r["database"] == db 
                       and r["architecture"] == "A"
                       and r["factor"] == str(f) and r["degree"] == str(k)]
                i_r = [r for r in rows if r["database"] == db 
                       and r["architecture"] == "IBFV"
                       and r["factor"] == str(f) and r["degree"] == str(k)]
                
                if not uni_r or not i_r:
                    continue
                
                u_eer = float(uni_r[0]["eer_pct"])
                a_eer = float(a_r[0]["eer_pct"]) if a_r else float("nan")
                i_eer = float(i_r[0]["eer_pct"])
                delta = i_eer - u_eer
                marker = " *" if delta < -0.01 else ""
                
                print(f"{db:<12} {f:3d} {k:3d}  {u_eer:10.4f} {a_eer:10.4f} "
                      f"{i_eer:10.4f} {delta:+8.4f}{marker}")


if __name__ == "__main__":
    # Setup 1
    s1_files = sorted(f for f in os.listdir(RESULTS_DIR) if f.startswith("setup1_results"))
    if s1_files:
        s1_path = os.path.join(RESULTS_DIR, s1_files[-1])
        rows1 = analyze_setup(s1_path, "SETUP 1 (Lock=imp1, Test=imp2)")
        compare_senior(rows1)
        detailed_table(rows1, 1)
    
    # Setup 2
    s2_files = sorted(f for f in os.listdir(RESULTS_DIR) if f.startswith("setup2_results"))
    if s2_files:
        s2_path = os.path.join(RESULTS_DIR, s2_files[-1])
        rows2 = analyze_setup(s2_path, "SETUP 2 (Lock=imp1, Test=all impressions) [IN PROGRESS]")
        detailed_table(rows2, 2)
