"""
Unimodal Fingerprint Fuzzy Vault — Baseline Replication.

Replicates Tables 1 and 5-9 from the JISAA 2025 paper.

Setup 1: Lock=impression 1, Unlock=impression 2
    - GA = N genuine attempts (1 per user)
    - IA = N*(N-1) impostor attempts
Setup 2: Lock=impression 1, Unlock=impressions 1..8
    - GA = N*8 genuine attempts
    - IA = N*8*(N-1) impostor attempts

Parameters: f ∈ {16,18,...,30}, k ∈ {5,6,...,11}
"""

import sys
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file,
    quantize_minutiae,
    create_vault,
    unlock_vault,
)


# Default parameter ranges from the paper
DEFAULT_FACTORS = list(range(16, 31, 2))          # f ∈ {16, 18, ..., 30}
DEFAULT_DEGREES = list(range(5, 12))               # k ∈ {5, 6, ..., 11}

# Data paths
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "minutiae"


def load_database(db_name: str) -> Dict[int, List[Path]]:
    """
    Load fingerprint minutiae file paths grouped by subject.

    Args:
        db_name: One of "fvc2002_1", "fvc2002_2", "fvc2002_3", "fvc2004_1"

    Returns:
        {subject_id: [path_imp1, path_imp2, ..., path_imp8]}
    """
    db_map = {
        "fvc2002_1": DATA_ROOT / "2002" / "Db1_a",
        "fvc2002_2": DATA_ROOT / "2002" / "Db2_a",
        "fvc2002_3": DATA_ROOT / "2002" / "Db3_a",
        "fvc2004_1": DATA_ROOT / "2004" / "Db1_a",
    }

    db_dir = db_map[db_name]
    subjects = {}

    for f in sorted(db_dir.glob("*.txt")):
        parts = f.stem.split("_")
        subj_id = int(parts[0])
        if subj_id not in subjects:
            subjects[subj_id] = []
        subjects[subj_id].append(f)

    # Sort by impression number
    for sid in subjects:
        subjects[sid].sort(
            key=lambda p: int(p.stem.split("_")[1].replace(".jpg", ""))
        )

    return subjects


def run_experiment(
    db_name: str = "fvc2002_1",
    setup: int = 1,
    factors: List[int] = None,
    degrees: List[int] = None,
    output_dir: Path = None,
):
    """
    Run the full unimodal fuzzy vault experiment.

    Args:
        db_name: Database identifier
        setup: 1 or 2
        factors: List of quantization factors
        degrees: List of polynomial degrees
        output_dir: Where to save results
    """
    if factors is None:
        factors = DEFAULT_FACTORS
    if degrees is None:
        degrees = DEFAULT_DEGREES
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "results" / "unimodal" / db_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load database
    subjects = load_database(db_name)
    subject_ids = sorted(subjects.keys())
    n_subjects = len(subject_ids)

    print(f"Database: {db_name}")
    print(f"Setup: {setup}")
    print(f"Subjects: {n_subjects}")
    print(f"Factors: {factors}")
    print(f"Degrees: {degrees}")

    # Determine which impressions to use
    if setup == 1:
        # Lock=imp1, Test=imp2 only
        lock_impression = 0   # index into subjects[sid]
        test_impressions = [1]
    else:
        # Lock=imp1, Test=all impressions
        lock_impression = 0
        test_impressions = list(range(len(next(iter(subjects.values())))))

    # Pre-load all minutiae
    print("Loading minutiae...")
    all_minutiae = {}
    for sid in subject_ids:
        all_minutiae[sid] = {}
        for imp_idx, fpath in enumerate(subjects[sid]):
            all_minutiae[sid][imp_idx] = read_minutiae_file(str(fpath))

    all_results = []
    total_combos = len(factors) * len(degrees)
    combo_idx = 0

    for factor in factors:
        for degree in degrees:
            combo_idx += 1
            t0 = time.time()

            # Quantize all minutiae for this factor
            quantized = {}
            for sid in subject_ids:
                quantized[sid] = {}
                for imp_idx in all_minutiae[sid]:
                    quantized[sid][imp_idx] = quantize_minutiae(
                        all_minutiae[sid][imp_idx], factor
                    )

            # Create vaults (lock impression for each subject)
            vaults = {}
            for sid in subject_ids:
                lock_mins = quantized[sid][lock_impression]
                if len(lock_mins) < degree + 1:
                    continue  # Not enough minutiae for this degree
                vaults[sid] = create_vault(lock_mins, degree)

            # Run unlock attempts
            ga, ia = 0, 0     # Total attempts
            gacc, iacc = 0, 0  # Accepts
            gr, ir = 0, 0      # Rejects

            for test_sid in subject_ids:
                for test_imp in test_impressions:
                    if test_imp >= len(subjects[test_sid]):
                        continue
                    test_mins = quantized[test_sid].get(test_imp)
                    if test_mins is None:
                        continue

                    for vault_sid, vault in vaults.items():
                        if vault_sid == test_sid:
                            # Genuine attempt
                            ga += 1
                            if unlock_vault(test_mins, vault):
                                gacc += 1
                            else:
                                gr += 1
                        else:
                            # Impostor attempt
                            ia += 1
                            if unlock_vault(test_mins, vault):
                                iacc += 1
                            else:
                                ir += 1

            # Compute metrics
            far = (iacc / ia * 100) if ia > 0 else 0.0
            frr = (gr / ga * 100) if ga > 0 else 0.0
            eer = (far + frr) / 2

            elapsed = time.time() - t0
            print(f"  [{combo_idx}/{total_combos}] f={factor}, k={degree}: "
                  f"GA={ga}, IA={ia}, FAR={far:.4f}%, FRR={frr:.4f}%, "
                  f"EER={eer:.4f}% ({elapsed:.1f}s)")

            all_results.append({
                "factor": factor,
                "degree": degree,
                "ga": ga, "ia": ia,
                "gacc": gacc, "iacc": iacc,
                "gr": gr, "ir": ir,
                "far": round(far, 6),
                "frr": round(frr, 6),
                "eer": round(eer, 6),
            })

    # Save results
    csv_path = output_dir / f"results_setup{setup}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    json_path = output_dir / f"results_setup{setup}.json"
    with open(json_path, "w") as f:
        json.dump({
            "db_name": db_name,
            "setup": setup,
            "n_subjects": n_subjects,
            "factors": factors,
            "degrees": degrees,
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to {csv_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unimodal FP Vault Baseline")
    parser.add_argument("--db", default="fvc2002_1", help="Database name")
    parser.add_argument("--setup", type=int, default=1, choices=[1, 2])
    parser.add_argument("--factors", nargs="*", type=int, default=None)
    parser.add_argument("--degrees", nargs="*", type=int, default=None)
    args = parser.parse_args()

    run_experiment(
        db_name=args.db,
        setup=args.setup,
        factors=args.factors,
        degrees=args.degrees,
    )
