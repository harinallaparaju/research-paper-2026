"""
Multimodal Evaluation Framework.

Runs all fusion architectures on the chimeric database and computes
GA, IA, GAcc, IAcc, GR, IR, FAR, FRR, EER for each parameter combination.

Supports:
    - Architecture A (Decision AND)
    - Architecture B (Polynomial Blinding)
    - Architecture C (Iris-Seeded Dense Chaff)
    - Architecture D (AES Encrypt)
    - Unimodal baseline (fingerprint only)

Setup 1: Lock=impression 1, Test=impression 2
Setup 2: Lock=impression 1, Test=all impressions
"""

import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unimodal.fingerprint_vault import (
    read_minutiae_file,
    quantize_minutiae,
    create_vault,
    unlock_vault,
)
from architectures.architecture_a import lock_vault_a, unlock_vault_a
from architectures.architecture_b import lock_vault_b, unlock_vault_b
from architectures.architecture_c import lock_vault_c, unlock_vault_c
from architectures.architecture_d import lock_vault_d, unlock_vault_d


# Paths
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CHIMERIC_DIR = Path(__file__).resolve().parent / "chimeric_db"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "multimodal"


def load_chimeric_data(
    fp_db: str = "fvc2002_1",
    min_iris: int = 2,
) -> List[dict]:
    """
    Load chimeric mapping and associated data.

    Returns list of subjects, each with:
        {
            "chimeric_id": int,
            "minutiae": {imp_idx: [(x,y,t), ...]},
            "iris": {imp_idx: {"code": np.array, "mask": np.array}},
        }
    """
    mapping_path = CHIMERIC_DIR / f"chimeric_{fp_db}_min{min_iris}.json"
    with open(mapping_path) as f:
        mapping = json.load(f)

    subjects = []
    for entry in mapping["mapping"]:
        subj = {
            "chimeric_id": entry["chimeric_id"],
            "minutiae": {},
            "iris": {},
        }

        for pair in entry["pairs"]:
            imp = pair["impression"] - 1  # 0-indexed

            # Load fingerprint minutiae
            fp_path = pair["fp_file"]
            subj["minutiae"][imp] = read_minutiae_file(fp_path)

            # Load iris code
            iris_path = pair["iris_file"]
            iris_data = np.load(iris_path)
            subj["iris"][imp] = {
                "code": iris_data["code"],
                "mask": iris_data["mask"],
            }

        subjects.append(subj)

    return subjects


def run_architecture_a(
    subjects: List[dict],
    setup: int,
    factor: int,
    degree: int,
    iris_threshold: float,
) -> dict:
    """
    Run Architecture A evaluation for a single parameter combination.

    Returns metrics dict.
    """
    n_subj = len(subjects)

    # Determine test impressions
    lock_imp = 0
    test_imps = [1] if setup == 1 else list(range(len(next(iter(subjects[0]["minutiae"].keys())) if isinstance(next(iter(subjects[0]["minutiae"].keys())), int) else 0, max(subjects[0]["minutiae"].keys()) + 1)))

    # Fix test impressions
    if setup == 1:
        test_imps = [1]
    else:
        test_imps = sorted(subjects[0]["minutiae"].keys())

    # Create vaults for all subjects
    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        iris = subj["iris"].get(lock_imp)
        if not mins or iris is None:
            continue

        vault = lock_vault_a(
            mins, iris["code"], iris["mask"],
            factor, degree, iris_threshold
        )
        if vault is not None:
            vaults[cid] = vault

    # Run unlock attempts
    ga, ia, gacc, iacc, gr, ir = 0, 0, 0, 0, 0, 0

    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]

        for test_imp in test_imps:
            test_mins = test_subj["minutiae"].get(test_imp, [])
            test_iris = test_subj["iris"].get(test_imp)
            if not test_mins or test_iris is None:
                continue

            for vault_cid, vault in vaults.items():
                is_genuine = (vault_cid == test_cid)

                success = unlock_vault_a(
                    test_mins, test_iris["code"], test_iris["mask"], vault
                )

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

    far = (iacc / ia * 100) if ia > 0 else 0
    frr = (gr / ga * 100) if ga > 0 else 0
    eer = (far + frr) / 2

    return {
        "architecture": "A",
        "factor": factor,
        "degree": degree,
        "iris_threshold": iris_threshold,
        "ga": ga, "ia": ia,
        "gacc": gacc, "iacc": iacc,
        "gr": gr, "ir": ir,
        "far": round(far, 6),
        "frr": round(frr, 6),
        "eer": round(eer, 6),
    }


def run_architecture_d(
    subjects: List[dict],
    setup: int,
    factor: int,
    degree: int,
) -> dict:
    """Run Architecture D evaluation."""
    lock_imp = 0
    test_imps = [1] if setup == 1 else sorted(subjects[0]["minutiae"].keys())

    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        iris = subj["iris"].get(lock_imp)
        if not mins or iris is None:
            continue

        vault = lock_vault_d(mins, iris["code"], iris["mask"], factor, degree)
        if vault is not None:
            vaults[cid] = vault

    ga, ia, gacc, iacc, gr, ir = 0, 0, 0, 0, 0, 0

    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        for test_imp in test_imps:
            test_mins = test_subj["minutiae"].get(test_imp, [])
            test_iris = test_subj["iris"].get(test_imp)
            if not test_mins or test_iris is None:
                continue

            for vault_cid, vault in vaults.items():
                is_genuine = (vault_cid == test_cid)
                success = unlock_vault_d(
                    test_mins, test_iris["code"], test_iris["mask"], vault
                )

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

    far = (iacc / ia * 100) if ia > 0 else 0
    frr = (gr / ga * 100) if ga > 0 else 0
    eer = (far + frr) / 2

    return {
        "architecture": "D",
        "factor": factor,
        "degree": degree,
        "iris_threshold": None,
        "ga": ga, "ia": ia,
        "gacc": gacc, "iacc": iacc,
        "gr": gr, "ir": ir,
        "far": round(far, 6),
        "frr": round(frr, 6),
        "eer": round(eer, 6),
    }


def _run_generic_bd(
    subjects: List[dict],
    setup: int,
    factor: int,
    degree: int,
    lock_fn,
    unlock_fn,
    arch_label: str,
) -> dict:
    """Generic evaluation runner for architectures B, C (no iris_threshold param)."""
    lock_imp = 0
    test_imps = [1] if setup == 1 else sorted(subjects[0]["minutiae"].keys())

    vaults = {}
    for subj in subjects:
        cid = subj["chimeric_id"]
        mins = subj["minutiae"].get(lock_imp, [])
        iris = subj["iris"].get(lock_imp)
        if not mins or iris is None:
            continue

        vault = lock_fn(mins, iris["code"], iris["mask"], factor, degree)
        if vault is not None:
            vaults[cid] = vault

    ga, ia, gacc, iacc, gr, ir = 0, 0, 0, 0, 0, 0

    for test_subj in subjects:
        test_cid = test_subj["chimeric_id"]
        for test_imp in test_imps:
            test_mins = test_subj["minutiae"].get(test_imp, [])
            test_iris = test_subj["iris"].get(test_imp)
            if not test_mins or test_iris is None:
                continue

            for vault_cid, vault in vaults.items():
                is_genuine = (vault_cid == test_cid)
                success = unlock_fn(
                    test_mins, test_iris["code"], test_iris["mask"], vault
                )

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

    far = (iacc / ia * 100) if ia > 0 else 0
    frr = (gr / ga * 100) if ga > 0 else 0
    eer = (far + frr) / 2

    return {
        "architecture": arch_label,
        "factor": factor,
        "degree": degree,
        "iris_threshold": None,
        "ga": ga, "ia": ia,
        "gacc": gacc, "iacc": iacc,
        "gr": gr, "ir": ir,
        "far": round(far, 6),
        "frr": round(frr, 6),
        "eer": round(eer, 6),
    }


def run_architecture_b(
    subjects: List[dict], setup: int, factor: int, degree: int,
) -> dict:
    """Run Architecture B (Polynomial Blinding) evaluation."""
    return _run_generic_bd(
        subjects, setup, factor, degree,
        lock_vault_b, unlock_vault_b, "B",
    )


def run_architecture_c(
    subjects: List[dict], setup: int, factor: int, degree: int,
) -> dict:
    """Run Architecture C (Iris-Seeded Dense Chaff) evaluation."""
    return _run_generic_bd(
        subjects, setup, factor, degree,
        lock_vault_c, unlock_vault_c, "C",
    )


def run_full_evaluation(
    fp_db: str = "fvc2002_1",
    setup: int = 1,
    architectures: List[str] = None,
    factors: List[int] = None,
    degrees: List[int] = None,
    iris_thresholds: List[float] = None,
):
    """
    Run full evaluation across architectures and parameters.
    """
    if architectures is None:
        architectures = ["A"]
    if factors is None:
        factors = [20, 24, 28]  # Subset for speed
    if degrees is None:
        degrees = [7, 9, 11]
    if iris_thresholds is None:
        iris_thresholds = [0.32, 0.35, 0.38]

    min_iris = 2 if setup == 1 else 8
    output_dir = RESULTS_DIR / fp_db
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chimeric data: {fp_db}, min_iris={min_iris}")
    subjects = load_chimeric_data(fp_db, min_iris)
    print(f"Loaded {len(subjects)} chimeric subjects")

    all_results = []

    for arch in architectures:
        print(f"\n=== Architecture {arch} ===")

        for factor in factors:
            for degree in degrees:
                if arch == "A":
                    for tau in iris_thresholds:
                        t0 = time.time()
                        result = run_architecture_a(
                            subjects, setup, factor, degree, tau
                        )
                        elapsed = time.time() - t0
                        print(f"  f={factor} k={degree} τ={tau}: "
                              f"FAR={result['far']:.4f}% FRR={result['frr']:.4f}% "
                              f"EER={result['eer']:.4f}% ({elapsed:.1f}s)")
                        all_results.append(result)

                elif arch in ("B", "C", "D"):
                    run_fn = {
                        "B": run_architecture_b,
                        "C": run_architecture_c,
                        "D": run_architecture_d,
                    }[arch]
                    t0 = time.time()
                    result = run_fn(subjects, setup, factor, degree)
                    elapsed = time.time() - t0
                    print(f"  f={factor} k={degree}: "
                          f"FAR={result['far']:.4f}% FRR={result['frr']:.4f}% "
                          f"EER={result['eer']:.4f}% ({elapsed:.1f}s)")
                    all_results.append(result)

    # Save results
    csv_path = output_dir / f"multimodal_setup{setup}.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    json_path = output_dir / f"multimodal_setup{setup}.json"
    with open(json_path, "w") as f:
        json.dump({
            "fp_db": fp_db,
            "setup": setup,
            "n_subjects": len(subjects),
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to {csv_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal Evaluation")
    parser.add_argument("--db", default="fvc2002_1")
    parser.add_argument("--setup", type=int, default=1, choices=[1, 2])
    parser.add_argument("--arch", nargs="*", default=["A"])
    parser.add_argument("--factors", nargs="*", type=int, default=None)
    parser.add_argument("--degrees", nargs="*", type=int, default=None)
    args = parser.parse_args()

    run_full_evaluation(
        fp_db=args.db,
        setup=args.setup,
        architectures=args.arch,
        factors=args.factors,
        degrees=args.degrees,
    )
