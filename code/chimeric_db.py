"""
Chimeric Multi-Modal Database Builder.

Creates a mapping between FVC fingerprint subjects and CASIA iris subjects
for multimodal biometric fusion experiments.

Chimeric approach (standard in multimodal biometrics research):
    - FVC subjects (100 users × 8 impressions) provide fingerprint minutiae
    - CASIA subjects (198 users × variable L-eye images) provide iris codes
    - Random 1-to-1 mapping creates "virtual" multimodal identities

References:
    Ross, A., & Jain, A. (2003). Information fusion in biometrics.
    Pattern Recognition Letters, 24(13), 2115-2125.
"""

import json
import random
import numpy as np
from pathlib import Path


# Data directories
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
FP_MINUTIAE_DIRS = {
    "fvc2002_1": DATA_ROOT / "minutiae" / "2002" / "Db1_a",
    "fvc2002_2": DATA_ROOT / "minutiae" / "2002" / "Db2_a",
    "fvc2002_3": DATA_ROOT / "minutiae" / "2002" / "Db3_a",
    "fvc2004_1": DATA_ROOT / "minutiae" / "2004" / "Db1_a",
}
IRIS_CODES_DIR = DATA_ROOT / "Iris" / "iris_codes"
OUTPUT_DIR = Path(__file__).resolve().parent / "chimeric_db"


def _get_fp_subjects(minutiae_dir: Path) -> dict:
    """
    Discover fingerprint subjects and their impressions.

    Returns:
        {subject_id: [file_path, ...]}  sorted by impression number
    """
    subjects = {}
    for f in sorted(minutiae_dir.glob("*.txt")):
        # Filename: {subject}_{impression}.jpg.txt
        name = f.stem  # e.g., "1_1.jpg"
        parts = name.split("_")
        if len(parts) < 2:
            continue
        subj_id = int(parts[0])
        if subj_id not in subjects:
            subjects[subj_id] = []
        subjects[subj_id].append(f)

    # Sort each subject's files by impression number
    for subj_id in subjects:
        subjects[subj_id].sort(key=lambda p: int(p.stem.split("_")[1].replace(".jpg", "")))

    return subjects


def _get_iris_subjects(codes_dir: Path) -> dict:
    """
    Discover iris subjects and their codes.

    Returns:
        {subject_id: [npz_path, ...]}  sorted by filename
    """
    subjects = {}
    for f in sorted(codes_dir.glob("*.npz")):
        # Filename: S{subj}_L_{stem}.npz  e.g., S001_L_S1001L01.npz
        parts = f.stem.split("_")
        subj_id = parts[0]  # e.g., "S001"
        if subj_id not in subjects:
            subjects[subj_id] = []
        subjects[subj_id].append(f)

    return subjects


def build_chimeric_mapping(
    fp_db: str = "fvc2002_1",
    min_iris_images: int = 2,
    seed: int = 42,
) -> dict:
    """
    Build chimeric database mapping: FVC subject ↔ CASIA subject.

    For each FVC subject with N impressions, we need a CASIA subject
    with at least `min_iris_images` images. Impression i of the FVC subject
    is paired with image i of the CASIA subject (or the last available iris
    image if CASIA subject has fewer images).

    Args:
        fp_db: Which fingerprint database to use (e.g., "fvc2002_1")
        min_iris_images: Minimum iris images required per subject
        seed: Random seed for reproducible mapping

    Returns:
        Mapping dict with structure:
        {
            "fp_db": str,
            "seed": int,
            "mapping": [
                {
                    "chimeric_id": int,
                    "fp_subject": int,
                    "iris_subject": str,
                    "pairs": [
                        {"impression": int, "fp_file": str, "iris_file": str},
                        ...
                    ]
                },
                ...
            ]
        }
    """
    random.seed(seed)

    fp_dir = FP_MINUTIAE_DIRS[fp_db]
    fp_subjects = _get_fp_subjects(fp_dir)
    iris_subjects = _get_iris_subjects(IRIS_CODES_DIR)

    # Filter iris subjects with enough images
    eligible_iris = {
        sid: files for sid, files in iris_subjects.items()
        if len(files) >= min_iris_images
    }

    n_fp = len(fp_subjects)
    n_iris = len(eligible_iris)

    if n_iris < n_fp:
        print(f"WARNING: Only {n_iris} iris subjects with >={min_iris_images} images, "
              f"but {n_fp} FP subjects. Some FP subjects will be dropped.")

    # Random 1-to-1 mapping
    fp_ids = sorted(fp_subjects.keys())
    iris_ids = sorted(eligible_iris.keys())
    random.shuffle(iris_ids)

    n_pairs = min(n_fp, n_iris)
    mapping = []

    for i in range(n_pairs):
        fp_id = fp_ids[i]
        iris_id = iris_ids[i]

        fp_files = fp_subjects[fp_id]
        iris_files = eligible_iris[iris_id]

        pairs = []
        for imp_idx, fp_file in enumerate(fp_files):
            # Use corresponding iris image, or last available if fewer iris images
            iris_idx = min(imp_idx, len(iris_files) - 1)
            pairs.append({
                "impression": imp_idx + 1,
                "fp_file": str(fp_file),
                "iris_file": str(iris_files[iris_idx]),
            })

        mapping.append({
            "chimeric_id": i + 1,
            "fp_subject": fp_id,
            "iris_subject": iris_id,
            "n_fp_impressions": len(fp_files),
            "n_iris_images": len(iris_files),
            "pairs": pairs,
        })

    result = {
        "fp_db": fp_db,
        "seed": seed,
        "min_iris_images": min_iris_images,
        "n_chimeric_subjects": n_pairs,
        "n_fp_subjects_available": n_fp,
        "n_iris_subjects_eligible": n_iris,
        "mapping": mapping,
    }

    return result


def save_chimeric_mapping(mapping: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    """Save chimeric mapping to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_db = mapping["fp_db"]
    min_imgs = mapping["min_iris_images"]
    out_path = output_dir / f"chimeric_{fp_db}_min{min_imgs}.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    return out_path


def build_all_mappings(seed: int = 42):
    """Build chimeric mappings for all FVC databases."""
    for fp_db in FP_MINUTIAE_DIRS:
        for min_imgs in [2, 8]:  # Setup 1 needs >=2, Setup 2 needs >=8
            print(f"\n--- {fp_db}, min_iris={min_imgs} ---")
            mapping = build_chimeric_mapping(fp_db, min_iris_images=min_imgs, seed=seed)
            out_path = save_chimeric_mapping(mapping)
            print(f"  Chimeric subjects: {mapping['n_chimeric_subjects']}")
            print(f"  FP subjects: {mapping['n_fp_subjects_available']}")
            print(f"  Iris eligible: {mapping['n_iris_subjects_eligible']}")
            print(f"  Saved: {out_path}")


if __name__ == "__main__":
    build_all_mappings()
