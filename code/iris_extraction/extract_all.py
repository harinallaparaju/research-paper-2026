"""
Batch Iris Feature Extraction from CASIA-Iris-Interval dataset.

Processes all subjects/eyes/images through the full pipeline:
    Segmentation → Normalization → Encoding → Save IrisCode

Output: .npz files containing {code, mask} per image.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris_extraction.segmentation import segment_iris
from iris_extraction.normalization import normalize_iris
from iris_extraction.encoding import encode_iris, encode_iris_2d


# Configuration
CASIA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "Iris" / "CASIA-Iris-Interval"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "Iris" / "iris_codes"
OUTPUT_ROOT_2D = Path(__file__).resolve().parent.parent.parent / "data" / "Iris" / "iris_codes_2d"

EYE = "L"   # Use left eye only for consistency


def extract_single_image(image_path: str, use_2d: bool = False) -> dict:
    """
    Extract IrisCode from a single image.

    Args:
        image_path: Path to the iris image
        use_2d: If True, use 2D Log-Gabor encoding (393K bits)

    Returns:
        dict with keys: success, code, mask, failure_reason,
        pupil_center, pupil_radius, iris_center, iris_radius
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"success": False, "failure_reason": f"Cannot read {image_path}"}

    # Stage 1: Segmentation
    seg = segment_iris(image)
    if not seg.success:
        return {"success": False, "failure_reason": seg.failure_reason}

    # Stage 2: Normalization
    normalized, norm_mask = normalize_iris(
        image,
        seg.pupil_center, seg.pupil_radius,
        seg.iris_center, seg.iris_radius,
        seg.noise_mask
    )

    # Stage 3: Encoding
    if use_2d:
        iris_code, code_mask = encode_iris_2d(normalized, norm_mask)
    else:
        iris_code, code_mask = encode_iris(normalized, norm_mask)

    return {
        "success": True,
        "code": iris_code,
        "mask": code_mask,
        "pupil_center": seg.pupil_center,
        "pupil_radius": seg.pupil_radius,
        "iris_center": seg.iris_center,
        "iris_radius": seg.iris_radius,
    }


def extract_all(
    casia_root: Path = CASIA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    eye: str = EYE,
    subjects: list = None,
    use_2d: bool = False,
):
    """
    Batch extract IrisCodes from all CASIA-Iris-Interval images.

    Args:
        casia_root: Path to CASIA-Iris-Interval root directory
        output_root: Where to save .npz iris code files
        eye: Which eye to process ("L" or "R")
        subjects: List of subject IDs to process (None = all)
        use_2d: If True, use 2D Log-Gabor encoding (393K bits)
    """
    if use_2d and output_root == OUTPUT_ROOT:
        output_root = OUTPUT_ROOT_2D
    output_root.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = sorted([
            d.name for d in casia_root.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

    report = {
        "timestamp": datetime.now().isoformat(),
        "eye": eye,
        "total_subjects": len(subjects),
        "total_images": 0,
        "successful": 0,
        "failed": 0,
        "failures": [],
        "per_subject": {},
    }

    for subj_idx, subj_id in enumerate(subjects):
        eye_dir = casia_root / subj_id / eye
        if not eye_dir.exists():
            report["per_subject"][subj_id] = {"images": 0, "success": 0, "failed": 0}
            continue

        images = sorted([f for f in eye_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.bmp', '.png')])
        subj_report = {"images": len(images), "success": 0, "failed": 0}

        for img_path in images:
            report["total_images"] += 1
            result = extract_single_image(str(img_path), use_2d=use_2d)

            if result["success"]:
                # Save as .npz
                out_name = f"S{subj_id}_{eye}_{img_path.stem}.npz"
                np.savez_compressed(
                    output_root / out_name,
                    code=result["code"],
                    mask=result["mask"]
                )
                report["successful"] += 1
                subj_report["success"] += 1
            else:
                report["failed"] += 1
                subj_report["failed"] += 1
                report["failures"].append({
                    "image": str(img_path.name),
                    "subject": subj_id,
                    "reason": result["failure_reason"]
                })

        report["per_subject"][subj_id] = subj_report

        if (subj_idx + 1) % 25 == 0:
            success_rate = report["successful"] / max(1, report["total_images"]) * 100
            print(f"  [{subj_idx+1}/{len(subjects)}] "
                  f"Processed: {report['total_images']}, "
                  f"Success: {report['successful']} ({success_rate:.1f}%)")

    # Save report
    success_rate = report["successful"] / max(1, report["total_images"]) * 100
    report["success_rate_percent"] = round(success_rate, 2)

    report_path = output_root / "extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Extraction Complete ===")
    print(f"Total images: {report['total_images']}")
    print(f"Successful:   {report['successful']} ({success_rate:.1f}%)")
    print(f"Failed:       {report['failed']}")
    print(f"Report saved: {report_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract IrisCodes from CASIA-Iris-Interval")
    parser.add_argument("--eye", default="L", choices=["L", "R"], help="Which eye to process")
    parser.add_argument("--subjects", nargs="*", help="Specific subject IDs (default: all)")
    parser.add_argument("--encoding-2d", action="store_true", help="Use 2D Log-Gabor encoding (393K bits)")
    args = parser.parse_args()

    report = extract_all(eye=args.eye, subjects=args.subjects, use_2d=args.encoding_2d)
