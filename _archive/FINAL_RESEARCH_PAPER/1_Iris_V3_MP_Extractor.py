import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path

unimodal_path = "/Users/suryanallaparaju/Desktop/Surya's Project/code/unimodal_implementation/2002_1"
sys.path.insert(0, unimodal_path)
iris_lib_path = "/Users/suryanallaparaju/Desktop/Surya's Project/code/multimodal_implementation/iris_engine"
sys.path.insert(0, iris_lib_path)

from iris_extractor_v3 import IrisExtractorV3

def process_iris(args):
    img_path, out_path = args
    if os.path.exists(out_path): return
    try:
        extractor = IrisExtractorV3()
        feature_vector = extractor.process_single_image(img_path)
        if feature_vector is not None:
            with open(out_path, "w") as f:
                f.write(",".join(map(str, feature_vector)))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    base_iris = "/Users/suryanallaparaju/Desktop/Surya's Project/data/Iris/CASIA-Iris-Interval"
    out_dir = "/Users/suryanallaparaju/Desktop/Surya's Project/data/Iris/Extracted_Codes_Daugman_Full"
    os.makedirs(out_dir, exist_ok=True)
    
    tasks = []
    base_path = Path(base_iris)
    for u in range(1, 101):
        class_folder = f"{u:03d}"
        user_dir = base_path / class_folder
        
        if not user_dir.exists(): continue
            
        images = sorted([p for p in user_dir.rglob("*") if p.suffix.lower() in ['.jpg', '.bmp']])
        
        for i, img_path in enumerate(images[:8]):
            impression = i + 1
            out_file = os.path.join(out_dir, f"{u}_{impression}.txt")
            if not os.path.exists(out_file):
                tasks.append((str(img_path), out_file))
                
    cores = cpu_count()
    print(f"[*] Found {len(tasks)} target images for extraction.")
    print(f"[*] Firing up {cores} parallel cores...")
    
    if len(tasks) > 0:
        with Pool(cores) as p:
            list(tqdm(p.imap(process_iris, tasks), total=len(tasks), desc="Daugman Extraction"))
    else:
        print("[*] All codes already extracted.")

if __name__ == "__main__":
    main()
