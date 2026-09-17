import os
import sys
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import itertools

from architectures.Architecture_A import evaluate_arch_a
from architectures.Architecture_B import evaluate_arch_b
from architectures.Architecture_C import evaluate_arch_c

def evaluate_single_setup(args):
    """
    Worker function to evaluate a single hyperparameter setup.
    """
    arch_name, arch_func, factor, degree, mode = args
    
    db_fvc = "/Users/suryanallaparaju/Desktop/Surya's Project/data/minutiae/2002/Db1_a"
    iris_dir = "/Users/suryanallaparaju/Desktop/Surya's Project/data/Iris/Extracted_Codes"
    
    try:
        # Evaluate using the exact mathematical protocol mappings
        res = arch_func(
            db_fvc=db_fvc,
            iris_dir=iris_dir,
            factor=factor,
            degree=degree,
            mode=mode
        )
        
        # Structure the row safely
        return {
            "Architecture": f"{arch_name} ({mode})",
            "Factor": res["Factor"],
            "Degree": res["Degree"],
            "GA": res["GA"],
            "IA": res["IA"],
            "GAcc": res["GAcc"],
            "IAcc": res["IAcc"],
            "FRR": res["FRR"],
            "FAR": res["FAR"],
            "EER": res["EER"]
        }
    except Exception as e:
        print(f"[-] Error in {arch_name} Factor {factor} Degree {degree}: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("--- BRUTAL MULTIPROCESSING PARAMETRIC EVALUATION ---")
    print("="*60)
    
    # 1. Complete Research Parameters Sweeps matched exactly to Unimodal logic
    factors = [2, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35]
    degrees = [5, 6, 7, 8, 9, 10, 11]
    modes = ["1_to_lock_1_to_unlock", "1_to_lock_rest_to_unlock"]
    
    arch_functions = [
        ("Architecture A", evaluate_arch_a),
        ("Architecture B", evaluate_arch_b),
        ("Architecture C", evaluate_arch_c)
    ]
    
    # 2. Build task queue
    tasks = []
    for arch_tuple in arch_functions:
        arch_name, arch_func = arch_tuple
        for mode in modes:
            for factor in factors:
                for degree in degrees:
                    tasks.append((arch_name, arch_func, factor, degree, mode))
                    
    total_tasks = len(tasks)
    cores = min(cpu_count(), 8) # Safely limit to 8 to avoid memory spiking
    print(f"[*] Total mathematical multimodality checks: {total_tasks}")
    print(f"[*] Booting {cores} CPU cores for parallel execution. Fasten your seatbelts...\n")
    
    results = []
    output_csv = os.path.join(os.path.dirname(__file__), "results", "Final_Parallel_Research_Results.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 3. Fire up the Multiprocessing Pool
    with Pool(cores) as pool:
        for result in tqdm(pool.imap_unordered(evaluate_single_setup, tasks), total=total_tasks, desc="Parallel Eval"):
            if result:
                results.append(result)
                
                # Dynamic Checkpoint Auto-Save
                pd.DataFrame(results).to_csv(output_csv, index=False)

    print("\n[+] Brutal Evaluation Matrix Complete!")
    print(f"[+] All mathematically secure multimodality results saved to: {output_csv}")

if __name__ == "__main__":
    main()