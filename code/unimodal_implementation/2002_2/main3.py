import matplotlib
matplotlib.use('TkAgg')

import unlocking
import os
import quantize2
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import csv
import numpy as np



metrics = {}  # Stores all performance data per factor and percent

def process_files(root_dir):
    vault_base_dir = "vaults"
    all_txt_files = []

    # factors and percent ranges
    f= [8,35,2]
    p= [5, 12, 1 ]

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.txt'):
                all_txt_files.append((dirpath, file))

    user_fingerprints = {}
    for dirpath, filename in all_txt_files:
        user_id = filename.split("_")[0] # Use first 3 characters as user ID
        file_path = os.path.join(dirpath, filename)
        user_fingerprints.setdefault(user_id, []).append((user_id, file_path))

    total_fingerprints = sum(len(files) for files in user_fingerprints.values())
    
    total_steps = total_fingerprints * len(range(f[0],f[1],f[2])) * len(range(p[0],p[1],p[2]))


    
    with tqdm(total=total_steps, desc="Processing", unit="step", mininterval=300) as pbar:
        for factor in range(f[0], f[1],f[2]):
            metrics[factor] = {}
            
            for percent in range(p[0],p[1],p[2]):
                #percent = i * 2
                metrics[factor][percent] = {
                    'ga': 0, 'ia': 0,
                    'gacc': 0, 'iacc': 0,
                    'gr': 0, 'ir': 0
                }

                vault_dir = os.path.join(vault_base_dir, str(factor), f"vaults{percent}")

                for genuine_user, genuine_files in user_fingerprints.items():
                    for _, genuine_path in genuine_files:
                        try:
                            with open(genuine_path, 'r') as file:
                                minutiae = [
                                    list(map(int, line.strip().split(',')))
                                    for line in file if len(line.strip().split(',')) == 3
                                ]
                                filtered = [m for m in minutiae if all(i >= 0 for i in m)]
                        except Exception as e:
                            print(f"[ERROR] Reading file {genuine_path}: {e}")
                            continue

                        points = set(quantize2.read_minutiae(filtered, factor))
                        results = unlocking.main(points, vault_dir)

                        genuine_user_id = os.path.basename(genuine_path).split("_")[0]

                        vault_files = [f for f in os.listdir(vault_dir) if f.endswith(".txt")]
                        total_impostor_attempts = sum(1 for f in vault_files if f.split("_")[0] != genuine_user_id)
                        metrics[factor][percent]['ia'] += total_impostor_attempts
                        #metrics[factor][percent]['ia'] += 9

                        found_match = False
                        impostor_accept_count = 0

                        for result_path in results:
                            matched_user_id = os.path.basename(result_path).split("_")[0]

                            if matched_user_id == genuine_user_id:
                                # Genuine Accept
                                metrics[factor][percent]['gacc'] += 1
                                found_match = True
                            else:
                                # Impostor Accept
                                metrics[factor][percent]['iacc'] += 1
                                impostor_accept_count += 1

                        # Genuine Attempt is always 1 per input
                        metrics[factor][percent]['ga'] += 1

                        if not found_match:
                            # Vaults were unlocked but none belonged to genuine user
                            metrics[factor][percent]['gr'] += 1

                        # Impostor Rejects = total IA - IAcc

                        metrics[factor][percent]['ir'] += (total_impostor_attempts - impostor_accept_count)

                        #metrics[factor][percent]['ir'] = (metrics[factor][percent]['ia'] - impostor_accept_count)

                        pbar.update(1)

    # Compute final metrics and save results
    final_results = {}
    csv_rows = [
        ["Factor", "degree", "GA", "IA", "GAcc", "IAcc", "GR", "IR", "FAR", "FRR", "EER"]
    ]

    for factor, percents in metrics.items():
        final_results[factor] = {}
        for percent, vals in percents.items():
            ga, ia = vals['ga'], vals['ia']
            gacc, iacc = vals['gacc'], vals['iacc']
            gr, ir = vals['gr'], vals['ir']

            far = (iacc / ia)*100 if ia > 0 else 0
            frr = (gr / ga)*100 if ga > 0 else 0
            eer = (far + frr) / 2

            final_results[factor][percent] = {
                **vals,
                'far': round(far, 4),
                'frr': round(frr, 4),
                'eer': round(eer, 4)
            }

            csv_rows.append([factor, percent, ga, ia, gacc, iacc, gr, ir, round(far, 4), round(frr, 4), round(eer, 4)])

    with open("results/final_results.txt", "w") as f:
        json.dump(final_results, f, indent=4)

    with open("results/ results.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)

    print("Results saved to final_results.txt and results.csv")

    # Plot FAR vs FRR (ROC) and DET
    for factor, percents in final_results.items():
        x_far, y_frr = [], []

        for percent, vals in sorted(percents.items()):
            x_far.append(vals['far'])
            y_frr.append(vals['frr'])

        plt.figure()
        plt.plot(x_far, y_frr, marker='o', label="ROC")
        plt.plot([0, 1], [0, 1], 'k--', label="EER Line")
        plt.xlabel("False Accept Rate (FAR)")
        plt.ylabel("False Reject Rate (FRR)")
        plt.title(f"ROC Curve (Factor {factor})")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"results/roc_curve_factor_{factor}.png")
        plt.close()

        # DET Curve (log-log)
        
        plt.figure()
        plt.plot(x_far, y_frr, marker='o', label="DET")
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("FAR (log scale)")
        plt.ylabel("FRR (log scale)")
        plt.title(f"DET Curve (Factor {factor})")
        plt.grid(True, which="both")
        plt.legend()
        plt.savefig(f"results/det_curve_factor_{factor}.png")
        plt.close()

    print("All plots saved.")

if __name__ == "__main__":
    minutiae_folder = r"minutiae"
    process_files(minutiae_folder)
