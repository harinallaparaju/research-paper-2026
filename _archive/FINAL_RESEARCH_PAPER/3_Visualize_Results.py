import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_results():
    csv_path = "results/Final_Research_Results.csv"
    
    if not os.path.exists(csv_path):
        print(f"[-] Data file not found at {csv_path}. Run 2_Parametric_Evaluation.py first.")
        return

    print("[*] Generating Publication-Ready Figures...")
    df = pd.read_csv(csv_path)

    # Clean Arch Column mapping dynamically
    df['Base_Arch'] = df['Architecture'].apply(lambda x: x.split(' (')[0])
    df['Mode'] = df['Architecture'].apply(lambda x: x.split('(')[1].replace(')', ''))

    os.makedirs("results/plots", exist_ok=True)
    
    # Global figure settings mimicking IEEE conventions
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")
    
    modes = df['Mode'].unique()

    for mode in modes:
        mode_df = df[df['Mode'] == mode]
        
        # Plot 1: Security vs Usability tradeoff mapping (Quantization to Error) per Architecture
        for arch in mode_df['Base_Arch'].unique():
            arch_df = mode_df[mode_df['Base_Arch'] == arch]
            
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=arch_df, x='Factor', y='EER', hue='Degree', marker='D', palette='mako', linewidth=2)
            plt.title(f"{arch} - Equal Error Rate (EER) by Vault Quantization\nEvaluation Subset: {mode}", fontweight='bold')
            plt.xlabel("Fingerprint Quantization Factor (δ)", fontweight='bold')
            plt.ylabel("Equal Error Rate (%)", fontweight='bold')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend(title='Polynomial Degree', loc='upper right')
            plt.tight_layout()
            plt.savefig(f"results/plots/{arch.replace(' ', '_')}_{mode}_Performance.png", dpi=400)
            plt.close()

        # Plot 2: Cross Architectural Comparison aggregated over all Factors/Degrees (Bar metrics)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=mode_df, x='Base_Arch', y='EER', errorbar='sd', palette='viridis', capsize=0.1)
        plt.title(f"Baseline Multimodal Security Comparison\nEvaluation Subset: {mode}", fontweight='bold')
        plt.ylabel("Mean EER (%)", fontweight='bold')
        plt.xlabel("Cryptographic Vault Protocol", fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"results/plots/Global_Architecture_Comparison_{mode}.png", dpi=400)
        plt.close()

    print(f"[+] Thesis Visualizations output perfectly to 'results/plots/'.")

if __name__ == "__main__":
    visualize_results()