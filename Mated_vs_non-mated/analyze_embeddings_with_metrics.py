import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyeer.eer_info import get_eer_stats

FONTSIZE =30 

# Load similarity scores
def load_scores(filename):
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

# Create output directories
os.makedirs("Figures_with_metrics", exist_ok=True)
os.makedirs("Metrics", exist_ok=True)

# Input directory
scores_dir = "SimilarityScores"
all_files = os.listdir(scores_dir)

# Initialize list for storing all metrics
all_metrics = []

# Find mated files
non_mated_files = [f for f in all_files if f.endswith("non_mated_similarities.txt")]

for non_mated_file in non_mated_files:
    base_name = non_mated_file.replace("_non_mated_similarities.txt", "")
    mated_file = f"{base_name}_mated_similarities.txt"

    if non_mated_file not in all_files:
        print(f"Warning: Missing non-mated file for {base_name}, skipping...")
        continue

    # Load scores
    mated_path = os.path.join(scores_dir, mated_file)
    non_mated_path = os.path.join(scores_dir, non_mated_file)
    mated_sims = load_scores(mated_path)
    non_mated_sims = load_scores(non_mated_path)

    # Compute stats
    stats = get_eer_stats(mated_sims, non_mated_sims, hformat=False, ds_scores=False)

    # Print to console
    print(f"\nDataset: {base_name}")
    print(f"Genuine mean/std: {stats.gmean:.4f} / {stats.gstd:.4f}")
    print(f"Impostor mean/std: {stats.imean:.4f} / {stats.istd:.4f}")
    print(f"EER: {stats.eer*100:.2f}% at threshold {stats.eer_th:.4f}")
    print(f"FMR100: {stats.fmr100*100:.2f}% at threshold {stats.fmr100_th:.4f}")
    print(f"FMR1000: {stats.fmr1000*100:.2f}% at threshold {stats.fmr1000_th:.4f}")

    # Save metrics to list (for master CSV)
    n_comparisons = len(mated_sims) + len(non_mated_sims)
    all_metrics.append({
        "Dataset": base_name,
        "Genuine Mean": stats.gmean,
        "Genuine Std": stats.gstd,
        "Impostor Mean": stats.imean,
        "Impostor Std": stats.istd,
        "EER": stats.eer,
        "EER Threshold": stats.eer_th,
        "FMR100": stats.fmr100,
        "FMR100 Threshold": stats.fmr100_th,
        "FMR1000": stats.fmr1000,
        "FMR1000 Threshold": stats.fmr1000_th,
        "Total Comparisons": n_comparisons
    })

    # Plot KDE
    plt.figure(figsize=(12, 7))
    ax = sns.kdeplot(mated_sims, label="Mated (Genuine)", fill=True, alpha=0.5)
    sns.kdeplot(non_mated_sims, label="Non-mated (Impostor)", fill=True, alpha=0.5)

    # Threshold lines
    plt.axvline(stats.eer_th, color='red', linestyle='-', linewidth=2, label="EER threshold")
    plt.axvline(stats.fmr100_th, color='green', linestyle='--', linewidth=2, label="FMR100 threshold")
    plt.axvline(stats.fmr1000_th, color='blue', linestyle=':', linewidth=2, label="FMR1000 threshold")

    # Labels and range
    plt.xlim(-0.5, 1)
    #plt.title(f"{base_name} - Cosine Similarity Distribution\n({len(mated_sims)} mated, {len(non_mated_sims)} non-mated)")
    plt.xlabel("Cosine Similarity", fontsize=FONTSIZE)
    plt.ylabel("Density", fontsize=FONTSIZE)
    #plt.legend(fontsize=20)

    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    # Keep only left and bottom spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save figure
    plot_path = os.path.join("Figures_with_metrics", f"big_{base_name}_MatedVsNonmated.pdf")
    plt.savefig(plot_path, dpi=1200, bbox_inches='tight', format="pdf")
    plt.close()
    print(f"Plot saved to: {plot_path}")

'''
# Save all metrics to a master CSV
master_df = pd.DataFrame(all_metrics)
master_csv_path = os.path.join("Metrics", "All_Datasets_metrics.csv")
master_df.to_csv(master_csv_path, index=False)
print(f"\nMaster metrics file saved to: {master_csv_path}")
'''