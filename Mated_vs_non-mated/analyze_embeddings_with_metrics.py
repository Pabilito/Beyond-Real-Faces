import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyeer.eer_info import get_eer_stats

# Load similarity scores
def load_scores(filename):
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

# File paths
mated_file = "Similarities/HyperFace_mated_similarities.txt"
non_mated_file = "Similarities/HyperFace_non_mated_similarities.txt"

mated_sims = load_scores(mated_file)
non_mated_sims = load_scores(non_mated_file)

# Compute stats using PyEER
stats = get_eer_stats(non_mated_sims, mated_sims, hformat=False, ds_scores=True)

# Print metrics to console
print(f"Genuine mean/std: {stats.gmean:.4f} / {stats.gstd:.4f}")
print(f"Impostor mean/std: {stats.imean:.4f} / {stats.istd:.4f}")
print(f"EER: {stats.eer*100:.2f}% at threshold {stats.eer_th:.4f}")
print(f"FMR100: {stats.fmr100*100:.2f}% at threshold {stats.fmr100_th:.4f}")
print(f"FMR1000: {stats.fmr1000*100:.2f}% at threshold {stats.fmr1000_th:.4f}")

# Create directories if they don't exist
os.makedirs("Figures_with_metrics", exist_ok=True)
os.makedirs("Metrics", exist_ok=True)

# Generate base name
base_name = os.path.splitext(os.path.basename(mated_file))[0]
n_comparisons = len(mated_sims) + len(non_mated_sims)

# Save metrics to CSV
metrics = {
    "Genuine Mean": [stats.gmean],
    "Genuine Std": [stats.gstd],
    "Impostor Mean": [stats.imean],
    "Impostor Std": [stats.istd],
    "EER": [stats.eer],
    "EER Threshold": [stats.eer_th],
    "FMR100": [stats.fmr100],
    "FMR100 Threshold": [stats.fmr100_th],
    "FMR1000": [stats.fmr1000],
    "FMR1000 Threshold": [stats.fmr1000_th],
    "Total Comparisons": [n_comparisons]
}
metrics_df = pd.DataFrame(metrics)
csv_path = os.path.join("Metrics", f"{base_name}_metrics.csv")
metrics_df.to_csv(csv_path, index=False)
print(f"Metrics saved to: {csv_path}")

# Plot and save KDE
plt.figure(figsize=(12, 7))
sns.kdeplot(mated_sims, label="Mated (Genuine)", fill=True, alpha=0.5)
sns.kdeplot(non_mated_sims, label="Non-mated (Impostor)", fill=True, alpha=0.5)

# Add vertical lines for thresholds
plt.axvline(stats.eer_th, color='red', linestyle='-', linewidth=2, label=f"EER")
plt.axvline(stats.fmr100_th, color='green', linestyle='--', linewidth=2, label="FMR100")
plt.axvline(stats.fmr1000_th, color='blue', linestyle=':', linewidth=2, label="FMR1000")

# Customize plot
plt.title(f"Cosine Similarity Distribution ({len(mated_sims)} mated, {len(non_mated_sims)} non-mated)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.legend()

# Save plot
plot_path = os.path.join("Figures_with_metrics", f"{base_name}_MatedVsNonmated_{n_comparisons}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.show()
