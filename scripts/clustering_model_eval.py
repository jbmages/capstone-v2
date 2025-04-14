import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FuncFormatter

# Output directory
output_dir = "clustering_eval"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("../model_eval/gridsearch_all_models_20250412-215308.csv")
dbscan_df = df[df["model"] == "DBScan"]
kmeans_gmm_df = df[df["model"].isin(["KMeans", "GMM"])]

sns.set(style="white", context="notebook")
plt.rcParams.update({"font.family": "serif", "axes.titlesize": 18, "axes.labelsize": 14})

# --- Aesthetic Plot 1: KMeans vs GMM Silhouette ---
plt.figure(figsize=(8, 5))
sns.violinplot(data=kmeans_gmm_df, x="model", y="silhouette", palette="pastel", inner=None)
sns.stripplot(data=kmeans_gmm_df, x="model", y="silhouette", color='black', alpha=0.5, jitter=True)
plt.title("Silhouette Distribution for KMeans vs GMM", weight='bold')
plt.xlabel("")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(f"{output_dir}/aesthetic_kmeans_gmm_silhouette.png", dpi=300)
plt.close()

# --- Aesthetic Plot 2: DBScan Tuning Heatmap (Silhouette) ---
pivot_sil = dbscan_df.pivot_table(values="silhouette", index="min_samples", columns="eps")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_sil, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Silhouette Score'})
plt.title("DBScan Hyperparameter Grid: Silhouette Scores", weight='bold')
plt.xlabel("eps")
plt.ylabel("min_samples")
plt.tight_layout()
plt.savefig(f"{output_dir}/aesthetic_dbscan_grid_silhouette.png", dpi=300)
plt.close()

# --- Aesthetic Plot 3: DBScan Tuning Heatmap (Davies-Bouldin) ---
pivot_db = dbscan_df.pivot_table(values="davies_bouldin", index="min_samples", columns="eps")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_db, annot=True, fmt=".2f", cmap="rocket_r", cbar_kws={'label': 'Davies-Bouldin Index'})
plt.title("DBScan Hyperparameter Grid: Davies-Bouldin Index", weight='bold')
plt.xlabel("eps")
plt.ylabel("min_samples")
plt.tight_layout()
plt.savefig(f"{output_dir}/aesthetic_dbscan_grid_davies.png", dpi=300)
plt.close()

# --- Aesthetic Plot 4: DBScan Tuning Heatmap (Calinski-Harabasz) ---
pivot_ch = dbscan_df.pivot_table(values="calinski_harabasz", index="min_samples", columns="eps")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_ch, annot=True, fmt=".0f", cmap="crest", cbar_kws={'label': 'Calinski-Harabasz Score'})
plt.title("DBScan Hyperparameter Grid: Calinski-Harabasz Score", weight='bold')
plt.xlabel("eps")
plt.ylabel("min_samples")
plt.tight_layout()
plt.savefig(f"{output_dir}/aesthetic_dbscan_grid_calinski.png", dpi=300)
plt.close()

# --- Aesthetic Plot 5: Silhouette vs # Clusters ---
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="n_clusters_found", y="silhouette", hue="model", palette="muted")
plt.title("Silhouette vs Number of Clusters Found", weight='bold')
plt.xlabel("Number of Clusters Found")
plt.ylabel("Silhouette Score")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{output_dir}/aesthetic_silhouette_vs_nclusters.png", dpi=300)
plt.close()

# --- Aesthetic Plot 6: Correlation Heatmap ---
metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
predictors = ["eps", "min_samples", "n_clusters", "batch_size", "max_iter", "n_clusters_found", "n_components"]
corr = df[metrics + predictors].corr().loc[predictors, metrics]

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title("Correlations: Predictors vs Evaluation Metrics", weight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/aesthetic_correlation_matrix.png", dpi=300)
plt.close()

print("\nAll aesthetic plots saved to 'clustering_eval' folder.")
