import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

pd.set_option('display.max_rows', None)

def meet_the_clusters(cluster_cols, idx):
    data = pd.read_csv('cluster_eval/combined_clusters.csv')

    # Select the target cluster column
    cluster_col = cluster_cols[idx]
    data = data[['O score', 'C score', 'E score', 'A score', 'N score', cluster_col]].copy()
    data.rename(columns={cluster_col: 'cluster'}, inplace=True)

    print(f"\n=== Preview of Clustering by: {cluster_col} ===")
    print(data.head(15))

    # Group by cluster
    trait_cols = ['O score', 'C score', 'E score', 'A score', 'N score']
    grouped = data.groupby('cluster')

    # Compute mean and std dev
    mean_df = grouped[trait_cols].mean().T
    std_df = grouped[trait_cols].std().T
    count_series = grouped.size()
    percent_series = count_series / len(data) * 100

    # Combine results
    summary = pd.concat([mean_df, std_df], keys=['Mean', 'StdDev'])
    summary.columns = [f'Cluster {c}' for c in summary.columns]
    summary.index.names = ['Stat', 'Trait']

    # Add n and percent as separate rows
    count_row = pd.DataFrame([count_series], index=['n'])
    percent_row = pd.DataFrame([percent_series], index=['%'])

    count_row.columns = [f'Cluster {c}' for c in count_row.columns]
    percent_row.columns = [f'Cluster {c}' for c in percent_row.columns]

    # Display final table
    full_summary = pd.concat([summary, count_row, percent_row])
    print("\n=== Table 1: Clustering Analysis of Big Five Traits ===")
    print(full_summary)

    # === Heatmap of Mean Trait Scores by Cluster ===
    cluster_ids = sorted(data['cluster'].unique())  # <-- fix added here!
    heatmap_data = mean_df.copy()
    heatmap_data.columns = [f"Cluster {c}" for c in cluster_ids]

    plt.figure(figsize=(1.5 * len(cluster_ids), 5))
    sns.heatmap(
        heatmap_data.T,
        annot=True,
        cmap='coolwarm',
        vmin=28, vmax=34,
        cbar_kws={'label': 'Mean Trait Score'},
        linewidths=0.5,
        linecolor='white'
    )
    plt.title("Heatmap of Big Five Trait Means by Cluster", fontsize=16)
    plt.ylabel("Cluster")
    plt.xlabel("Trait")
    plt.tight_layout()
    plt.show()

# Example usage
cluster_cols = ['gmm_4_both_cluster', 'gmm_3_both_cluster',
                'gmm_3_survey_cluster', 'gmm_5_survey_cluster']
meet_the_clusters(cluster_cols, 0)
#meet_the_clusters('GMM_Cluster)

#thing = pd.read_csv('../data/data_with_clusters.csv')
#print(thing.head(15))


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def thing():
    # Output directory
    output_dir = "cluster_eval"
    os.makedirs(output_dir, exist_ok=True)

    # File paths
    gmm_3_both_file = 'cluster_labels_GMM_n_components-3_covariance_type-tied_max_iter-100_20250415-021545.csv'
    gmm_4_both_file = 'cluster_labels_GMM_n_components-4_covariance_type-tied_max_iter-100_20250415-021634.csv'
    gmm_3_survey_file = 'cluster_labels_GMM_n_components-3_covariance_type-tied_max_iter-100_20250415-021106.csv'
    gmm_5_survey_file = 'cluster_labels_GMM_n_components-5_covariance_type-tied_max_iter-100_20250415-021235.csv'

    # Load data
    gmm_4_both = pd.read_csv(gmm_4_both_file)
    gmm_3_both = pd.read_csv(gmm_3_both_file)
    gmm_3_survey = pd.read_csv(gmm_3_survey_file)
    gmm_5_survey = pd.read_csv(gmm_5_survey_file)

    # Base df
    df = gmm_4_both.copy()

    # Sanity check row counts
    n = len(df)
    assert all(len(x) == n for x in [gmm_3_both, gmm_3_survey, gmm_5_survey]), "Row count mismatch!"

    # Add renamed cluster cols to base
    df['gmm_3_both_cluster'] = gmm_3_both['cluster'].values
    df['gmm_3_survey_cluster'] = gmm_3_survey['cluster'].values
    df['gmm_5_survey_cluster'] = gmm_5_survey['cluster'].values
    df = df.rename(columns={'cluster': 'gmm_4_both_cluster'})  # from original df

    # Show result
    print(df.head())

    # Save
    df.to_csv(os.path.join(output_dir, 'combined_clusters.csv'), index=False)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
results = pd.read_csv('../model_eval/gridsearch_all_models_20250412-215308.csv')

# Filter out DBScan
results = results[results['model'].str.lower() != 'dbscan']

# Compute DB / CH ratio
results['db_ch_ratio'] = results['davies_bouldin'] / results['calinski_harabasz']

# Create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=results,
    x='db_ch_ratio',
    y='silhouette',
    hue='model',
    style='data_type',
    palette='tab10',
    s=100
)

plt.xlabel('Davies-Bouldin / Calinski-Harabasz (Lower is Better)')
plt.ylabel('Silhouette Score (Higher is Better)')
plt.title('Clustering Quality (Excl. DBScan): Silhouette vs. DB/CH Ratio')
plt.grid(True)
plt.gca().invert_xaxis()  # Reverse x-axis to show better models on right
plt.tight_layout()
plt.show()
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
results = pd.read_csv('../model_eval/gridsearch_all_models_20250412-215308.csv')

# Filter to GMM and KMeans
results = results[results['model'].str.lower().isin(['gmm', 'kmeans'])]
results['model'] = results['model'].str.upper()

# Metrics to plot
metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
titles = {
    'silhouette': 'Silhouette Score (Higher is Better)',
    'calinski_harabasz': 'Calinski-Harabasz Score (Higher is Better)',
    'davies_bouldin': 'Davies-Bouldin Index (Lower is Better)'
}

# Plot heatmap for each metric
sns.set(style="whitegrid", font_scale=1.2)

for metric in metrics:
    pivot = results.pivot_table(
        index='n_clusters_found',
        columns='model',
        values=metric,
        aggfunc='mean'
    )

    if pivot.empty:
        continue

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap='YlGnBu',
        linewidths=0.3,
        linecolor='gray',
        cbar_kws={'label': metric}
    )
    plt.title(titles[metric], fontsize=16, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Clusters Found')
    plt.tight_layout()
    plt.show()
