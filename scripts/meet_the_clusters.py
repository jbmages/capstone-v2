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