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
