import pandas as pd
import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

import scripts.utils as utils


class ImprovedClusteringModel:

    def __init__(self, dataset: pd.DataFrame, scoring, mode='score', max_rows=-1):
        """Initialize clustering model object"""
        print('Initiating modeling infrastructure...')
        self.max_rows = max_rows
        self.max_r_bool = max_rows > 0
        if self.max_r_bool:
            self.data = dataset.sample(n=max_rows, random_state=42).copy()
        self.data = dataset
        self.scoring = scoring
        self.mode = mode
        self.survey_data, self.time_data, self.score_data = self.prep_data()
        self.model_results = {}

    def prep_data(self):
        """Prepare data segments for clustering"""
        print('Preparing data for modeling...')
        survey_answer_cols = self.scoring['id'].tolist()
        time_cols = [col + '_E' for col in survey_answer_cols]
        score_cols = ['O score', 'C score', 'E score', 'A score', 'N score']

        scaler = StandardScaler()

        survey_data = self.data[survey_answer_cols] if all(col in self.data.columns for col in survey_answer_cols) else None
        time_data = self.data[time_cols] if all(col in self.data.columns for col in time_cols) else None
        score_data = self.data[score_cols] if all(col in self.data.columns for col in score_cols) else None

        return (
            scaler.fit_transform(survey_data) if survey_data is not None else None,
            scaler.fit_transform(time_data) if time_data is not None else None,
            scaler.fit_transform(score_data) if score_data is not None else None
        )

    def get_active_data(self):
        if self.mode == 'survey':
            return self.survey_data
        elif self.mode == 'time':
            return self.time_data
        else:
            return self.score_data

    def evaluate_clustering(self, data, method_name, labels, skip_sil=False):
        """Calculate evaluation metrics for a clustering result"""
        if not skip_sil:
            sil_score = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
        else:
            sil_score = -1
        db_score = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else float('inf')
        return {'method': method_name, 'silhouette': sil_score, 'davies_bouldin': db_score, 'labels': labels}

    def assign_clusters(self):
        """Try multiple clustering methods with varying hyperparameters and pick best"""
        data = self.get_active_data()
        results = []
        best_score = -1
        best_labels = None
        best_method = None

        for k in range(2, 10):
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
            kmeans_labels = kmeans.fit_predict(data)
            eval_kmeans = self.evaluate_clustering(data, f'KMeans_k={k}', kmeans_labels)
            results.append(eval_kmeans)

            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm_labels = gmm.fit_predict(data)
            eval_gmm = self.evaluate_clustering(data, f'GMM_k={k}', gmm_labels)
            results.append(eval_gmm)

        for eps in [0.3, 0.5, 0.7]:
            for min_samples in [3, 5]:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                try:
                    db_labels = dbscan.fit_predict(data)
                    eval_db = self.evaluate_clustering(data, f'DBSCAN_eps={eps}_min={min_samples}', db_labels)
                    results.append(eval_db)
                except Exception as e:
                    print(f"DBSCAN failed for eps={eps}, min_samples={min_samples}: {e}")

        for res in results:
            if res['silhouette'] > best_score:
                best_score = res['silhouette']
                best_labels = res['labels']
                best_method = res['method']

        self.data['BestCluster'] = best_labels
        self.best_method = best_method
        if not self.max_r_bool:
            self.data.to_csv(f'data/data_with_best_clusters_{self.mode}.csv', index=False)
        print(f"Best clustering method: {best_method} with silhouette score: {best_score:.3f} (mode: {self.mode})")
        return results

    def visualize_results(self, results):
        """Plot silhouette and Davies-Bouldin scores across methods"""
        df = pd.DataFrame(results)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.barplot(data=df, y='method', x='silhouette', palette='coolwarm')
        plt.title('Silhouette Scores')
        plt.axvline(x=df['silhouette'].max(), color='black', linestyle='--')

        plt.subplot(1, 2, 2)
        sns.barplot(data=df, y='method', x='davies_bouldin', palette='viridis')
        plt.title('Davies-Bouldin Scores (lower is better)')
        plt.axvline(x=df['davies_bouldin'].min(), color='black', linestyle='--')

        plt.tight_layout()
        plot_path = f"data/clustering_evaluation_metrics_{self.mode}.png"
        plt.savefig(plot_path)
        print(f"Clustering evaluation plot saved to {plot_path}.")

    def csv_to_json(self, sample_frac=0.1, max_rows=10000):
        start_time = time.time()
        selected_columns = list(self.data.columns[:50]) + ["BestCluster"]
        df_subset = self.data[selected_columns]
        num_rows = min(int(len(df_subset) * sample_frac), max_rows)
        df_sample = df_subset.sample(n=num_rows, random_state=42)
        df_sample["BestCluster"] = pd.to_numeric(df_sample["BestCluster"], errors="coerce")

        data_json = df_sample.to_dict(orient="records")
        output_dir = "dashboard/dash-data"
        os.makedirs(output_dir, exist_ok=True)
        output_json = os.path.join(output_dir, f"cluster_data_{self.mode}.json")

        with open(output_json, "w") as f:
            json.dump(data_json, f, indent=2)

        file_size = os.path.getsize(output_json) / (1024 * 1024)
        end_time = time.time()

        print(f"JSON saved at {output_json} ({file_size:.2f} MB)")
        print(f"Process took {end_time - start_time:.2f} seconds")
        return output_json
