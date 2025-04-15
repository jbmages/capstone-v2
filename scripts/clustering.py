import os
import time
import itertools
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from scripts.models import KMeans, GMM, DBScan, Hierarchical
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class ClusteringWorkflow:
    def __init__(self, data, scoring_table, cluster_data, model_space,
                 max_time=1200, data_subset=0.06, save_results=True,
                 apply_factor_analysis=False, n_factors=5):
        self.raw_data = data
        self.scoring = scoring_table
        self.model_space = model_space
        self.cluster_data_type = cluster_data
        self.max_time = max_time
        self.save_results = save_results
        self.apply_factor_analysis = apply_factor_analysis
        self.n_factors = n_factors

        self.MODEL_CLASS_MAP = {
            'KMeans': KMeans,
            'GMM': GMM,
            'DBScan': DBScan,
            'Hierarchical': Hierarchical
        }

        self.prep_data(cluster_data, data_subset)

    def prep_data(self, cluster_data, subset):
        cols = []

        if 'scores' in cluster_data:
            cols += ['O score', 'C score', 'E score', 'A score', 'N score']
        if 'survey_answers' in cluster_data:
            cols += self.scoring['id'].tolist()
        if 'time_cols' in cluster_data:
            cols += [col + '_E' for col in self.scoring['id'].tolist()]

        cols = [col for col in cols if col in self.raw_data.columns]
        self.sampled_raw = self.raw_data[cols].copy().sample(frac=subset, random_state=42).reset_index(drop=True)

        data = self.sampled_raw.copy()
        if self.apply_factor_analysis:
            print(f"[INFO] Applying Factor Analysis: {self.n_factors} factors")
            fa = FactorAnalysis(n_components=self.n_factors, random_state=42)
            data = fa.fit_transform(data)

        self.data = StandardScaler().fit_transform(data)

    def save_cluster_assignments(self, model_name, param_dict, labels):
        df_clusters = self.sampled_raw.copy()
        df_clusters["cluster"] = labels

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        param_str = '_'.join(f"{k}-{v}" for k, v in param_dict.items())
        label_tag = f"{model_name}_{param_str}_{timestamp}"

        os.makedirs("model_eval", exist_ok=True)
        csv_path = f"model_eval/cluster_labels_{label_tag}.csv"
        json_path = f"model_eval/cluster_labels_{label_tag}.json"

        df_clusters.to_csv(csv_path, index=False)
        df_clusters.to_json(json_path, orient="records", indent=2)
        print(f"[SAVED] Cluster labels to:\n  {csv_path}\n  {json_path}")

    def grid_search(self):
        results = []
        start_time = time.time()

        for model_name, config in self.model_space.items():
            keys, values = zip(*config['params'].items())
            model_class = self.MODEL_CLASS_MAP.get(config['class'])
            if not model_class:
                raise ValueError(f"Unknown model class: {config['class']}")

            for param_combo in tqdm(itertools.product(*values), desc=f"Evaluating {model_name}", ncols=100):
                param_dict = dict(zip(keys, param_combo))
                model = model_class(self.data, param_dict)
                model.fit()
                model.evaluate()

                result = {
                    'model': model_name,
                    'data_type': '+'.join(self.cluster_data_type),
                    'factor_analysis': self.apply_factor_analysis,
                    **param_dict,
                    **model.scores,
                }

                if hasattr(model, 'labels_'):
                    result['n_clusters_found'] = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
                    self.save_cluster_assignments(model_name, param_dict, model.labels_)

                results.append(result)

                df_partial = pd.DataFrame(results)
                os.makedirs("model_eval", exist_ok=True)
                timestamp = time.strftime("%Y%m%d")
                partial_path = f"model_eval/live_grid_results_{timestamp}.csv"
                df_partial.to_csv(partial_path, index=False)

                print(f"[RESULT] {model_name} | {param_dict} | {model.scores}")

                if time.time() - start_time > self.max_time:
                    print("[WARNING] Max time exceeded. Ending early.")
                    break

        if self.save_results:
            os.makedirs("model_eval", exist_ok=True)
            df = pd.DataFrame(results)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df.to_csv(f"model_eval/clustering_results_{timestamp}.csv", index=False)
            print(f"[SAVED] clustering_results_{timestamp}.csv")

        return results


    def kde_density_comparison(self, labels, n_trials=10):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(self.data)
        log_densities = kde.score_samples(self.data)
        actual_avg_density = np.mean([log_densities[i] for i in range(len(labels))])

        null_densities = []
        for _ in range(n_trials):
            permuted_labels = np.random.permutation(labels)
            null_avg = np.mean([log_densities[i] for i in range(len(permuted_labels))])
            null_densities.append(null_avg)

        null_mean = np.mean(null_densities)
        density_gain = actual_avg_density - null_mean
        return {
            'avg_density': actual_avg_density,
            'null_avg_density': null_mean,
            'density_gain': density_gain
        }



