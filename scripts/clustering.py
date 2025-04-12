import os
import time
import itertools
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# Import all model classes from Models.py
from models import KMeans, GMM, DBScan, Hierarchical


class ClusteringWorkflow:
    def __init__(self, data, scoring_table, cluster_data, model_space,
                 max_time=120, data_subset=0.4, save_results=True,
                 apply_factor_analysis=False, n_factors=5):
        self.raw_data = data
        self.scoring = scoring_table
        self.model_space = model_space
        self.max_time = max_time
        self.save_results = save_results
        self.apply_factor_analysis = apply_factor_analysis
        self.n_factors = n_factors

        # Model class map
        self.MODEL_CLASS_MAP = {
            'KMeans': KMeans,
            'GMM': GMM,
            'DBScan': DBScan,
            'Hierarchical': Hierarchical
        }

        # Prepares self.data for clustering
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
        data = self.raw_data[cols].copy()
        data = data.sample(frac=subset, random_state=42)

        if self.apply_factor_analysis:
            print(f"[INFO] Applying Factor Analysis with {self.n_factors} factors...")
            fa = FactorAnalysis(n_components=self.n_factors, random_state=42)
            data = fa.fit_transform(data)

        scaler = StandardScaler()
        self.data = scaler.fit_transform(data)

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
                    **param_dict,
                    **model.scores
                }
                results.append(result)

                print(f"[RESULT] {model_name} | Params: {param_dict} | Scores: {model.scores}")

                if time.time() - start_time > self.max_time:
                    print("[WARNING] Max time exceeded. Stopping early.")
                    break

        if self.save_results:
            os.makedirs("model_eval", exist_ok=True)
            df = pd.DataFrame(results)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df.to_csv(f"model_eval/clustering_results_{timestamp}.csv", index=False)
            print(f"[SAVED] Results written to model_eval/clustering_results_{timestamp}.csv")

        return results
