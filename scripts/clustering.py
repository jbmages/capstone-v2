import os
import time
import itertools
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


class Model:
    def __init__(self):
        self.labels_ = None
        self.scores = {}

    def evaluate(self):
        if self.labels_ is None:
            raise ValueError("No labels found. Fit the model first.")

        unique_labels = set(self.labels_)
        if len(unique_labels) > 1:
            self.scores = {
                'calinski_harabasz': calinski_harabasz_score(self.data, self.labels_),
                'davies_bouldin': davies_bouldin_score(self.data, self.labels_),
                'silhouette': silhouette_score(self.data, self.labels_)
            }
        else:
            self.scores = {
                'calinski_harabasz': -1,
                'davies_bouldin': float('inf'),
                'silhouette': -1
            }


class KMeans(Model):
    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params

    def fit(self):
        model = MiniBatchKMeans(
            n_clusters=self.params.get('n_clusters', 5),
            batch_size=self.params.get('batch_size', 100),
            max_iter=self.params.get('max_iter', 100),
            random_state=42
        )
        model.fit(self.data)
        self.labels_ = model.labels_
        self.data = self.data  # Needed for evaluation


class ClusteringWorkflow:
    def __init__(self, data, scoring_table, cluster_data, model_space,
                 max_time=120, data_subset=0.4, save_results=True):
        self.raw_data = data
        self.scoring = scoring_table
        self.model_space = model_space
        self.max_time = max_time
        self.save_results = save_results

        self.MODEL_CLASS_MAP = {
            'KMeans': KMeans,
            # Add GMM and DBSCAN here as needed
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

        # Sanity check for actual existence
        cols = [col for col in cols if col in self.raw_data.columns]
        self.data = self.raw_data[cols].sample(frac=subset, random_state=42)

        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

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

                print(f"Evaluated {model_name} with {param_dict} → Scores: {model.scores}")

                if time.time() - start_time > self.max_time:
                    print("Max time exceeded. Stopping early.")
                    break

        if self.save_results:
            os.makedirs("model_eval", exist_ok=True)
            df = pd.DataFrame(results)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            df.to_csv(f"model_eval/clustering_results_{timestamp}.csv", index=False)
            print(f"Results saved to model_eval/clustering_results_{timestamp}.csv")

        return results
