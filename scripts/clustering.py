"""

Clustering workflow script.
This script manages the clustering workflow of our project,
including model training and evaluation, and hyperparameter tuning.

"""
from sklearn.preprocessing import StandardScaler
import time
import itertools
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


class Model:
    def __init__(self):
        return

    def evaluate(self):
        # Ensure that `self.labels_` is available before evaluation
        labels = self.labels_
        if labels is None:
            raise ValueError("No labels found. Please fit the model first.")

        if len(set(labels)) > 1:
            # Calculate evaluation metrics only if we have more than one cluster
            self.scores = {
                'calinski_harabasz': calinski_harabasz_score(self.data, labels),
                'davies_bouldin': davies_bouldin_score(self.data, labels),
                'silhouette': silhouette_score(self.data, labels) if len(set(labels)) < len(self.data) else -1
            }
        else:
            # If we have only one cluster, assign worst possible evaluation scores
            self.scores = {
                'calinski_harabasz': -1,
                'davies_bouldin': float('inf'),
                'silhouette': -1
            }

    def push_to_kaggle(self):
        # push model to KaggleHub for deployment
        # will develop later
        return


class KMeans(Model):

    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params
        self.model = None
        self.labels_ = None
        self.scores = {}

    def fit(self):
        self.model = MiniBatchKMeans(
            n_clusters=self.params.get('n_clusters', 5),
            random_state=42,
            batch_size=self.params.get('batch_size', 100),
            max_iter=self.params.get('max_iter', 100)
        )
        self.model.fit(self.data)
        self.labels_ = self.model.labels_


class ClusteringWorkflow:

    def __init__(self, data, scoring_table, cluster_data, max_time = 120, data_subset = 0.4,):
        self.data = data
        self.scoring = scoring_table
        self.max_time = max_time

        self.MODEL_CLASS_MAP = {
            'KMeans': KMeans
        }
        self.prep_data(cluster_data, data_subset)
        return

    def prep_data(self, cluster_data, subset):
        # Prep data for clustering

        cols = []

        if 'scores' in cluster_data:
            cols += ['O score', 'C score', 'E score', 'A score', 'N score']

        if 'survey_answers' in cluster_data:
            cols += self.scoring['id'].tolist()

        if 'time_cols' in cluster_data:
            cols += [col + '_E' for col in self.scoring['id'].tolist()]

        # Filter to only include columns that actually exist in self.data
        cols = [col for col in cols if col in self.data.columns]
        self.data = self.data[cols].copy()

        # SUBSET DATA
        num_rows = int(subset * len(self.data))
        self.data = self.data.sample(n=num_rows, random_state=42)  # Fix seed for reproducibility

        # SCALE DATA
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        print('hi pookie lol')

    def grid_search(self, model_space):
        start_time = time.time()
        results = []

        # Iterate through the model space using tqdm for progress tracking
        for model_name, config in model_space.items():

            keys, values = zip(*config['params'].items())

            # Wrap the grid search loop with tqdm for a progress bar
            for param_combo in tqdm(itertools.product(*values), desc=f"Evaluating {model_name}", unit="model",
                                    ncols=100):

                param_dict = dict(zip(keys, param_combo))

                # Instantiate the model class based on the string in the model space
                model_class = self.MODEL_CLASS_MAP.get(config['class'])
                if model_class is None:
                    raise ValueError(f"Unknown model class: {config['class']}")

                model = model_class(self.data, param_dict)

                # Fit and evaluate the model
                model.fit()
                model.evaluate()

                # Print evaluation metrics for this model configuration
                print(f"Model: {model_name}, Params: {param_dict}")
                print(f"Scores - Calinski Harabasz: {model.scores['calinski_harabasz']}, "
                      f"Davies Bouldin: {model.scores['davies_bouldin']}, "
                      f"Silhouette: {model.scores['silhouette']}")

                # Store the results
                result = {
                    'model': model_name,
                    'params': param_dict,
                    'scores': model.scores
                }
                results.append(result)

                # Optional: stop if we exceed max time
                if time.time() - start_time > self.max_time:
                    print("Max time exceeded. Ending grid search early.")
                    break

        return results
