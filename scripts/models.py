import os
import time
import itertools
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.decomposition import FactorAnalysis


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
        self.data = self.data


class GMM(Model):
    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params

    def fit(self):
        model = GaussianMixture(
            n_components=self.params.get('n_components', 5),
            covariance_type=self.params.get('covariance_type', 'full'),
            max_iter=self.params.get('max_iter', 100),
            random_state=42
        )
        model.fit(self.data)
        self.labels_ = model.predict(self.data)
        self.data = self.data


class DBScan(Model):
    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params

    def fit(self):
        model = DBSCAN(
            eps=self.params.get('eps', 0.5),
            min_samples=self.params.get('min_samples', 5)
        )
        model.fit(self.data)
        self.labels_ = model.labels_
        self.data = self.data


class Hierarchical(Model):
    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params

    def fit(self):
        model = AgglomerativeClustering(
            n_clusters=self.params.get('n_clusters', 5),
            affinity=self.params.get('affinity', 'euclidean'),
            linkage=self.params.get('linkage', 'ward')
        )
        self.labels_ = model.fit_predict(self.data)
        self.data = self.data
