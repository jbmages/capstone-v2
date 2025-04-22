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

from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_squared_error,
                             log_loss, classification_report)
from scipy.spatial import distance
import numpy as np


class ClusteringModel:
    def __init__(self):
        self.labels_ = None
        self.scores = {}

    def _homegrown_silhouette(self, X, labels):
        n = X.shape[0]
        unique_labels = set(labels)
        sil_scores = np.zeros(n)

        for i in range(n):
            same_cluster = X[labels == labels[i]]
            other_clusters = [X[labels == label] for label in unique_labels if label != labels[i]]

            a = np.mean([np.linalg.norm(X[i] - x) for x in same_cluster if not np.array_equal(x, X[i])]) if len(same_cluster) > 1 else 0
            b = np.min([np.mean([np.linalg.norm(X[i] - x) for x in cluster]) for cluster in other_clusters]) if other_clusters else 0
            sil_scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0

        return np.mean(sil_scores)

    def _homegrown_davies_bouldin(self, X, labels):
        clusters = [X[labels == k] for k in np.unique(labels)]
        centroids = [np.mean(c, axis=0) for c in clusters]
        scatter = [np.mean(np.linalg.norm(c - centroids[i], axis=1)) for i, c in enumerate(clusters)]

        dbi = 0
        for i in range(len(clusters)):
            max_ratio = 0
            for j in range(len(clusters)):
                if i == j:
                    continue
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist > 0:
                    ratio = (scatter[i] + scatter[j]) / dist
                    max_ratio = max(max_ratio, ratio)
            dbi += max_ratio
        return dbi / len(clusters)

    def _homegrown_calinski_harabasz(self, X, labels):
        n_samples = X.shape[0]
        n_clusters = len(np.unique(labels))
        overall_mean = np.mean(X, axis=0)

        between = sum(len(X[labels == k]) * np.linalg.norm(np.mean(X[labels == k], axis=0) - overall_mean) ** 2 for k in np.unique(labels))
        within = sum(np.sum((X[labels == k] - np.mean(X[labels == k], axis=0)) ** 2) for k in np.unique(labels))

        return (between / (n_clusters - 1)) / (within / (n_samples - n_clusters)) if within > 0 else 0

    def evaluate(self):
        if self.labels_ is None:
            raise ValueError("No labels found. Fit the model first.")

        X = self.data
        labels = self.labels_
        unique_labels = set(labels)

        if len(unique_labels) > 1:
            if self.use_homegrown:
                self.scores = {
                    'calinski_harabasz': self._homegrown_calinski_harabasz(X, labels),
                    'davies_bouldin': self._homegrown_davies_bouldin(X, labels),
                    'silhouette': self._homegrown_silhouette(X, labels)
                }
            else:
                self.scores = {
                    'calinski_harabasz': calinski_harabasz_score(X, labels),
                    'davies_bouldin': davies_bouldin_score(X, labels),
                    'silhouette': silhouette_score(X, labels)
                }
        else:
            self.scores = {
                'calinski_harabasz': -1,
                'davies_bouldin': float('inf'),
                'silhouette': -1
            }




class KMeans(ClusteringModel):
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


class GMM(ClusteringModel):
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


class DBScan(ClusteringModel):
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


class Hierarchical(ClusteringModel):
    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params

    def fit(self):
        model = AgglomerativeClustering(
            n_clusters=self.params.get('n_clusters', 5),
            metric=self.params.get('affinity', 'euclidean'),
            linkage=self.params.get('linkage', 'ward')
        )
        self.labels_ = model.fit_predict(self.data)
        self.data = self.data


class KMeansHomegrown(ClusteringModel):
    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params
        self.k = self.params.get('n_clusters', 5)
        self.max_iters = self.params.get('max_iter', 100)
        self.tol = self.params.get('tol', 1e-6)

    def initialize_centroids(self, X):
        return np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

    def euclidean(self, datapoint, centroids):
        return np.sqrt(np.sum((centroids - datapoint)**2, axis=1))

    def fit(self):
        X = self.data
        centroids = self.initialize_centroids(X)
        assignments = 0

        for _ in range(self.max_iters):
            assignments = np.array([np.argmin(self.euclidean(x, centroids)) for x in X])
            new_centroids = np.array([np.mean(X[assignments == i], axis=0) for i in range(self.k)])

            if np.max(np.abs(new_centroids - centroids)) < self.tol:
                break

            centroids = new_centroids

        self.labels_ = assignments
        self.data = self.data  # store data for evaluation
        self.evaluate()


class GMMHomegrown(ClusteringModel):
    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params
        self.k = self.params.get('n_components', 3)
        self.max_iters = self.params.get('max_iter', 100)
        self.tol = self.params.get('tol', 1e-4)

    def _gaussian_pdf(self, X, mean, cov):
        n = X.shape[1]
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        norm_factor = 1 / np.sqrt((2 * np.pi) ** n * cov_det)
        diff = X - mean
        exp_term = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        return norm_factor * exp_term

    def fit(self):
        X = self.data
        N, D = X.shape
        np.random.seed(42)

        # Initialize parameters
        self.pi = np.full(self.k, 1 / self.k)
        self.mu = X[np.random.choice(N, self.k, replace=False)]
        self.sigma = np.array([np.eye(D)] * self.k)

        log_likelihood_old = 0
        responsibilities = 0
        for _ in range(self.max_iters):
            # E-step
            responsibilities = np.zeros((N, self.k))
            for k in range(self.k):
                responsibilities[:, k] = self.pi[k] * self._gaussian_pdf(X, self.mu[k], self.sigma[k])
            responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

            # M-step
            N_k = np.sum(responsibilities, axis=0)
            self.pi = N_k / N
            self.mu = (responsibilities.T @ X) / N_k[:, None]
            for k in range(self.k):
                diff = X - self.mu[k]
                weighted_sum = np.zeros((D, D))
                for i in range(N):
                    weighted_sum += responsibilities[i, k] * np.outer(diff[i], diff[i])
                self.sigma[k] = weighted_sum / N_k[k]

            # Check for convergence via log-likelihood
            log_likelihood = np.sum(np.log(np.sum([
                self.pi[k] * self._gaussian_pdf(X, self.mu[k], self.sigma[k]) for k in range(self.k)
            ], axis=0)))

            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

        self.labels_ = np.argmax(responsibilities, axis=1)
        self.data = self.data  # store data for evaluation
        self.evaluate()


class DBScanHomegrown(ClusteringModel):
    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params
        self.eps = self.params.get('eps', 0.5)
        self.min_samples = self.params.get('min_samples', 5)

    def fit(self):
        X = self.data
        n = X.shape[0]
        visited = np.zeros(n, dtype=bool)
        labels = np.full(n, -1)
        cluster_id = 0

        def region_query(point_idx):
            dists = distance.cdist([X[point_idx]], X)[0]
            return np.where(dists <= self.eps)[0]

        def expand_cluster(idx, neighbors, cluster_id):
            labels[idx] = cluster_id
            i = 0
            while i < len(neighbors):
                pt_idx = neighbors[i]
                if not visited[pt_idx]:
                    visited[pt_idx] = True
                    new_neighbors = region_query(pt_idx)
                    if len(new_neighbors) >= self.min_samples:
                        neighbors = np.unique(np.concatenate((neighbors, new_neighbors)))
                if labels[pt_idx] == -1:
                    labels[pt_idx] = cluster_id
                i += 1

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = region_query(i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Noise
            else:
                expand_cluster(i, neighbors, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        self.data = self.data
        self.evaluate()


class PredictionModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate(self, model, name):
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr') if y_proba is not None else None
        mse = mean_squared_error(self.y_test, y_pred)
        logloss = log_loss(self.y_test, y_proba) if y_proba is not None else None
        n = len(self.y_test)
        k = self.X_test.shape[1]
        bic = logloss * n + k * np.log(n) if logloss is not None else None

        print(f"{name} Accuracy: {acc:.4f}")
        if auc is not None: print(f"{name} AUC: {auc:.4f}")
        print(f"{name} MSE: {mse:.4f}")
        if bic is not None: print(f"{name} BIC: {bic:.2f}")
        print(f"Time: {time.time() - start:.2f} sec")
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

        return {
            'accuracy': acc,
            'auc': auc,
            'mse': mse,
            'log_loss': logloss,
            'bic': bic
        }


class LogisticRegression(PredictionModel):
    def __init__(self, X_train, X_test, y_train, y_test, params):
        super().__init__(X_train, X_test, y_train, y_test)
        self.params = params

    def run(self):
        model = SkLogisticRegression(
            penalty=self.params.get('penalty', 'l2'),
            C=self.params.get('c', 1.0),
            solver=self.params.get('solver', 'lbfgs'),
            max_iter=self.params.get('max_iter', 1000),
            n_jobs=-1
        )
        return self.evaluate(model, "Logistic Regression")


class SVM(PredictionModel):
    def __init__(self, X_train, X_test, y_train, y_test, params):
        super().__init__(X_train, X_test, y_train, y_test)
        self.params = params

    def run(self):
        model = LinearSVC(
            C=self.params.get('c', 1.0),
            loss=self.params.get('loss', 'squared_hinge'),
            max_iter=self.params.get('max_iter', 1000)
        )
        return self.evaluate(model, "SVM")


class NeuralNet(PredictionModel):
    def __init__(self, X_train, X_test, y_train, y_test, params):
        super().__init__(X_train, X_test, y_train, y_test)
        self.params = params

    def run(self):
        model = MLPClassifier(
            hidden_layer_sizes=self.params.get('hidden_layer_sizes', (128, 64)),
            alpha=self.params.get('alpha', 0.0001),
            solver=self.params.get('solver', 'adam'),
            max_iter=self.params.get('max_iter', 300),
            learning_rate_init=self.params.get('learning_rate', 0.001),
            early_stopping=True,
            random_state=42
        )
        return self.evaluate(model, "Neural Net")


class RandomForest(PredictionModel):
    def __init__(self, X_train, X_test, y_train, y_test, params):
        super().__init__(X_train, X_test, y_train, y_test)
        self.params = params

    def run(self):
        model = RandomForestClassifier(
            n_estimators=self.params.get('n_estimators', 100),
            max_depth=self.params.get('max_depth', None),
            min_samples_split=self.params.get('min_samples_split', 2),
            min_samples_leaf=self.params.get('min_samples_leaf', 1),
            max_features=self.params.get('max_features', 'auto'),
            n_jobs=-1,
            random_state=42
        )
        return self.evaluate(model, "Random Forest")