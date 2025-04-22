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
from collections import Counter


class ClusteringModel:
    def __init__(self):
        self.labels_ = None
        self.scores = {}
        self.use_homegrown = False
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


class LogisticRegressionHomegrown(PredictionModel):
    def __init__(self, X_train, X_test, y_train, y_test, params):
        super().__init__(X_train, X_test, y_train, y_test)
        self.params = params
        self.lr = 0.1
        self.max_iter = self.params.get('max_iter', 1000)
        self.penalty = self.params.get('penalty', 'l2')
        self.C = self.params.get('c', 1.0)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y):
        n_classes = np.max(y) + 1
        return np.eye(n_classes)[y]

    def run(self):
        X, y = self.X_train, self.y_train
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        y_oh = self.one_hot(y)
        n_classes = y_oh.shape[1]

        self.weights = np.zeros((X.shape[1], n_classes))

        for _ in range(self.max_iter):
            logits = X @ self.weights
            probs = self.softmax(logits)
            grad = X.T @ (probs - y_oh) / X.shape[0]

            if self.penalty == 'l2':
                grad += self.weights / self.C

            self.weights -= self.lr * grad

        X_test = np.hstack([np.ones((self.X_test.shape[0], 1)), self.X_test])
        logits_test = X_test @ self.weights
        probs_test = self.softmax(logits_test)
        y_pred = np.argmax(probs_test, axis=1)

        return self.evaluate(y_pred, probs_test, "Logistic Regression (Homegrown)")


class SVMHomegrown(PredictionModel):
    def __init__(self, X_train, X_test, y_train, y_test, params):
        super().__init__(X_train, X_test, y_train, y_test)
        self.params = params
        self.C = self.params.get('c', 1.0)
        self.max_iter = self.params.get('max_iter', 1000)

    def run(self):
        X, y = self.X_train, self.y_train
        y_unique = np.unique(y)
        n_classes = len(y_unique)
        classifiers = []

        for c in y_unique:
            y_binary = np.where(y == c, 1, -1)
            w = np.zeros(X.shape[1])
            b = 0
            lr = 0.01

            for _ in range(self.max_iter):
                for i in range(len(y_binary)):
                    if y_binary[i] * (np.dot(X[i], w) + b) < 1:
                        w += lr * (y_binary[i] * X[i] - 2 * (1 / self.C) * w)
                        b += lr * y_binary[i]
                    else:
                        w -= lr * 2 * (1 / self.C) * w
            classifiers.append((w, b))

        scores = np.array([X @ w + b for w, b in classifiers]).T
        y_pred = np.argmax(scores, axis=1)

        return self.evaluate(y_pred, None, "SVM (Homegrown)")


class RandomForestHomegrown(PredictionModel):
    def __init__(self, X_train, X_test, y_train, y_test, params):
        super().__init__(X_train, X_test, y_train, y_test)
        self.params = params
        self.n_estimators = self.params.get('n_estimators', 10)
        self.max_depth = self.params.get('max_depth', 5)
        self.trees = []

    def gini_impurity(self, y):
        classes = np.unique(y)
        impurity = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity

    def best_split(self, X, y):
        best_gini = float('inf')
        best_index, best_value = None, None
        for index in range(X.shape[1]):
            thresholds = np.unique(X[:, index])
            for value in thresholds:
                left = y[X[:, index] <= value]
                right = y[X[:, index] > value]
                if len(left) == 0 or len(right) == 0:
                    continue
                gini = (len(left) * self.gini_impurity(left) + len(right) * self.gini_impurity(right)) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_index, best_value = index, value
        return best_index, best_value

    def build_tree(self, X, y, depth):
        if len(set(y)) == 1 or depth == 0:
            return Counter(y).most_common(1)[0][0]
        idx, val = self.best_split(X, y)
        if idx is None:
            return Counter(y).most_common(1)[0][0]
        left_mask = X[:, idx] <= val
        right_mask = X[:, idx] > val
        left_branch = self.build_tree(X[left_mask], y[left_mask], depth - 1)
        right_branch = self.build_tree(X[right_mask], y[right_mask], depth - 1)
        return idx, val, left_branch, right_branch

    def predict_tree(self, tree, x):
        if not isinstance(tree, tuple):
            return tree
        idx, val, left, right = tree
        if x[idx] <= val:
            return self.predict_tree(left, x)
        else:
            return self.predict_tree(right, x)

    def run(self):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(self.X_train), len(self.X_train), replace=True)
            X_sample = self.X_train[indices]
            y_sample = self.y_train[indices]
            tree = self.build_tree(X_sample, y_sample, self.max_depth)
            self.trees.append(tree)

        predictions = [np.bincount([self.predict_tree(tree, x) for tree in self.trees]).argmax() for x in self.X_test]
        return self.evaluate(np.array(predictions), None, "Random Forest (Homegrown)")


class NeuralNetHomegrown(PredictionModel):
    def __init__(self, X_train, X_test, y_train, y_test, params):
        super().__init__(X_train, X_test, y_train, y_test)
        self.params = params
        self.hidden_sizes = self.params.get('hidden_layer_sizes', (64,))
        self.alpha = self.params.get('alpha', 0.0001)
        self.learning_rate = self.params.get('learning_rate', 0.01)
        self.max_iter = self.params.get('max_iter', 200)

    def one_hot(self, y):
        n_classes = np.max(y) + 1
        return np.eye(n_classes)[y]

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def run(self):
        np.random.seed(42)
        X, y = self.X_train, self.y_train
        y_oh = self.one_hot(y)
        layers = [X.shape[1]] + list(self.hidden_sizes) + [y_oh.shape[1]]

        # Weight initialization
        weights = [np.random.randn(layers[i], layers[i+1]) * 0.01 for i in range(len(layers) - 1)]
        biases = [np.zeros((1, size)) for size in layers[1:]]

        for _ in range(self.max_iter):
            # Forward pass
            activations = [X]
            z_values = []
            for W, b in zip(weights[:-1], biases[:-1]):
                z = activations[-1] @ W + b
                z_values.append(z)
                a = self.relu(z)
                activations.append(a)
            z = activations[-1] @ weights[-1] + biases[-1]
            z_values.append(z)
            probs = self.softmax(z)
            activations.append(probs)

            # Backward pass
            delta = probs - y_oh
            grads_w = [activations[-2].T @ delta / X.shape[0]]
            grads_b = [np.sum(delta, axis=0, keepdims=True) / X.shape[0]]

            for i in range(len(layers) - 2, 0, -1):
                delta = (delta @ weights[i].T) * self.relu_deriv(z_values[i-1])
                grads_w.insert(0, activations[i-1].T @ delta / X.shape[0])
                grads_b.insert(0, np.sum(delta, axis=0, keepdims=True) / X.shape[0])

            # Regularization and update
            for i in range(len(weights)):
                grads_w[i] += self.alpha * weights[i]
                weights[i] -= self.learning_rate * grads_w[i]
                biases[i] -= self.learning_rate * grads_b[i]

        # Prediction
        def forward(X_):
            a = X_
            for i in range(len(weights) - 1):
                a = self.relu(a @ weights[i] + biases[i])
            return self.softmax(a @ weights[-1] + biases[-1])

        probs_test = forward(self.X_test)
        y_pred = np.argmax(probs_test, axis=1)

        return self.evaluate(y_pred, probs_test, "Neural Net (Homegrown)")