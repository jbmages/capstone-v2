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
import numpy as np


class ClusteringModel:
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