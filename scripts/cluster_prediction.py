import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import warnings
import time

import matplotlib.pyplot as plt
import seaborn as sns

class ClusterPredictor:
    def __init__(self, data, scoring, cluster_type='gmm_4_both_cluster', test_size=0.2, use_subset=True, subset_size=100000):
        self.data = data.copy()
        self.scoring = scoring
        self.cluster_type = cluster_type
        self.test_size = test_size
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.models = {}
        self.accuracies = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.cpu_count = max(1, os.cpu_count() - 1)
        self._prepare_data()

    def _prepare_data(self):
        print("Preparing data...")
        id_to_question = dict(zip(self.scoring['id'], self.scoring['trait']))
        original_cols = list(self.data.columns[:50])
        new_cols = [f"{col}: {id_to_question.get(col, '')}" for col in original_cols]
        self.data.columns = new_cols + list(self.data.columns[50:])

        max_val = self.data[self.cluster_type].max()
        original_count = len(self.data)
        print('DATASET LENGTH', original_count)
        self.filtered_data = self.data[self.data[self.cluster_type] != max_val]
        print(f"Filtered out {original_count - len(self.filtered_data)} rows ({(original_count - len(self.filtered_data))/original_count:.2%})")

        if self.use_subset and len(self.filtered_data) > self.subset_size:
            print(f"Using subset of {self.subset_size} rows")
            self.filtered_data = self.filtered_data.sample(self.subset_size, random_state=42)

        self.X = self.filtered_data.iloc[:, :50]
        self.y = self.filtered_data[self.cluster_type]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def _train_and_evaluate(self, model, name):
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        y_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr') if y_proba is not None else None
        mse = mean_squared_error(self.y_test, y_pred)
        logloss = log_loss(self.y_test, y_proba) if y_proba is not None else None
        n = len(self.y_test)
        k = len(self.X_test.columns)
        bic = logloss * n + k * np.log(n) if logloss is not None else None

        print(f"{name} Accuracy: {acc:.4f}")
        if auc is not None: print(f"{name} AUC: {auc:.4f}")
        print(f"{name} MSE: {mse:.4f}")
        if bic is not None: print(f"{name} BIC: {bic:.2f}")
        print(f"Time: {time.time() - start:.2f} sec")
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

        self.models[name] = model
        self.accuracies[name] = acc
        self.model_scores[name] = {
            'accuracy': acc,
            'auc': auc,
            'mse': mse,
            'log_loss': logloss,
            'bic': bic
        }

        return acc

    def train_all(self):
        self.model_scores = {}

        self._train_and_evaluate(
            RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=self.cpu_count, random_state=42),
            "Random Forest"
        )
        self._train_and_evaluate(
            LogisticRegression(max_iter=200, n_jobs=self.cpu_count, random_state=42),
            "Logistic Regression"
        )
        self._train_and_evaluate(
            SVC(kernel='linear', probability=True, random_state=42),
            "SVM"
        )
        self._train_and_evaluate(
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
            "Neural Net"
        )
        self._train_and_evaluate(
            XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False,
                          eval_metric='mlogloss', random_state=42),
            "XGBoost"
        )

        best_model_name = max(self.accuracies, key=self.accuracies.get)
        self.best_model = self.models[best_model_name]
        print(f"\n✅ Best model: {best_model_name} with accuracy {self.accuracies[best_model_name]:.4f}")

    def print_model_metrics(self):
        if not hasattr(self, "model_scores"):
            print("No models trained yet. Please run train_all() first.")
            return

        metrics_df = pd.DataFrame(self.model_scores).T
        print("\n=== Model Evaluation Metrics ===")
        print(metrics_df.round(4))

        plt.figure(figsize=(10, 5))
        sns.heatmap(metrics_df, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Model Evaluation Metrics")
        plt.tight_layout()
        plt.show()