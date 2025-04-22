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

class PredictionWorkflow:
    def __init__(self, data, scoring, cluster_type='gmm_4_both_cluster', test_size=0.2, use_subset=True, subset_size=100000, params = {}):
        self.data = data.copy()
        self.scoring = scoring
        self.cluster_type = cluster_type
        self.test_size = test_size
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.params = params

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

    def grid_search(self):
        """
        execute supervised predictive grid search
        """

        pass
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