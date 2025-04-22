import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import itertools
import joblib
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scripts.models import *

class PredictionWorkflow:
    def __init__(self, data, scoring, cluster_type='gmm_4_both_cluster', test_size=0.2, use_subset=True, subset_size=100000, params={},
                 time_limit = 2400):
        self.data = data.copy()
        self.scoring = scoring
        self.cluster_type = cluster_type
        self.test_size = test_size
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.params = params
        self.scaler = StandardScaler()
        self.time_limit = time_limit
        self.best_model = None
        self.best_model_name = None

        self.MODEL_CLASS_MAP = {
            'LogisticRegression': LogisticRegression,
            'SVM': SVM,
            'NeuralNet': NeuralNet,
            'RandomForest': RandomForest,
            'HGLogisticRegression': LogisticRegressionHomegrown,
            'HGSVM': SVMHomegrown,
            'HGNeuralNet': NeuralNetHomegrown,
            'HGRandomForest': RandomForestHomegrown
        }

    def _prepare_data(self):
        print("Preparing data...")

        # Map column IDs to trait descriptions
        id_to_question = dict(zip(self.scoring['id'], self.scoring['trait']))
        original_cols = list(self.data.columns[:50])
        new_cols = [f"{col}: {id_to_question.get(col, '')}" for col in original_cols]
        self.data.columns = new_cols + list(self.data.columns[50:])

        # Print dataset info
        original_count = len(self.data)
        print('ORIGINAL COUNT', original_count)
        print('DATASET LENGTH', original_count)

        # Filtering logic (placeholder for now)
        self.filtered_data = self.data
        print(
            f"Filtered out {original_count - len(self.filtered_data)} rows ({(original_count - len(self.filtered_data)) / original_count:.2%})")

        # Subset if needed
        if self.use_subset and len(self.filtered_data) > self.subset_size:
            print(f"Using subset of {self.subset_size} rows")
            self.filtered_data = self.filtered_data.sample(n=self.subset_size, random_state=42).reset_index(drop=True)
        else:
            self.filtered_data = self.filtered_data.reset_index(drop=True)

        # Feature and label extraction
        self.X = self.filtered_data.iloc[:, :50]
        self.y = self.filtered_data[self.cluster_type]

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )

        # Reset indices to avoid mismatch during evaluation
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def grid_search(self):
        print("\nStarting grid search...")
        results = []
        start_time = time.time()

        for model_name, config in self.params.items():

            if time.time() - start_time > self.time_limit:
                print("[WARNING] Max time exceeded. Ending early.")
                break

            print(f"\nModel: {model_name}")
            keys, values = zip(*config['params'].items())
            model_class = self.MODEL_CLASS_MAP.get(config['class'])

            if not model_class:
                print(f"Unknown model class: {config['class']}")
                continue

            for param_combo in itertools.product(*values):
                param_dict = dict(zip(keys, param_combo))
                print(f"Evaluating with params: {param_dict}")

                model = model_class(
                    self.X_train_scaled,
                    self.X_test_scaled,
                    self.y_train,
                    self.y_test,
                    param_dict
                )

                scores = model.run()
                scores.update({
                    'model': model_name,
                    **param_dict
                })
                scores['model_instance'] = model
                results.append(scores)

        results_df = pd.DataFrame(results)
        results_df.sort_values(by='accuracy', ascending=False, inplace=True)
        self.best_model = results_df.iloc[0]['model_instance']
        self.best_model_name = results_df.iloc[0]['model']

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("model_eval", exist_ok=True)
        results_df.drop(columns=['model_instance'], inplace=True)
        results_df.to_csv(f"model_eval/sml_grid_search_results_{timestamp}.csv", index=False)
        print(f"\nSaved results to model_eval/sml_grid_search_results_{timestamp}.csv")

        return results_df

    def save_best_model(self, output_path="model_eval/final_model.joblib"):
        if self.best_model is None:
            raise ValueError("No best model found. Run grid_search() first.")

        joblib.dump(self.best_model, output_path)
        print(f"\nSaved best model ({self.best_model_name}) to {output_path}")
