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
import time

import matplotlib.pyplot as plt
import seaborn as sns

class ClusterPredictor:
    def __init__(self, data, scoring, cluster_type='KMeans_Cluster', test_size=0.2, use_subset=True, subset_size=100000):
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
        acc = accuracy_score(self.y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        print(f"Time: {time.time() - start:.2f} sec")
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

        self.models[name] = model
        self.accuracies[name] = acc
        return acc

    def train_all(self):
        self._train_and_evaluate(RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=self.cpu_count, random_state=42), "Random Forest")
        self._train_and_evaluate(LogisticRegression(max_iter=200, n_jobs=self.cpu_count, random_state=42), "Logistic Regression")
        self._train_and_evaluate(SVC(kernel='linear', random_state=42), "SVM")
        self._train_and_evaluate(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42), "Neural Net")

        best_model_name = max(self.accuracies, key=self.accuracies.get)
        self.best_model = self.models[best_model_name]
        print(f"\n✅ Best model: {best_model_name} with accuracy {self.accuracies[best_model_name]:.4f}")

    def save_models(self, folder='models'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for name, model in self.models.items():
            joblib.dump(model, os.path.join(folder, f"{name.replace(' ', '_').lower()}_model.joblib"))
        joblib.dump(self.scaler, os.path.join(folder, "scaler.joblib"))
        print(f"Models saved to '{folder}'")

    def get_feature_importance(self, model_name='Random Forest'):
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return

        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print(f"No feature importance method for model '{model_name}'")
            return

        importance_df = pd.DataFrame({
            'Question': self.X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        return importance_df

    def plot_model_accuracies(self):
        if not self.accuracies:
            print("No model accuracies found. Please run train_all() first.")
            return
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(self.accuracies.keys()), y=list(self.accuracies.values()), palette="viridis")
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        for i, acc in enumerate(self.accuracies.values()):
            plt.text(i, acc + 0.01, f"{acc:.2%}", ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def stepwise_feature_analysis(self, top_n=10, model_type='logistic'):
        """
        Evaluate model accuracy using top-N most important features.
        """
        print(f"\nPerforming stepwise feature analysis using {model_type} model...")

        X = self.filtered_data.iloc[:, :50]
        y = self.filtered_data[self.cluster_type]

        scaler = StandardScaler()

        if model_type == 'random_forest':
            base_model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
            base_model.fit(scaler.fit_transform(X), y)
            importances = base_model.feature_importances_
        elif model_type == 'logistic':
            base_model = LogisticRegression(max_iter=200, random_state=42)
            base_model.fit(scaler.fit_transform(X), y)
            importances = np.abs(base_model.coef_).mean(axis=0)
        else:
            raise ValueError("model_type must be 'random_forest' or 'logistic'")

        feature_names = X.columns
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        print("\nTop features:")
        for i, (name, importance) in enumerate(top_features[:top_n]):
            print(f"{i + 1}. {name} - Importance: {importance:.4f}")

        accuracies = []
        for i in range(1, top_n + 1):
            selected_cols = [name for name, _ in top_features[:i]]
            X_subset = X[selected_cols]
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if model_type == 'logistic':
                step_model = LogisticRegression(max_iter=200, random_state=42)
            else:
                step_model = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)

            step_model.fit(X_train_scaled, y_train)
            y_pred = step_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, top_n + 1), accuracies, marker='o', color='dodgerblue')
        plt.title(f"Stepwise Accuracy using Top-{top_n} Features ({model_type})")
        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.xticks(range(1, top_n + 1))
        plt.tight_layout()
        plt.show()
