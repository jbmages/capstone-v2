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

        # Remove max cluster valu  e
        max_val = self.data[self.cluster_type].max()
        original_count = len(self.data)
        print('DATASET LENGTH', original_count)
        self.filtered_data = self.data[self.data[self.cluster_type] != max_val]
        print(f"Filtered out {original_count - len(self.filtered_data)} rows ({(original_count - len(self.filtered_data))/original_count:.2%})")

        # Use subset if needed
        if self.use_subset and len(self.filtered_data) > self.subset_size:
            print(f"Using subset of {self.subset_size} rows")
            self.filtered_data = self.filtered_data.sample(self.subset_size, random_state=42)

        # Prepare X and y
        self.X = self.filtered_data.iloc[:, :50]
        self.y = self.filtered_data[self.cluster_type]

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)

        # Scale features
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

        # Pick best model
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

