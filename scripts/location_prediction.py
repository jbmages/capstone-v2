import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

class ImprovedPredictiveModel:
    def __init__(self, data, model_save_path='models/improved_rf_predictive_model.joblib', sample_frac=0.6 ):
        self.data = data.copy()
        self.sample_frac = sample_frac
        self.target_options = ['region', 'sub-region', 'country', 'latlong']
        self.selected_target = None
        self.X, self.y = None, None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_save_path = model_save_path

    def join_country_data(self, country_metadata_path='scoring/country-data.csv'):
        """
        Join external country metadata (including region/sub-region) into the main dataset.
        Group countries with small sample sizes into their region.
        Sample the dataset by the specified fraction.
        """
        country_data = pd.read_csv(country_metadata_path)
        self.data = self.data.merge(country_data, on='country', how='left')
        self.group_small_countries_by_region(min_count=5000)

        print("Class distribution BEFORE sampling:")
        print(self.data['country'].value_counts().head(20))

        if 0 < self.sample_frac < 1:
            sample_n = int(len(self.data) * self.sample_frac)
            self.data = self.data.sample(n=sample_n, random_state=42)
            print(f"\nSampled {sample_n} rows ({self.sample_frac*100:.1f}%).")
        else:
            print("Using full dataset (no sampling).")

        print("\nClass distribution AFTER sampling:")
        print(self.data['country'].value_counts().head(20))

    def group_small_countries_by_region(self, min_count=5000):
        """
        Replace countries with fewer than `min_count` respondents with their region name.
        """
        if 'region' not in self.data.columns or 'country' not in self.data.columns:
            raise ValueError("Data must contain 'region' and 'country' columns before grouping.")

        value_counts = self.data['country'].value_counts()
        small_countries = value_counts[value_counts < min_count].index

        original_unique = self.data['country'].nunique()
        self.data['country'] = self.data.apply(
            lambda row: row['region'] if row['country'] in small_countries else row['country'], axis=1
        )
        print(f"Grouped {len(small_countries)} countries with < {min_count} rows into their regions.")
        print(f"Country labels reduced from {original_unique} to {self.data['country'].nunique()}")

    def analyze_target_correlations(self):
        """Analyze and visualize the correlation between trait scores and geography."""
        trait_cols = ['O score', 'C score', 'E score', 'A score', 'N score']
        targets = ['region', 'sub-region', 'country']
        corr_results = {}

        for target in targets:
            if target in self.data.columns:
                encoded_target = LabelEncoder().fit_transform(self.data[target].astype(str))
                self.data[f'{target}_encoded'] = encoded_target
                corr = self.data[trait_cols].corrwith(pd.Series(encoded_target))
                corr_results[target] = corr

        corr_df = pd.DataFrame(corr_results)
        sns.heatmap(corr_df, annot=True, cmap='coolwarm')
        plt.title("Correlation between Trait Scores and Geographic Targets")
        plt.tight_layout()
        plt.show()

    def prepare_features(self, include_times=True):
        question_cols = [col for col in self.data.columns if col[:3] in ['EXT', 'AGR', 'CSN', 'OPN', 'EST'] and len(col) <= 5]
        if include_times:
            time_cols = [col + '_E' for col in question_cols if col + '_E' in self.data.columns]
            self.feature_cols = question_cols + time_cols
        else:
            self.feature_cols = question_cols
        self.X = self.data[self.feature_cols]

    def set_target(self, target):
        if target == 'latlong':
            self.y = self.data[['lat_appx_lots_of_err', 'long_appx_lots_of_err']]
        else:
            self.y = self.label_encoder.fit_transform(self.data[target].astype(str))
            class_names = self.label_encoder.classes_
            print("\nLabel encoding for target '{}':".format(target))
            for i, name in enumerate(class_names):
                print(f"{i}: {name}")
        self.selected_target = target

    def split_and_resample(self, test_size=0.2):
        self.X = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

        if len(np.unique(self.y_train)) < 20 and not isinstance(self.y_train[0], (list, np.ndarray)):
            class_counts = dict(Counter(self.y_train))
            if len(class_counts) > 1:
                sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                undersampling_strategy = {sorted_counts[0][0]: sorted_counts[1][1]}
                undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)
                self.X_train, self.y_train = undersampler.fit_resample(self.X_train, self.y_train)
                print("Applied undersampling to balance the classes.")

    def train_model(self, use_random_search=True):
        rf = RandomForestClassifier(random_state=42)
        if use_random_search:
            param_dist = {
                'n_estimators': [100, 150, 200],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, random_state=42, n_jobs=-1)
            search.fit(self.X_train, self.y_train)
            self.model = search.best_estimator_
            print("Best parameters:", search.best_params_)
        else:
            self.model = rf.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("\nTest Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(self.model, self.model_save_path)
        print(f"Model saved at {self.model_save_path}")

    def run(self, target='region', country_metadata_path='data/country-data.csv'):
        self.join_country_data(country_metadata_path)
        self.analyze_target_correlations()
        self.prepare_features()
        self.set_target(target)
        self.split_and_resample()
        self.train_model()
        self.evaluate_model()
        self.save_model()
