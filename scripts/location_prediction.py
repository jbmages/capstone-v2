import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,classification_report,confusion_matrix
import json
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

class ImprovedPredictiveModel:
    def __init__(self, data, model_save_path='models/improved_rf_predictive_model.joblib', sample_frac=0.6 ):
        self.data = data.copy()
        self.sample_frac = sample_frac
        self.target_options = ['region', 'sub-region', 'country']
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
        Join external country dataset (region/sub-region)
        Group countries with small amounts of surveys into their region
        Sample the dataset by using a fraction.
        """
        country_data = pd.read_csv(country_metadata_path)
        self.data = self.data.merge(country_data, on='country', how='left')
        self.group_small_countries_by_region(min_count=5000)

        # print("Class distribution BEFORE sampling:")
        # print(self.data['country'].value_counts().head(5))

        if 0 < self.sample_frac < 1:
            sample_n = int(len(self.data) * self.sample_frac)
            self.data = self.data.sample(n=sample_n, random_state=42)
        

        # print("\nClass distribution AFTER sampling:")
        # print(self.data['country'].value_counts().head(5))

   
    def prepare_features(self, include_times=True):
        '''Take in question data and prepare it'''
        question_cols = [col for col in self.data.columns if col[:3] in ['EXT', 'AGR', 'CSN', 'OPN', 'EST'] and len(col) <= 5]
        if include_times:
            time_cols = [col + '_E' for col in question_cols if col + '_E' in self.data.columns]
            self.feature_cols = question_cols + time_cols
        else:
            self.feature_cols = question_cols
        self.X = self.data[self.feature_cols]

    def set_target(self, target): 
        '''Adjust for sub-region, region, country'''
        self.y = self.label_encoder.fit_transform(self.data[target].astype(str))
        self.selected_target = target

    def split_and_resample(self, test_size=0.2):
        self.X = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
############
        class_counts = Counter(self.y_train)
        undersampling_strat = {cls: min(count, 10000) for cls, count in class_counts.items()}

        undersampler = RandomUnderSampler(sampling_strategy=undersampling_strat, random_state=42)
        self.X_train, self.y_train = undersampler.fit_resample(self.X_train, self.y_train)

        print("Applied fixed-cap undersampling (max 10k per class).")
        print("Undersampled class distribution:")
        print(dict(Counter(self.y_train)))


    def train_model(self):
        """
        Train Random Forest classifier using RandomizedSearchCV and save the best model
        """
        print("Training model with RandomizedSearchCV...")

        param_grid = {
        'n_estimators': [150, 200, 250, 300],
        'max_depth': [20, 30, 40, 50, None],
        'max_features': ['sqrt', 'log2', 0.3, 0.7],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_leaf_nodes': [None, 50, 100],
        'bootstrap': [True, False]}

        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        self.search = RandomizedSearchCV(
            rf, param_distributions=param_grid,
            n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
        )

        self.search.fit(self.X_train, self.y_train)
        self.model = self.search.best_estimator_
        self.best_params_ = self.search.best_params_
        self.grid_search_results_ = pd.DataFrame(self.search.cv_results_)


        # print("Best parameters found:", self.best_params_)

    def evaluate_model(self, save_path="model_eval/predictive_model_evaluation.csv"):
        print("Evaluating model...")
        y_pred = self.model.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        class_counts = {str(k): int(v) for k, v in Counter(y_pred).items()}
        top_class = max(class_counts.items(), key=lambda x: x[1])[0]

        results = {
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'n_classes': len(set(self.y_test)),
            'top_predicted_class': top_class,
            'class_distribution': json.dumps(class_counts),
            **self.best_params_  # n_estimators, max_depth
        }

        print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"Class Distribution: {class_counts}")

        # save CSV
        results_df = pd.DataFrame([results])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            results_df.to_csv(save_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(save_path, index=False)



    def save_model(self):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(self.model, self.model_save_path)
        joblib.dump(self.label_encoder, 'models/label_encoder.joblib')
        print(f"Model saved at {self.model_save_path}")

    def run(self, target='region', country_metadata_path='scoring/country-data.csv'):
        self.join_country_data(country_metadata_path)
        self.analyze_target_correlations()
        self.prepare_features()
        self.set_target(target)
        self.split_and_resample()
        self.train_model()
        self.evaluate_model()
        self.save_model()
