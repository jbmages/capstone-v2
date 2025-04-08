import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler

def get_us_region(lat, lon):
    if (36.5 <= lat <= 47.5) and (-82.0 <= lon <= -66.9):
        return "Northeast"
    elif (36.5 <= lat <= 49.0) and (-104.0 <= lon <= -82.0):
        return "Midwest"
    elif (24.5 <= lat <= 36.5) and (-105.0 <= lon <= -75.0):
        return "South"
    elif (31.0 <= lat <= 49.0) and (-125.0 <= lon <= -102.0):
        return "West"
    else:
        return "Unknown/Outside U.S."

class PredictiveModel:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the predictive model with raw data.
        """
        self.data = data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.model = None  # Will be Random Forest model

    def preprocess(self):
        """
        Preprocess data:
         - Filter to US respondents.
         - Compute region using get_us_region.
         - Drop unneeded columns.
         - Encode target labels.
         - Scale features.
        """
        print("Preprocessing data...")
        # Filter to US respondents and assign region using the helper function.
        usdata = self.data[self.data["country"] == "US"].copy()
        usdata["region"] = usdata.apply(
            lambda row: get_us_region(row["lat_appx_lots_of_err"],
                                      row["long_appx_lots_of_err"]), axis=1
        )
        
        # Drop target columns (latitude, longitude, country, region) from features
        self.X = usdata.drop(columns=['lat_appx_lots_of_err', 'long_appx_lots_of_err', 'country', 'region'])
        self.y = usdata['region']
        
        # Encode target labels
        y_encoded = self.le.fit_transform(self.y)
        print("Encoded region labels:")
        print(dict(zip(self.le.classes_, range(len(self.le.classes_)))))
        self.y = y_encoded
        
        # Standardize features
        self.X = self.scaler.fit_transform(self.X)

    def split_and_undersample(self):
        """
        Split data into train and test sets, then undersample the largest class to the level of the second largest.
        """
        print("Splitting data and undersampling...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Compute class counts
        class_counts = dict(Counter(self.y_train))
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        # Undersample the largest class to match the size of the second largest
        undersampling_strategy = {sorted_counts[0][0]: sorted_counts[1][1]}
        print(f"Undersampling strategy: {undersampling_strategy}")
        
        undersampler = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)
        self.X_train_resampled, self.y_train_resampled = undersampler.fit_resample(self.X_train, self.y_train)

    def train_model(self):
        """
        Train a Random Forest classifier using the best-found hyperparameters.
        These parameters are fixed and hardcoded: 
            max_depth=30, max_features='sqrt', max_leaf_nodes=100,
            min_samples_leaf=2, min_samples_split=7, n_estimators=188.
        """
        print("Training model with fixed best parameters...")
        self.model = RandomForestClassifier(
            n_estimators=188,
            max_depth=30,
            max_features='sqrt',
            max_leaf_nodes=100,
            min_samples_leaf=2,
            min_samples_split=7,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(self.X_train_resampled, self.y_train_resampled)

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set and print accuracy and a classification report.
        """
        print("Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print("\nTest Accuracy: {:.4f}".format(acc))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.le.classes_))

    def run(self):
        """
        Run the full predictive modeling pipeline.
        """
        self.preprocess()
        self.split_and_undersample()
        self.train_model()
        self.evaluate_model()