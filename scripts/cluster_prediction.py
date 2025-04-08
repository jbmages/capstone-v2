import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm.notebook import tqdm
import time
import os
import joblib
class ClusterPredictor:

    def __init__(self, data, scoring):
        # initialize clustering predictive model
        self.data = data
        self.scoring = scoring
        self.data_prep()

    def data_prep(self):
        """ Remove maximum cluster value
        this cluster is irreleavant and represents an insignificant proportion of data"""
        id_to_question = dict(zip(self.scoring['id'], self.scoring['trait']))
        original_cols = list(self.data.columns[:50])
        new_cols = [f"{col}: {id_to_question[col]}" if col in id_to_question else col for col in original_cols]
        self.data.columns = new_cols + list(self.data.columns[50:])

        # remove max cluster value
        # Find the maximum cluster value for each clustering algorithm
        kmeans_max = self.data['KMeans_Cluster'].max()
        gmm_max = self.data['GMM_Cluster'].max()

        # Filter out rows where either cluster assignment is the max value
        original_count = len(self.data)
        self.filtered_data = self.data[(self.data['KMeans_Cluster'] != kmeans_max) & (self.data['GMM_Cluster'] != gmm_max)]
        removed_count = original_count - len(self.filtered_data)

        # Print the number of rows removed and remaining
        print(f"Original dataset size: {original_count} rows")
        print(f"Rows removed: {removed_count} rows ({removed_count / original_count:.2%} of data)")
        print(f"Remaining dataset size: {len(self.filtered_data)} rows")







