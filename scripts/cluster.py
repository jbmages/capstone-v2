import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import scripts.utils as utils
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import json
import os
import time


class ClusteringModel:

    def __init__(self, dataset: pd.DataFrame):
        """Initialize clustering model object"""
        print('initiating modeling infrastructure')
        self.data = dataset

        # Retrieve scoring table first
        self.scoring_table = utils.retrieve_excel('scoring/scoring.xlsx')
        print('scoring table correctly read in')
        # Prepare data segments
        self.survey_data, self.time_data, self.score_data = self.prep_data()

        print('data is prepped')
        # Placeholder for model
        self.model = None

    def prep_data(self):
        """Prepare data segments for clustering"""

        print('preparing data for modeling')
        survey_answer_cols = self.scoring_table['id'].tolist()
        time_cols = [col + '_E' for col in survey_answer_cols]
        score_cols = ['O score', 'C score', 'E score', 'A score', 'N score']

        # Ensure selected columns exist in dataset before slicing
        survey_data = self.data[survey_answer_cols] if all(
            col in self.data.columns for col in survey_answer_cols) else None

        time_data = self.data[time_cols] if all(col in self.data.columns for col in time_cols) else None
        # scale data
        score_data = self.data[score_cols] if all(col in self.data.columns for col in score_cols) else None

        print('scaling data')
        scaler = StandardScaler()

        return scaler.fit_transform(survey_data), scaler.fit_transform(time_data)  , scaler.fit_transform(score_data)

    def assign_clusters(self):
        """Assign clusters to data """

        # Apply MiniBatch KMeans (6 clusters)
        kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=100)
        self.data['KMeans_Cluster'] = kmeans.fit_predict(self.score_data)

        # Apply Gaussian Mixture Model (5 clusters)
        gmm = GaussianMixture(n_components=5, random_state=42)
        self.data['GMM_Cluster'] = gmm.fit_predict(self.score_data)

        # Save the updated dataset with cluster assignments
        self.data.to_csv('data/data_with_clusters.csv', index=False)

        print("Clustering complete. Data saved as 'cleaned_data_with_clusters.csv'.")

    def csv_to_json(self):
        """Send cluster data to dashboard data folder"""
        start_time = time.time()

        # Select the first 50 columns + the last 2 (cluster assignments)
        selected_columns = list(self.data.columns[:50]) + ["KMeans_Cluster", "GMM_Cluster"]
        df_subset = self.data[selected_columns]

        # Convert to JSON format
        print("Converting to JSON...")
        data_json = df_subset.to_dict(orient="records")

        # Define output path and ensure the directory exists
        output_dir = "dashboard/data"
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists before saving
        output_json = os.path.join(output_dir, "cluster_data.json")

        # Save JSON file
        with open(output_json, "w") as f:
            json.dump(data_json, f, indent=2)

        # Get file size
        file_size = os.path.getsize(output_json) / (1024 * 1024)  # Convert to MB
        end_time = time.time()

        print(f"✅ JSON saved at {output_json} ({file_size:.2f} MB)")
        print(f"⏳ Process took {end_time - start_time:.2f} seconds")

        return output_json  # Return file path if needed