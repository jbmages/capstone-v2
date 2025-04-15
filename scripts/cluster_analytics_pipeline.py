import pandas as pd
import json
import time
import os

class ClusterAnalyticsPipeline:

    def __init__(self, data, scoring):
        self.data = data
        self.scoring = scoring

        return

    def assign_question(self):
        """Assign actual question names to the first 50 columns in self.data"""
        # Create a mapping from id to question
        id_to_question = dict(zip(self.scoring['id'], self.scoring['trait']))

        # Get the first 50 column names
        original_cols = list(self.data.columns[:50])

        # Build new column names in the format "EXT9: I am the life of the party"
        new_cols = [f"{col}: {id_to_question[col]}" if col in id_to_question else col for col in original_cols]

        # Assign new column names to the DataFrame
        self.data.columns = new_cols + list(self.data.columns[50:])

        print("question names assigned to data")
        return


    def to_json(self, sample_frac=0.1, max_rows=10000):
        start_time = time.time()

        # Select relevant columns (first 50 + last 2 for clustering info)
        selected_columns = list(self.data.columns[:50]) + ["KMeans_Cluster", "GMM_Cluster"]
        df_subset = self.data[selected_columns]

        # Randomly sample 10% of the data, but no more than max_rows
        num_rows = min(int(len(df_subset) * sample_frac), max_rows)
        df_sample = df_subset.sample(n=num_rows, random_state=42)  # Fix seed for reproducibility

        df_sample["KMeans_Cluster"] = pd.to_numeric(df_sample["KMeans_Cluster"], errors="coerce")
        df_sample["GMM_Cluster"] = pd.to_numeric(df_sample["GMM_Cluster"], errors="coerce")

        # Convert to JSON format
        print(f"Converting {num_rows}/{len(df_subset)} rows to JSON...")
        data_json = df_sample.to_dict(orient="records")

        # Define output path and ensure the directory exists
        output_dir = "dashboard/dash-data"
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists before saving
        output_json = os.path.join(output_dir, "cluster_data.json")

        # Save JSON file
        with open(output_json, "w") as f:
            json.dump(data_json, f, indent=2)

        # Get file size
        file_size = os.path.getsize(output_json) / (1024 * 1024)  # Convert to MB
        end_time = time.time()

        print(f"JSON saved at {output_json} ({file_size:.2f} MB)")
        print(f"Process took {end_time - start_time:.2f} seconds")

        return output_json  # Return file path if needed
