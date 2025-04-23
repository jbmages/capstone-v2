import pandas as pd
import json
import time
import os
pd.set_option('display.max_columns', None)

class ClusterAnalyticsPipeline:

    def __init__(self, data, scoring):
        self.data = data
        self.scoring = scoring

        self.assign_question()
        self.to_json()
        return

    def assign_question(self):
        """Assign actual question names to the first 50 columns in self.data"""
        id_to_question = dict(zip(self.scoring['id'], self.scoring['trait']))

        original_cols = list(self.data.columns[:55])

        new_cols = [f"{col}: {id_to_question[col]}" if col in id_to_question else col for col in original_cols]

        self.data.columns = new_cols + list(self.data.columns[55:])

        print("question names assigned to data")
        print(self.data.head(15))
        return


    def to_json(self, sample_frac=0.1, max_rows=10000):

        cluster_cols = ['gmm_4_both_cluster', 'gmm_3_both_cluster',
                        'gmm_3_survey_cluster', 'gmm_5_survey_cluster']

        start_time = time.time()

        # Select relevant columns (first 50 + last 2 for clustering info)
        selected_columns = list(self.data.columns[5:55]) + cluster_cols
        df_subset = self.data[selected_columns]

        # Randomly sample 10% of the data, but no more than max_rows
        num_rows = max_rows
        df_sample = df_subset.sample(n=num_rows, random_state=42)  # Fix seed for reproducibility

        for cluster in cluster_cols:
            df_sample[cluster] = pd.to_numeric(df_sample[cluster], errors="coerce")

        # Convert to JSON format
        print(f"Converting {num_rows}/{len(df_subset)} rows to JSON...")
        data_json = df_sample.to_dict(orient="records")

        # Define output path and ensure the directory exists
        output_dir = "dashboard/dash-data"
        os.makedirs(output_dir, exist_ok=True)
        output_json = os.path.join(output_dir, "cluster_data_v2.json")

        # Save JSON file
        with open(output_json, "w") as f:
            json.dump(data_json, f, indent=2)

        # Get file size
        file_size = os.path.getsize(output_json) / (1024 * 1024)  # Convert to MB
        end_time = time.time()

        print(f"JSON saved at {output_json} ({file_size:.2f} MB)")
        print(f"Process took {end_time - start_time:.2f} seconds")

        return output_json
