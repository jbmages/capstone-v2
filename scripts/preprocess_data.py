import pandas as pd
import os
import time
from tqdm import tqdm


class DataPreprocessor:
    def __init__(self, dataset_path: str, output_path: str = './data/cleaned_data_v2.csv'):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.data = pd.DataFrame()

    def process_data(self):
        """Run the complete data pipeline"""
        start_time = time.time()
        self.clean_data()
        self.normalize_data()
        self.trait_scores()
        self.remove_nan()
        self.save_data()
        total_time = time.time() - start_time
        print(f"Data processing complete. Total time taken: {self.format_time(total_time)}")

    def clean_data(self):
        """Clean data into desired format"""
        print("Cleaning data...")
        start_time = time.time()

        # Fix the dataset path if the script is in another folder
        dataset_path = os.path.join(self.dataset_path, 'data-final.csv')
        self.data = pd.read_csv(dataset_path)

        rows = []
        cols = self.data.columns[0].split('\t')
        for index in tqdm(range(len(self.data)), desc="Cleaning rows", unit="row"):
            value = self.data.iloc[index, 0]
            row = value.split('\t')
            rows.append(row)

        self.data = pd.DataFrame(rows, columns=cols)
        clean_time = time.time() - start_time
        print(f"Data cleaned. Columns: {', '.join(self.data.columns)}")
        print(f"Cleaning data took: {self.format_time(clean_time)}")

    def normalize_data(self):
        """Normalize data types and filter for 1 user per entry"""
        print("Normalizing data...")
        start_time = time.time()

        non_numeric_cols = ['dateload', 'country']
        numeric_cols = [col for col in self.data.columns if col not in non_numeric_cols]

        # Convert only the selected columns to numeric
        self.data[numeric_cols] = self.data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        normalize_time = time.time() - start_time
        print(f"Data normalized in {self.format_time(normalize_time)}.")

    def trait_scores(self):
        """Calculate cumulative trait scores for survey entries"""
        print("Calculating trait scores...")
        start_time = time.time()

        score_cols = ["O score", "C score", "E score", "A score", "N score"]
        self.data[score_cols] = 0  # Initialize the score columns

        # Mapping traits to corresponding column prefixes
        mapping = {'O score': 'OPN', 'C score': 'CSN', 'E score': 'EXT', 'A score': 'AGR', 'N score': 'EST'}

        survey_cols = self.data.columns[:50]  # Limit to first 50 columns

        # Vectorized calculation of trait scores
        for trait, code in tqdm(mapping.items(), desc="Calculating trait scores", unit="trait"):
            relevant_cols = [col for col in survey_cols if col.startswith(code)]
            # Sum over relevant columns and assign to the trait score
            self.data[trait] = self.data[relevant_cols].sum(axis=1)

        trait_time = time.time() - start_time
        print(f"Trait scores calculated in {self.format_time(trait_time)}.")

    def remove_nan(self):
        """Remove rows with NaN values in survey answers"""
        print("Removing rows with NaN values in survey answers...")
        print(self.data)
        start_time = time.time()

        # Select the first 50 columns as the subset of interest
        survey_answer_cols = self.data.columns[:50]

        # Drop rows with NaN values in these selected columns
        self.data = self.data.dropna(subset=survey_answer_cols)

        remove_time = time.time() - start_time
        print(f"Removed {len(self.data)} rows with missing survey answers.")
        print(f"Removing NaN values took: {self.format_time(remove_time)}")

    def save_data(self):
        """Save the processed data to a CSV file"""
        print(f"Saving cleaned data to {self.output_path}...")
        start_time = time.time()

        self.data.to_csv(self.output_path, index=False)

        save_time = time.time() - start_time
        print(f"Data saved to {self.output_path} in {self.format_time(save_time)}")

    def format_time(self, seconds: float):
        """Helper function to format time"""
        m, s = divmod(seconds, 60)
        return f"{int(m)} min {int(s)} sec"

