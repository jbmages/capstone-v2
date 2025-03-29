from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
import scripts.utils as utils
from scripts.cluster import ClusteringModel

import os
import traceback

GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id=1FzmqQDt_Amv0Gga4Rvo5iDrHuHBFGgrP'

class FullWorkflow:
    def __init__(self, dataset_url: str):
        self.dataset_url = dataset_url
        self.dataset = self.data_setup()

    def data_setup(self):
        """Run the entire data pipeline."""

        try:
            print("Starting the data download...")
            DataDownloader(url=self.dataset_url)
            print("Data download and processing complete.")

        except Exception as e:
            print("An error occurred during data download:")
            traceback.print_exc()  # Print full traceback
            return  # If download fails, stop the workflow

        try:
            print('Pre-processing data...')
            # Ensure the correct path is passed to the preprocessor
            preprocessor = DataPreprocessor(dataset_path='./data')
            preprocessor.process_data()  # Call process_data to trigger the full pipeline
            print("Data preprocessing complete.")
            print("Cleaned and ready dataset saved to data folder")
        except Exception as e:
            print("An error occurred during data preprocessing:")
            traceback.print_exc()  # Print full traceback

        return utils.retrieve_data('../data/cleaned_data_v2.csv')

    def clustering(self):
        """ Cluster survey score data into personalities """
        clustering_model = ClusteringModel(dataset = self.dataset)
        print(clustering_model)


# Example usage
if __name__ == "__main__":
    workflow = FullWorkflow(GOOGLE_DRIVE_URL)
