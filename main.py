from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
import os
import traceback

GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id=1FzmqQDt_Amv0Gga4Rvo5iDrHuHBFGgrP'

class FullWorkflow:
    def __init__(self, dataset_url: str):
        self.dataset_url = dataset_url
        self.run_workflow()

    def run_workflow(self):
        """Run the entire data pipeline."""
        try:
            print("Starting the data download...")
            #DataDownloader(url=self.dataset_url)
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
        except Exception as e:
            print("An error occurred during data preprocessing:")
            traceback.print_exc()  # Print full traceback


# Example usage
if __name__ == "__main__":
    workflow = FullWorkflow(GOOGLE_DRIVE_URL)
