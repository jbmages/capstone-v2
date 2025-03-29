import os
import logging
import traceback
from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
from scripts.cluster import ClusteringModel
import scripts.utils as utils

# Configure logging
logging.basicConfig(
    filename="logs/workflow.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Dataset URL (Google Drive or any other source)
GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id=1FzmqQDt_Amv0Gga4Rvo5iDrHuHBFGgrP'

# Define paths but check their existence dynamically
RAW_DATA_PATH = 'data/raw_data.csv'
CLEANED_DATA_PATH = 'data/cleaned_data_v2.csv'


class FullWorkflow:
    """
    Full data science pipeline for clustering survey responses into personality profiles.
    Includes data downloading, preprocessing, and clustering.
    """

    def __init__(self, dataset_url: str, skip_download=True, skip_preprocessing=True):
        """
        Initializes the full workflow.

        :param dataset_url: URL for downloading the dataset
        :param skip_download: If True, skips the data download step
        :param skip_preprocessing: If True, skips data preprocessing
        """
        self.dataset_url = dataset_url
        self.dataset = None

        try:
            # Only execute steps if files don’t already exist
            if not skip_download or not os.path.exists(RAW_DATA_PATH):
                self.data_download()

            if not skip_preprocessing or not os.path.exists(CLEANED_DATA_PATH):
                self.data_preprocessing()

            self.load_dataset()
            self.clustering()
        except Exception as e:
            logging.error("Critical error in workflow initialization: %s", e)
            traceback.print_exc()

    def data_download(self):
        """Handles downloading of dataset from external sources if not already downloaded."""
        try:
            if os.path.exists(RAW_DATA_PATH):
                logging.info("Raw data already exists, skipping download.")
                return

            logging.info("Starting data download...")
            DataDownloader(url=self.dataset_url)
            #downloader.download()  # Ensure `download()` exists in DataDownloader
            logging.info("Data downloaded and saved at: %s", RAW_DATA_PATH)
        except Exception as e:
            logging.error("Data download failed: %s", e)
            traceback.print_exc()

    def data_preprocessing(self):
        """Handles data preprocessing pipeline if not already processed."""
        try:
            if os.path.exists(CLEANED_DATA_PATH):
                logging.info("Cleaned data already exists, skipping preprocessing.")
                return

            logging.info("Starting data preprocessing...")
            preprocessor = DataPreprocessor(dataset_path=RAW_DATA_PATH)
            preprocessor.process_data()
            logging.info("Data preprocessing complete. Cleaned dataset saved at: %s", CLEANED_DATA_PATH)
        except Exception as e:
            logging.error("Data preprocessing failed: %s", e)
            traceback.print_exc()

    def load_dataset(self):
        """Loads the processed dataset from file if available."""
        try:
            if os.path.exists(CLEANED_DATA_PATH):
                self.dataset = utils.retrieve_data(CLEANED_DATA_PATH)
                logging.info("Dataset successfully loaded from: %s", CLEANED_DATA_PATH)
            else:
                raise FileNotFoundError(f"Dataset not found at {CLEANED_DATA_PATH}")
        except Exception as e:
            logging.error("Failed to load dataset: %s", e)
            traceback.print_exc()

    def clustering(self):
        """Performs clustering on survey data."""
        try:
            if self.dataset is not None:
                logging.info("Starting clustering process...")
                clustering_model = ClusteringModel(dataset=self.dataset)
                clustering_model.assign_clusters()
                logging.info("Clustering completed successfully.")
            else:
                logging.error("Clustering aborted: Dataset is not loaded.")
        except Exception as e:
            logging.error("Error in clustering process: %s", e)
            traceback.print_exc()


# Example usage
if __name__ == "__main__":
    workflow = FullWorkflow(
        dataset_url=GOOGLE_DRIVE_URL,
        skip_download=True,
        skip_preprocessing=True
    )
