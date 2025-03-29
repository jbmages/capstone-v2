import traceback
import scripts.utils as utils
from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
from scripts.cluster import ClusteringModel

class FullWorkflow:
    def __init__(self, dataset_url: str):
        self.dataset_url = dataset_url
        self.dataset = None

    def data_setup(self):
        """Runs the full data pipeline: download, preprocess, and return cleaned data."""
        try:
            print("Starting the data download...")
            DataDownloader(url=self.dataset_url)
            print("Data downloaded successfully.")
        except Exception as e:
            print("Error during data download:")
            traceback.print_exc()
            return None  

        try:
            print('Pre-processing data...')
            preprocessor = DataPreprocessor(dataset_path='./data')
            preprocessor.process_data()
            print("Data preprocessing complete.")
        except Exception as e:
            print("Error during preprocessing:")
            traceback.print_exc()
            return None  

        self.dataset = utils.retrieve_data('data/cleaned_data_v2.csv')
        return self.dataset

    def clustering(self):
        """Runs clustering on the cleaned dataset."""
        if self.dataset is None:
            print("No dataset available. Run data_setup() first.")
            return
        clustering_model = ClusteringModel(dataset=self.dataset)
        print("Clustering completed successfully.")
