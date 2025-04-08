import os
import traceback
from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
from scripts.cluster import ClusteringModel
from scripts.cluster_prediction import ClusterPredictor
from scripts.clusering_v2 import ImprovedClusteringModel
import scripts.utils as utils


GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id=1FzmqQDt_Amv0Gga4Rvo5iDrHuHBFGgrP'

# Define paths
RAW_DATA_PATH = 'data/data-final.csv'
CLEANED_DATA_PATH = 'data/cleaned_data_v2.csv'
CLUSTERED_DATA_PATH = 'data/data_with_clusters.csv'


class FullWorkflow:
    """
    Full data science pipeline for clustering survey responses into personality profiles.
    Includes data downloading, preprocessing, and clustering.
    """

    def __init__(self, dataset_url: str, skip_download=True, skip_preprocessing=True,
                 skip_clustering=False, skip_predictive=True, use_clustering_v2=True):
        """
        Initializes the full workflow.

        :param dataset_url: URL for downloading the dataset
        :param skip_download: If True, skips the data download step
        :param skip_preprocessing: If True, skips data preprocessing
        """
        self.dataset_url = dataset_url
        self.dataset = 0

        self.scoring = utils.retrieve_excel('scoring/scoring.xlsx')
        print('scoring table correctly read in')

        try:
            if not skip_download or not os.path.exists(RAW_DATA_PATH):
                self.data_download()

            if not skip_preprocessing or not os.path.exists(CLEANED_DATA_PATH):
                self.data_preprocessing()

            self.dataset = self.load_dataset()

            if not skip_clustering:
                if use_clustering_v2:
                    self.clustering_v2()
                else:
                    self.clustering()

            if not skip_predictive:
                self.cluster_prediction()


        except Exception:
            traceback.print_exc()

    def data_download(self):
        """Handles downloading of dataset from external sources if not already downloaded."""
        try:
            if os.path.exists(RAW_DATA_PATH):
                print("Raw data already exists, skipping download.")
                return

            print("Starting data download...")
            DataDownloader(url=self.dataset_url)
            print(f"Data downloaded and saved at: {RAW_DATA_PATH}")
        except Exception:
            print("Data download failed:")
            traceback.print_exc()

    def data_preprocessing(self):
        """Handles data preprocessing pipeline if not already processed."""
        try:
            #if os.path.exists(CLEANED_DATA_PATH):
                #print("Cleaned data already exists, skipping preprocessing.")
                #return

            print("Starting data preprocessing...")
            preprocessor = DataPreprocessor(dataset_path=RAW_DATA_PATH)
            preprocessor.process_data()
            print(f"Data preprocessing complete. Cleaned dataset saved at: {CLEANED_DATA_PATH}")
        except Exception:
            print("Data preprocessing failed:")
            traceback.print_exc()

    def load_dataset(self):
        """Loads the processed dataset from file if available."""
        try:
            if os.path.exists(CLEANED_DATA_PATH):
                self.dataset = utils.retrieve_data(CLEANED_DATA_PATH)
                print(f"Dataset successfully loaded from: {CLEANED_DATA_PATH}")
                return self.dataset
            else:
                raise FileNotFoundError(f"Dataset not found at {CLEANED_DATA_PATH}")
        except Exception:
            print("Failed to load dataset:")
            traceback.print_exc()

    def clustering(self):
        """Performs clustering on survey data."""
        try:
            if self.dataset is not None:
                print("Starting clustering process...")
                clustering_model = ClusteringModel(dataset=self.dataset, scoring=self.scoring)
                clustering_model.assign_clusters()
                clustering_model.assign_question()
                clustering_model.csv_to_json()

                print("Clustering completed successfully.")
            else:
                print("Clustering aborted: Dataset is not loaded.")
        except Exception:
            print("Error in clustering process:")
            traceback.print_exc()

    def clustering_v2(self, mode = 'score'):
        """ Implements updated clustering workflow """
        try:
            if self.dataset is not None:
                print("Starting clustering process...")
                cluster_model = ImprovedClusteringModel(dataset=self.dataset,
                                                        scoring=self.scoring,mode=mode)
                results = cluster_model.assign_clusters()
                cluster_model.visualize_results(results)
                cluster_model.csv_to_json()
                #json_path = cluster_model.csv_to_json()

                #print(f"Workflow complete for mode: '{mode}'. JSON output at: {json_path}\n")
            else:
                print("Clustering aborted: Dataset is not loaded.")
        except Exception:
            print("Error in clustering process:")
            traceback.print_exc()

    def cluster_prediction(self):
        """ Runs cluster prediction algorithm """
        try:
            if os.path.exists(CLUSTERED_DATA_PATH):

                predictor = ClusterPredictor(data=utils.retrieve_data(CLUSTERED_DATA_PATH),
                                                   scoring=self.scoring)
                predictor.train_all()

                # Plot accuracies
                predictor.plot_model_accuracies()

                # Optional: Save models
                predictor.save_models()

                # Optional: Stepwise analysis (logistic or random_forest)
                predictor.stepwise_feature_analysis(top_n=10, model_type='logistic')

        except Exception:
            print('you suck. cluster prediction failed bruh...')
            traceback.print_exc()



if __name__ == "__main__":
    workflow = FullWorkflow(
        dataset_url=GOOGLE_DRIVE_URL,
        skip_download=True,
        skip_preprocessing=True,
        skip_clustering=False,
        skip_predictive=True,
        use_clustering_v2=True
    )
