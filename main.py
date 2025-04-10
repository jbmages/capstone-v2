import os
import traceback
from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
from scripts.cluster import ClusteringModel
from scripts.cluster_prediction import ClusterPredictor
import scripts.utils as utils
from scripts.region_prediction import PredictiveModel

# v2 imports
from scripts.clustering import ClusteringWorkflow

GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id=1FzmqQDt_Amv0Gga4Rvo5iDrHuHBFGgrP'

# Define paths
RAW_DATA_PATH = 'data/data-final.csv'
CLEANED_DATA_PATH = 'data/cleaned_data_v2.csv'
CLUSTERED_DATA_PATH = 'data/data_with_clusters.csv'


class FullWorkflow:
    """
    Full project pipeline for clustering survey responses into personality profiles.
    Includes data downloading, preprocessing, and clustering.
    """

    def __init__(self, dataset_url: str, skip_download=True, skip_preprocessing=True,
                 skip_clustering=False, skip_predictive=True, use_clustering_v2=True,
                 use_prediction_v2=True, skip_region_predictive=False):
        """
        Initializes the full workflow.

        :param dataset_url: URL for downloading the dataset
        :param skip_download: If True, skips the data download step
        :param skip_preprocessing: If True, skips data preprocessing
        """
        self.dataset_url = dataset_url
        self.dataset = 0
        self.scoring = utils.retrieve_excel('scoring/scoring.xlsx')

        try:

            ### DATA DOWNLOAD
            if not skip_download or not os.path.exists(RAW_DATA_PATH):
                print('downloading data..')
                self.data_download()

            ### DATA PRE-PROCESSING
            if not skip_preprocessing or not os.path.exists(CLEANED_DATA_PATH):
                print('pre-processing data..')
                self.data_preprocessing()
                print(f"Data preprocessing complete. Cleaned dataset saved at: {CLEANED_DATA_PATH}")

            ### DATA RETRIEVAL
            print('loading pre-processed data..')
            self.dataset = self.load_dataset()

            ### CLUSTERING
            if not skip_clustering:
                if use_clustering_v2:
                    print('initiating updated clustering workflow')
                    self.clustering_v2()
                else:
                    print('initiating clustering workflow')
                    self.clustering()

            ### CLUSTER PREDICTION
            if not skip_predictive:
                if use_prediction_v2:
                    self.cluster_prediction_v2()
                else:
                    self.cluster_prediction()

            ### REGION PREDICTION
            if not skip_region_predictive:
                self.predictive_modeling()

        except Exception:
            traceback.print_exc()

    def data_download(self):
        """Handles downloading of dataset from external sources if not already downloaded."""
        try:
            if os.path.exists(RAW_DATA_PATH):
                print("Raw data already exists, skipping download.")
                return

            DataDownloader(url=self.dataset_url)
            print(f"Data downloaded and saved at: {RAW_DATA_PATH}")

        except Exception:
            print("Data download failed:")
            traceback.print_exc()

    def data_preprocessing(self):
        """Handles data preprocessing pipeline if not already processed."""
        try:
            if os.path.exists(CLEANED_DATA_PATH):
                print("Cleaned data already exists, skipping preprocessing.")
                return

            preprocessor = DataPreprocessor(dataset_path=RAW_DATA_PATH)
            preprocessor.process_data()

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

    def clustering_v2(self):
        """ Implements updated clustering workflow """

        clustering_data = ['scores', 'time']

        model_space = {
            'kmeans': {
                'class': 'KMeans',
                'params': {
                    'n_clusters': [3, 4, 5, 6, 7]
                }
            }
        }

        """
            'gmm': {
                'class': GMMModel,
                'params': {
                    'n_components': [3, 4, 5, 6, 7],
                    'covariance_type': ['full', 'tied']
                }
            },
            'dbscan': {
                'class': DBSCANModel,
                'params': {
                    'eps': [0.5, 1.0, 1.5],
                    'min_samples': [5, 10]
                }
            },
            'hierarchical': {
                'class': HierarchicalModel,
                'params': {
                    'n_clusters': [3, 4, 5],
                    'linkage': ['ward', 'complete']
                }
        """

        try:
            if self.dataset is not None:
                print("Starting model evaluation...")
                print("model space:", model_space)

                CWF = ClusteringWorkflow(self.dataset, self.scoring,
                                         cluster_data=clustering_data)
                CWF.grid_search(model_space=model_space)

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

    def cluster_prediction_v2(self):
        """ Runs cluster prediction algorithm """
        try:
            if os.path.exists(CLUSTERED_DATA_PATH):
                # run clustering workflow
                # set dataset subset amount and max time length
                #
                pass

        except Exception:
            traceback.print_exc()

    def predictive_modeling(self):
        "Runs the predictive model"""
        try:
            if self.dataset is not None:
                print("Starting predictive model...")
                model = PredictiveModel(self.dataset)
                model.run()
                print("Predictive model completed.")
            else:
                print("Predictive modeling didn't work: Dataset is not loaded.")
        except Exception:
            print("Error in predictive modeling:")
            traceback.print_exc()


if __name__ == "__main__":
    print('Do we have defined personalities? Lets find out..')
    print('Initiating workflow..')
    workflow = FullWorkflow(
        dataset_url=GOOGLE_DRIVE_URL,
        skip_download=True,
        skip_preprocessing=True,
        skip_clustering=False,
        skip_predictive=True,
        use_clustering_v2=True,
        use_prediction_v2=True,
        skip_region_predictive=True

    )
