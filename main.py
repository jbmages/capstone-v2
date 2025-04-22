import os
import traceback
from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
from scripts.clustering import ClusteringWorkflow
from scripts.cluster_analytics_pipeline import ClusterAnalyticsPipeline


from scripts.cluster_prediction import PredictionWorkflow
import scripts.utils as utils
from scripts.location_prediction import ImprovedPredictiveModel
import pandas as pd
import time


GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id=1FzmqQDt_Amv0Gga4Rvo5iDrHuHBFGgrP'

# Define paths
RAW_DATA_PATH = 'data/data-final.csv'
CLEANED_DATA_PATH = 'data/cleaned_data_v2.csv'
CLUSTERED_DATA_PATH = 'data/data_with_clusters.csv'
CLUSTERED_DATA_PATH_V2 = 'scripts/cluster_eval/combined_clusters.csv'

class FullWorkflow:
    """
    Full project pipeline for clustering survey responses into personality profiles.
    Includes data downloading, preprocessing, and clustering.
    """

    def __init__(self, dataset_url: str, skip_download=True, skip_preprocessing=True,
                 skip_clustering=True, skip_predictive=False,
                 use_prediction_v2=False, skip_region_predictive=True,
                 skip_cluster_analytics=True):
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



            if not skip_cluster_analytics:
                self.clustering_analytics()


            ### CLUSTER PREDICTION®

            self.clustering()

            if not skip_predictive:

                 self.cluster_prediction()

            ### REGION PREDICTION
            if not skip_region_predictive:
                self.predictive_modeling(target='country', sample_frac=0.09)

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
        """ Comprehensive clustering benchmark """

        try:
            print("Starting full clustering grid search...")
            # Targeted combinations based on prior findings
            cluster_data_variants = [

                (['scores'], True)

            ]

            model_space = {
                'GMM': {
                    'class': 'GMMHomegrown',
                    'params': {
                        'n_components': [3, 4, 5, 6],
                        'max_iter': [50, 100],
                        'tol': [1e-3, 1e-4],
                        'n_factors': [5]
                    }
                },
                'DBScan': {
                    'class': 'DBScanHomegrown',
                    'params': {
                        'eps': [0.5, 1.0, 1.5],
                        'min_samples': [5, 10, 15],
                        'n_factors': [5]
                    }
                },
                'KMeans': {
                    'class': 'KMeansHomegrown',
                    'params': {
                        'n_clusters': [3, 4, 5, 6],
                        'max_iter': [50, 100],
                        'n_init': [10],
                        'n_factors': [5]
                    }
                }
            }

            all_results = []

            for cluster_data, fa_flag in cluster_data_variants:
                for model_key in model_space.keys():
                    subspace = {model_key: model_space[model_key]}

                    print(f"\n[CONFIG] Data: {cluster_data} | FA: {fa_flag} | Model: {model_key}")

                    workflow = ClusteringWorkflow(
                        data=self.dataset,
                        scoring_table=self.scoring,
                        cluster_data=cluster_data,
                        model_space=subspace,
                        apply_factor_analysis=fa_flag,
                        n_factors=5,
                        max_time=120,
                        save_results=True  #
                    )

                    results = workflow.grid_search()
                    all_results.extend(results)

            # Save full results
            df = pd.DataFrame(all_results)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_path = f"model_eval/gridsearch_all_models_{timestamp}.csv"
            os.makedirs("model_eval", exist_ok=True)
            df.to_csv(out_path, index=False)


        except Exception as e:
            import traceback
            print("[ERROR] Exception occurred during clustering:")
            traceback.print_exc()

    def clustering_analytics(self):
        try:
            if os.path.exists(CLUSTERED_DATA_PATH_V2):
                print('initiating clustering analytics pipeline for dashboarding')
                d = self.dataset = utils.retrieve_data(CLUSTERED_DATA_PATH_V2)
                p = ClusterAnalyticsPipeline(data=d, scoring=self.scoring)
            else:
                print('error: path', CLUSTERED_DATA_PATH_V2, 'not found')
                return
        except Exception as e:
            import traceback
            print("[ERROR] Exception occurred in clustering analytics pipeline:")
            traceback.print_exc()
    def cluster_prediction_v2(self):
        """ Runs cluster prediction algorithm """

        params = {
            'LogisticRegression': {
                'class': 'LogisticRegression',
                'params': {
                    'penalty': ['l1', 'l2'],
                    'c': [0.01, 0.1, 1.0, 10],
                    'solver': ['saga'],  # only solver that supports all penalties
                    'max_iter': [200, 500]
                }
            },
            'SVM': {
                'class': 'SVM',
                'params': {
                    'c': [0.01, 0.1, 1.0, 10],
                    'loss': ['hinge', 'squared_hinge'],
                    'max_iter': [500, 1000]
                }
            },
            'NeuralNet': {
                'class': 'NeuralNet',
                'params': {
                    'hidden_layer_sizes': [(64,), (128,), (128, 64)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'solver': ['adam', 'sgd'],
                    'max_iter': [200, 300],
                    'learning_rate': [0.001, 0.01]
                }
            },
            'RandomForest': {
                'class': 'RandomForest',
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': [1]
                }
            }
        }

        params_2 = {

            'RandomForest': {
                'class': 'HGRandomForest',
                'params': {
                    'n_estimators': [100],
                    'max_depth': [10],
                    'min_samples_split': [2],
                    'min_samples_leaf': [1],
                    'max_features': [1]
                }
            }
        }
        try:
            if os.path.exists(CLUSTERED_DATA_PATH):

                original_count = len(utils.retrieve_data(CLUSTERED_DATA_PATH_V2))
                print('ORIGINAL COUNT', original_count)


                predictor = PredictionWorkflow(
                    data=utils.retrieve_data(CLUSTERED_DATA_PATH_V2),
                    scoring=self.scoring,
                    params=params_2
                )

                predictor._prepare_data()
                predictor.grid_search()
                #predictor.save_best_model("model_eval/final_model.joblib")

        except Exception as e:
            print('failed')
            print(str(e))


if __name__ == "__main__":
    print('Do we have defined personalities? Lets find out..')
    print('Initiating workflow..')
    workflow = FullWorkflow(
        dataset_url=GOOGLE_DRIVE_URL,
        skip_download=True,
        skip_preprocessing=True,
        skip_clustering=True,
        skip_predictive=True,
        skip_cluster_analytics=True,
        use_prediction_v2=True,
        skip_region_predictive=True

    )
