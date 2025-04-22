import os
import traceback
from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
from scripts.clustering import ClusteringWorkflow
from scripts.cluster_analytics_pipeline import ClusterAnalyticsPipeline

from scripts.cluster_prediction import ClusterPredictor
import scripts.utils as utils
from scripts.location_prediction import ImprovedPredictiveModel
import pandas as pd
import time



from scripts.cluster_analytics_pipeline import ClusterAnalyticsPipeline

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
            if not skip_clustering:
                self.clustering()

            if not skip_cluster_analytics:
                self.clustering_analytics()


            ### CLUSTER PREDICTION
            if not skip_predictive:
                if use_prediction_v2:
                    self.cluster_prediction_v2()
                else:
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
                (['scores'], True),
                (['survey_answers'], True),
                (['scores', 'survey_answers'], True),  # FA helps
            ]

            # Expanded hyperparameter space
            model_space = {

                'DBScan': {
                    'class': 'DBScan',
                    'params': {
                        'eps': [1.0, 1.5, 2.0],
                        'min_samples': [8, 12, 16, 20],
                        'n_factors': [5]  # NEW
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
                        save_results=False  #
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

    def cluster_prediction(self):
        """ Runs cluster prediction algorithm """
        try:
            if os.path.exists(CLUSTERED_DATA_PATH):

                predictor = ClusterPredictor(data=utils.retrieve_data(CLUSTERED_DATA_PATH_V2),
                                                   scoring=self.scoring)
                predictor.train_all()

                # Plot accuracies
                predictor.plot_model_accuracies()

        except Exception:
            print('failed')
            traceback.print_exc()


    def cluster_prediction_v2(self):
        """ Runs cluster prediction algorithm """
        try:
            if os.path.exists(CLUSTERED_DATA_PATH):

                predictor = ClusterPredictor(data=utils.retrieve_data(CLUSTERED_DATA_PATH_V2),
                                                   scoring=self.scoring)
                predictor.train_all()

                # Plot accuracies
                predictor.plot_model_accuracies()

        except Exception:
            print('failed')
            traceback.print_exc()

    def predictive_modeling(self,sample_frac,target):
        "Runs the predictive model"""
        try:
            if self.dataset is not None:
                print(f"Running improved predictive model on target: {target}")
                model = ImprovedPredictiveModel(
                    data=self.dataset,
                    sample_frac=sample_frac,
                    model_save_path=f"models/improved_rf_{target}.joblib"
                )
                model.run(target=target)
            else:
                print("Dataset not loaded.")
        except Exception:
            print("Improved predictive modeling failed:")
            traceback.print_exc()


    #########################################################
    #########################################################
    #########################################################


if __name__ == "__main__":
    print('Do we have defined personalities? Lets find out..')
    print('Initiating workflow..')
    workflow = FullWorkflow(
        dataset_url=GOOGLE_DRIVE_URL,
        skip_download=True,
        skip_preprocessing=True,
        skip_clustering=True,
        skip_predictive=False,
        skip_cluster_analytics=False,
        use_prediction_v2=False,
        skip_region_predictive=True

    )
