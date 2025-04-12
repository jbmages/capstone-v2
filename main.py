import os
import traceback
from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
from scripts.cluster import ClusteringModel
from scripts.cluster_prediction import ClusterPredictor
import scripts.utils as utils
from scripts.region_prediction import PredictiveModel
import pandas as pd
import time

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
                 use_prediction_v2=True, skip_region_predictive=True):
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
        """ Comprehensive clustering benchmark """

        try:
            print("[🚀] Starting full clustering grid search...")

            # Combinations of inputs to try
            cluster_data_variants = [
                (['scores'], False),
                (['survey_answers'], False),
                (['scores'], True),
                (['survey_answers'], True),
                (['scores', 'survey_answers'], False),
                (['scores', 'survey_answers'], True),
            ]

            # Master model space
            model_space = {
                'KMeans': {
                    'class': 'KMeans',
                    'params': {
                        'n_clusters': [3, 5, 7, 9],
                        'batch_size': [64],
                        'max_iter': [100]
                    }
                },
                'GMM': {
                    'class': 'GMM',
                    'params': {
                        'n_components': [3, 5, 7],
                        'covariance_type': ['full', 'tied'],
                        'max_iter': [100]
                    }
                },
                'DBScan': {
                    'class': 'DBScan',
                    'params': {
                        'eps': [0.3, 0.5, 0.7],
                        'min_samples': [5, 10]
                    }
                },
                'Hierarchical': {
                    'class': 'Hierarchical',
                    'params': {
                        'n_clusters': [3, 5, 7],
                        'affinity': ['euclidean'],
                        'linkage': ['ward']
                    }
                }
            }

            all_results = []

            for cluster_data, fa_flag in cluster_data_variants:
                for model_key in model_space.keys():
                    subspace = {model_key: model_space[model_key]}

                    print(f"\n[⚙️ CONFIG] Data: {cluster_data} | FA: {fa_flag} | Model: {model_key}")

                    workflow = ClusteringWorkflow(
                        data=self.dataset,
                        scoring_table=self.scoring,
                        cluster_data=cluster_data,
                        model_space=subspace,
                        apply_factor_analysis=fa_flag,
                        n_factors=5,
                        max_time=120,
                        save_results=False  # We'll save one final combined CSV instead
                    )

                    results = workflow.grid_search()
                    all_results.extend(results)

            # Save full results
            df = pd.DataFrame(all_results)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_path = f"model_eval/gridsearch_all_models_{timestamp}.csv"
            os.makedirs("model_eval", exist_ok=True)
            df.to_csv(out_path, index=False)

            # Show top results
            print("\n[🏁 DONE] Top Models by Silhouette Score:")
            top_df = df.sort_values(by='silhouette', ascending=False).head(10)
            #print(top_df[['model', 'data_type', 'factor_analysis', 'silhouette', 'calinski_harabasz', 'density_gain']])
            print(top_df[['model', 'data_type', 'factor_analysis', 'silhouette', 'calinski_harabasz']])
            print(f"\n[SAVED] Full results to: {out_path}")

        except Exception as e:
            import traceback
            print("[ERROR] Exception occurred during clustering:")
            traceback.print_exc()

    #########################################################
    #########################################################
    #########################################################
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


    #########################################################
    #########################################################
    #########################################################


if __name__ == "__main__":
    print('Do we have defined personalities? Lets find out..')
    print('Initiating workflow..')
    workflow = FullWorkflow(
        dataset_url=GOOGLE_DRIVE_URL,
        #skip_download=True,
        #skip_preprocessing=True,
        #skip_clustering=False,
        skip_predictive=True,
        use_clustering_v2=True,
        use_prediction_v2=True,
        skip_region_predictive=True

    )
