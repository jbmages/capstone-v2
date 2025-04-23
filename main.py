import os
import traceback
import time
import pandas as pd

from scripts.download_data import DataDownloader
from scripts.preprocess_data import DataPreprocessor
from scripts.clustering import ClusteringWorkflow
from scripts.cluster_analytics_pipeline import ClusterAnalyticsPipeline
from scripts.cluster_prediction import PredictionWorkflow
from scripts.location_prediction import ImprovedPredictiveModel
import scripts.utils as utils

GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id=1FzmqQDt_Amv0Gga4Rvo5iDrHuHBFGgrP'

RAW_DATA_PATH = 'data/data-final.csv'
CLEANED_DATA_PATH = 'data/cleaned_data_v2.csv'
CLUSTERED_DATA_PATH_V2 = 'scripts/cluster_eval/combined_clusters.csv'

class FullWorkflow:
    def __init__(self, dataset_url, demo=False):
        self.dataset_url = dataset_url
        self.dataset = None
        self.demo = demo
        self.scoring = utils.retrieve_excel('scoring/scoring.xlsx')

        os.makedirs("data", exist_ok=True)
        os.makedirs("model_eval", exist_ok=True)

        try:
            if demo or not os.path.exists(RAW_DATA_PATH):
                self.data_download()
            if demo or not os.path.exists(CLEANED_DATA_PATH):
                self.data_preprocessing()
            self.dataset = self.load_dataset()
            self.clustering()
            self.clustering_analytics()
            self.cluster_prediction()
        except Exception:
            traceback.print_exc()

    def data_download(self):
        print("[→] Downloading raw data...")
        DataDownloader(url=self.dataset_url)
        print(f"[✓] Data saved at {RAW_DATA_PATH}")

    def data_preprocessing(self):
        print("[→] Preprocessing raw data...")
        preprocessor = DataPreprocessor(dataset_path=RAW_DATA_PATH)
        preprocessor.process_data()
        print(f"[✓] Cleaned data saved at {CLEANED_DATA_PATH}")

    def load_dataset(self):
        print("[→] Loading cleaned dataset...")
        return utils.retrieve_data(CLEANED_DATA_PATH)

    def get_clustering_model_space(self):
        if self.demo:
            return {
                'KMeans': {
                    'class': 'KMeansHomegrown',
                    'params': {
                        'n_clusters': [4],
                        'max_iter': [50],
                        'n_init': [10],
                        'n_factors': [5]
                    }
                }
            }
        else:
            return {
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

    def clustering(self):
        print("[→] Running clustering grid search...")
        model_space = self.get_clustering_model_space()
        cluster_data_variants = [(['scores'], True)]
        all_results = []

        for cluster_data, fa_flag in cluster_data_variants:
            for model_key in model_space:
                subspace = {model_key: model_space[model_key]}
                print(f"[CONFIG] Data: {cluster_data} | FA: {fa_flag} | Model: {model_key}")
                workflow = ClusteringWorkflow(
                    data=self.dataset,
                    scoring_table=self.scoring,
                    cluster_data=cluster_data,
                    model_space=subspace,
                    apply_factor_analysis=fa_flag,
                    n_factors=5,
                    max_time=60 if self.demo else 120,
                    save_results=True
                )
                results = workflow.grid_search()
                all_results.extend(results)

        df = pd.DataFrame(all_results)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = f"model_eval/gridsearch_all_models_{timestamp}.csv"
        df.to_csv(out_path, index=False)
        print(f"[✓] Clustering results saved to {out_path}")

    def clustering_analytics(self):
        if os.path.exists(CLUSTERED_DATA_PATH_V2):
            print("[→] Running clustering analytics...")
            d = utils.retrieve_data(CLUSTERED_DATA_PATH_V2)
            ClusterAnalyticsPipeline(data=d, scoring=self.scoring)
        else:
            print("[!] Clustered data not found for analytics.")

    def get_prediction_model_space(self):
        if self.demo:
            return {
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
        else:
            return {
                'LogisticRegression': {
                    'class': 'LogisticRegression',
                    'params': {
                        'penalty': ['l1', 'l2'],
                        'c': [0.01, 0.1, 1.0, 10],
                        'solver': ['saga'],
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

    def cluster_prediction(self):
        if not os.path.exists(CLUSTERED_DATA_PATH_V2):
            print("[!] No clustered data available for prediction.")
            return

        print("[→] Running supervised cluster prediction...")
        params = self.get_prediction_model_space()

        predictor = PredictionWorkflow(
            data=utils.retrieve_data(CLUSTERED_DATA_PATH_V2),
            scoring=self.scoring,
            params=params
        )
        predictor._prepare_data()
        predictor.grid_search()
        print("[✓] Prediction complete.")

if __name__ == "__main__":
    print("\nDo we have defined personalities? Let’s find out.\n")

    # Set to demo=True to run the entire pipeline with demo configs
    FullWorkflow(
        dataset_url=GOOGLE_DRIVE_URL,
        demo=True  # <- Change this to False for full-sized model runs
    )
