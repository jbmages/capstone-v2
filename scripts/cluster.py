import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import scripts.utils as utils


class ClusteringModel:
    def __init__(self, dataset: pd.DataFrame):
        """Initialize clustering model object"""
        print('initiating modeling infrastructure')
        self.data = dataset

        # Retrieve scoring table first
        self.scoring_table = utils.retrieve_data('../scoring/scoring.xlsx')
        print('scoring table correctly read in')
        # Prepare data segments
        self.survey_data, self.time_data, self.score_data = self.prep_data()

        print('data is prepped')
        # Placeholder for model
        self.model = None

    def prep_data(self):
        """Prepare data segments for clustering"""

        print('preparing data for modeling')
        survey_answer_cols = self.scoring_table['id'].tolist()
        time_cols = [col + '_E' for col in survey_answer_cols]
        score_cols = ['O score', 'C score', 'E score', 'A score', 'N score']

        # Ensure selected columns exist in dataset before slicing
        survey_data = self.data[survey_answer_cols] if all(
            col in self.data.columns for col in survey_answer_cols) else None

        time_data = self.data[time_cols] if all(col in self.data.columns for col in time_cols) else None
        # scale data
        score_data = self.data[score_cols] if all(col in self.data.columns for col in score_cols) else None

        print('scaling data')
        scaler = StandardScaler()

        return scaler.fit_transform(survey_data), scaler.fit_transform(time_data)  , scaler.fit_transform(score_data)

    def assign_clusters(self):
        """Assign clusters to data """


        pass  # Implement clustering logic later
