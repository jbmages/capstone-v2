import pandas as pd
import numpy as np
from tqdm import tqdm
import scripts.utils as utils  # Ensure correct path for utils.py


class ClusteringModel:
    def __init__(self, dataset: pd.DataFrame):
        """Initialize clustering model object"""
        self.data = dataset

        # Retrieve scoring table first
        self.scoring_table = utils.retrieve_data('../scoring/scoring.xlsx')

        # Prepare data segments
        self.survey_data, self.time_data, self.score_data = self.prep_data()

        # Placeholder for model
        self.model = None

    def prep_data(self):
        """Prepare data segments for clustering"""
        survey_answer_cols = self.scoring_table['id'].tolist()
        time_cols = [col + '_E' for col in survey_answer_cols]
        score_cols = ['O score', 'C score', 'E score', 'A score', 'N score']

        # Ensure selected columns exist in dataset before slicing
        survey_data = self.data[survey_answer_cols] if all(
            col in self.data.columns for col in survey_answer_cols) else None
        time_data = self.data[time_cols] if all(col in self.data.columns for col in time_cols) else None
        score_data = self.data[score_cols] if all(col in self.data.columns for col in score_cols) else None

        return survey_data, time_data, score_data

    def assign_clusters(self):
        """Assign clusters to data"""
        pass  # Implement clustering logic later
