import pandas as pd


class ClusterPredictor:

    def __init__(self, data, model):
        # initialize clustering predictive model
        self.data = data
        self.model = model
