import pandas as pd

def retrieve_data(filepath):
    """
    :param filepath:
    :return: returns pandas dataframe of datast for model training and evaluation
    """
    return pd.read_csv(filepath)
