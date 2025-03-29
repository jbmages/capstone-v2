import pandas as pd

def retrieve_data(filepath):
    """
    :param filepath:
    :return: returns pandas dataframe of dataset for model training and evaluation
    """
    return pd.read_csv(filepath)

def retrieve_excel(filepath):
    """

    :param filepath:
    :return: returns pandas df of excel
    """
    return pd.read_excel(filepath)