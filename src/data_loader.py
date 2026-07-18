import pandas as pd


def load_data(path):
    """
    Load the industrial AI dataset.
    """

    df = pd.read_csv(path)

    return df