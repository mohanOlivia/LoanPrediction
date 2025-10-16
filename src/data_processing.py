import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def fill_missing_values(df):
    return df