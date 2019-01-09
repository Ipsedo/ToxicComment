import pandas as pd


def load_train_csv(file_name):
    return pd.read_csv(file_name, sep=',', header='infer')

