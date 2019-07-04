import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from core.config_loader import Config as Loader


def split_dataset(df, config):

    return df, df, df, df


def get_model(config):

    return GaussianNB()


def run(config):

    dataset_dir = os.path.join(Loader.path('datasets'), config.dataset_id, 'df_final.csv')
    df = pd.read_csv(dataset_dir)

    train_x, train_y, test_x, test_y = split_dataset(df, config)

    model = get_model(config)


