import os
import models.features as ft
import pandas as pd

from core.config_loader import SysConfig


class Dataset:

    def __init__(self, setup):
        self.setup = setup

        dataset_dir = os.path.join(SysConfig.path('datasets'), setup.dataset_id, 'final.csv')
        self.data = pd.read_csv(dataset_dir)
        self.features = ft.get_features(setup)
        self.target = 'result'

        if self.setup.is_minute_feature:
            self.train_size = self.setup.train_size * 96
        else:
            self.train_size = self.setup.train_size

        self.train = self.data[:self.train_size]
        self.test = self.data[self.train_size:]

        self.x_train = self.train[self.features]
        self.y_train = self.train[self.target]
        self.x_test = self.test[self.features]
        self.y_test = self.test[self.target]

    def reset_train_test_by_minute(self, minute):

        self.train = self.data[:self.train_size][self.data['minute'] == minute]
        self.test = self.data[self.train_size:][self.data['minute'] == minute]

        self.x_train = self.train[self.features]
        self.y_train = self.train[self.target]
        self.x_test = self.test[self.features]
        self.y_test = self.test[self.target]
