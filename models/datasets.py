import os

import pandas as pd

from core.config_loader import SysConfig


class DatasetFactory:

    @staticmethod
    def load(setup):

        if setup.is_minute_feature:
            return DatasetWithMinute(setup)
        else:
            return DatasetStandard(setup)


class Dataset:

    def __init__(self, setup):

        self._list_features_cards = ['yellow_cards', 'red_cards']
        self._list_comparatives = ['ratio', 'sub']
        self._list_locales = ['home', 'away']
        self._list_features_trends = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']
        self._list_features_scouts = self._list_features_trends + ['possession']

        self.setup = setup

        dataset_dir = os.path.join(SysConfig.path('datasets'), self.setup.dataset_id, 'final.csv')
        self.data = pd.read_csv(dataset_dir)
        self.features = self.get_features()
        self.target = 'result'

        self.train_size = self.setup.train_size

        self.train = self.data[:self.train_size]
        self.test = self.data[self.train_size:]

        self.x_train = self.train[self.features]
        self.y_train = self.train[self.target]
        self.x_test = self.test[self.features]
        self.y_test = self.test[self.target]

    def get_features(self):

        method_to_call = getattr(self, '_get_ft_' + str(self.setup.features))
        return method_to_call()


class DatasetStandard(Dataset):

    def __init__(self, setup):
        super().__init__(setup)

    def _get_ft_cum_goals(self):

        t = 'goals'
        features = list()

        for l in self._list_locales:
            col = '_'.join(['accum_trends', t, l, str(self.setup.minute)])
            features.append(col)

        return features

    def _get_ft_cum_scouts(self):

        features = list()

        for t in self._list_features_trends:
            for l in self._list_locales:
                col = '_'.join(['accum_trends', t, l, str(self.setup.minute)])
                features.append(col)

        for t in self._list_features_cards:
            for l in self._list_locales:
                col = '_'.join(['accum_cards', t, l, str(self.setup.minute)])
                features.append(col)

        t = 'possession'
        col = '_'.join(['ratio_trends', t, str(self.setup.minute)])
        features.append(col)

        return features

    def _get_ft_comp_goals(self):

        features = list()
        t = 'goals'
        for c in self._list_comparatives:
            col = '_'.join([c, 'trends', t, str(self.setup.minute)])
            features.append(col)

        return features

    def _get_ft_comp_scouts(self):

        features = list()

        for t in self._list_features_scouts:
            for c in self._list_comparatives:
                col = '_'.join([c, 'trends', t, str(self.setup.minute)])
                features.append(col)

        for t in self._list_features_cards:
            for c in self._list_comparatives:
                col = '_'.join([c, 'cards', t, str(self.setup.minute)])
                features.append(col)

        return features


class DatasetWithMinute(Dataset):

    def __init__(self, setup):

        setup.dataset_id = '_'.join([setup.dataset_id, 'min'])
        setup.train_size = setup.train_size * 96

        super().__init__(setup)

    def filter_train_test_by_minute(self, minute):

        self.train = self.data[:self.train_size]
        self.train = self.train[self.train['minute'] == minute]
        self.test = self.data[self.train_size:]
        self.test = self.test[self.test['minute'] == minute]

        self.x_train = self.train[self.features]
        self.y_train = self.train[self.target]
        self.x_test = self.test[self.features]
        self.y_test = self.test[self.target]

    def _get_ft_cum_goals(self):

        features = list()

        t = 'goals'
        for l in self._list_locales:
            col = '_'.join(['accum_trends', t, l])
            features.append(col)

        features.append('minute')

        return features

    def _get_ft_cum_scouts(self):

        features = list()

        for t in self._list_features_trends:
            for l in self._list_comparatives:
                col = '_'.join(['accum_trends', t, l])
                features.append(col)

        for t in self._list_features_cards:
            for l in self._list_comparatives:
                col = '_'.join(['accum_cards', t, l])
                features.append(col)

        t = 'possession'
        col = '_'.join(['ratio_trends', t])
        features.append(col)

        features.append('minute')

        return features

    def _get_ft_comp_goals(self):

        features = list()

        t = 'goals'
        for c in self._list_comparatives:
            col = '_'.join([c, 'trends', t])
            features.append(col)

        features.append('minute')

        return features

    def _get_ft_comp_scouts(self):

        features = list()

        for t in self._list_features_scouts:
            for c in self._list_comparatives:
                col = '_'.join([c, 'trends', t])
                features.append(col)

        for t in self._list_features_cards:
            for c in self._list_comparatives:
                col = '_'.join([c, 'cards', t])
                features.append(col)

        features.append('minute')

        return features
