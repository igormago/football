import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from models import scores
import models.grids as gp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from core.config_loader import SysConfig

from core.logger import log_in_out, logging
logger = logging.getLogger('Model')


class ModelFactory:

    @staticmethod
    def load(setup):

        if setup.is_minute_feature:
            return ModelSingle(setup)
        else:
            return ModelByMinute(setup)


class Model:

    def __init__(self, setup):
        self.setup = setup
        self.classifier = self._init_model()

    @log_in_out
    def _init_model(self):

        if self.setup.is_to_tune or self.setup.is_to_train:
            if self.setup.classifier == 'GaussianNB':
                clf = GaussianNB()
            elif self.setup.classifier == 'KNeighborsClassifier':
                clf = KNeighborsClassifier()
            elif self.setup.classifier == 'LogisticRegression':
                clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
            elif self.setup.classifier == 'RandomForestClassifier':
                clf = RandomForestClassifier()
            elif self.setup.classifier == 'XGBClassifier':
                clf = XGBClassifier()
            else:
                raise Exception
        else:
            clf = self._load_model()

        return clf

    @log_in_out
    def _load_model(self):

        file_name = str(self.setup.minute) + '.pkl'
        dir_name = os.path.join(SysConfig.path('models'), self.setup.classifier,
                                self.setup.dataset_id, self.setup.features)

        file_path = os.path.join(dir_name, file_name)

        with open(file_path, 'rb') as outfile:
            model = joblib.load(outfile)

        return model

    @log_in_out
    def train(self, data):

        if self.setup.is_to_tune:
            logger.debug('Tuning model')
            params = self._get_grid_params()
            grid = self._get_grid(params)
            grid.fit(data.x_train, data.y_train)
            self.classifier = grid.best_estimator_
            self._save_cv_results(grid)

        else:
            logger.debug('Training model')
            self.classifier.fit(data.x_train, data.y_train)

        self._save_model(self.classifier)

    def _get_grid(self, params):

        rps_error = make_scorer(scores.rps_error, greater_is_better=False, needs_proba=True)

        if self.setup.grid_type == 'grid':
            grid = GridSearchCV(self.classifier, params, scoring=rps_error, return_train_score=True, cv=3, verbose=10)
        elif self.setup.grid_type == 'random':
            grid = RandomizedSearchCV(self.classifier, params, scoring=rps_error, cv=3)
        else:
            raise Exception

        return grid

    def _get_grid_params(self):

        method_to_call = getattr(gp, str(self.setup.grid_params))
        return method_to_call()

    def _save_model(self, model):

        file_name = str(self.setup.minute) + '.pkl'
        dir_name = os.path.join(SysConfig.path('models'), self.setup.classifier,
                                self.setup.dataset_id, self.setup.features)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        file_path = os.path.join(dir_name, file_name)

        with open(file_path, 'wb') as outfile:
            joblib.dump(model, outfile)

    def _save_cv_results(self, grid):

        file_name = str(self.setup.minute) + '.csv'
        dir_name = os.path.join(SysConfig.path('tuning'), self.setup.classifier,
                                self.setup.dataset_id, self.setup.features)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        file_path = os.path.join(dir_name, file_name)

        df = pd.DataFrame(grid.cv_results_)
        with open(file_path, 'w') as outfile:
            df.to_csv(outfile)

    def _save_results(self, acc_train, acc_test, rps_train, rps_test, minute):

        results = dict()
        results['acc_train'] = acc_train
        results['acc_test'] = acc_test

        results['rps_train'] = rps_train
        results['rps_test'] = rps_test

        filename = str(minute) + '.json'

        score_dir = os.path.join(SysConfig.path('results'),
                                 self.setup.classifier, self.setup.dataset_id, self.setup.features)
        score_file = os.path.join(score_dir, filename)

        if not os.path.exists(score_dir):
            os.makedirs(score_dir)

        with open(score_file, 'w') as outfile:
            json.dump(results, outfile)

    def _get_scores(self, data):

        acc_train = self.classifier.score(data.x_train, data.y_train)
        acc_test = self.classifier.score(data.x_test, data.y_test)

        y_hat_train_prob = self.classifier.predict_proba(data.x_train)
        y_hat_test_prob = self.classifier.predict_proba(data.x_test)

        y_train_prob = data.train[['observed_away', 'observed_draw', 'observed_home']].values
        y_test_prob = data.test[['observed_away', 'observed_draw', 'observed_home']].values

        rps_train = scores.rps(y_train_prob, y_hat_train_prob)
        rps_test = scores.rps(y_test_prob, y_hat_test_prob)

        return acc_train, acc_test, rps_train, rps_test


class ModelByMinute(Model):

    def __init__(self, setup):
        super().__init__(setup)

    @log_in_out
    def test(self, data):

        logger.debug('Evaluating model')
        acc_train, acc_test, rps_train, rps_test = self._get_scores(data)
        self._save_results(acc_train, acc_test, rps_train, rps_test, self.setup.minute)


class ModelSingle(Model):

    def __init__(self, setup):
        super().__init__(setup)

    @log_in_out
    def test(self, data):

        for minute in range(96):
            logger.debug('Evaluating motel at minute %i' % minute)
            data.filter_train_test_by_minute(minute)
            acc_train, acc_test, rps_train, rps_test = self._get_scores(data)
            self._save_results(acc_train, acc_test, rps_train, rps_test, minute)