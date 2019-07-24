import numpy as np


def knn_step_1():

    params = dict(n_neighbors=np.arange(2, 4, 1))

    return params


def xgb_step1():

    params = dict(max_depth=np.arange(1, 1000),
                  n_estimators=[1, 2, 3, 4, 5],
                  learning_rate=[0.01, 0.05, 0.1],
                  colsample_bytree=[0.6, 0.7, 0.8, 0.9, 1],
                  subsample=[0.7, 0.8, 0.9, 1],
                  min_child_weight=[1, 2, 3, 4])

    return params


def lr_step_1():

    params = dict(penalty=['l2'],
                  C=[0.001, 0.01, 0.1, 1, 10, 100])

    return params
