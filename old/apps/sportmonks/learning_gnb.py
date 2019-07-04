from sklearn.naive_bayes import GaussianNB
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.neighbors import KNeighborsClassifier

from core.config import PATH_JOBS_RESULTS

stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']

def rps(y_true, y_pred):

    op1  = (y_pred[0] - y_true[0] + y_pred[1] - y_true[1])**2
    op2  = (y_pred[0] - y_true[0])**2

    return round(0.5*(op1+op2),5)

def rps_avg(y_true, y_pred):

    op1  = np.square(y_pred[:,0] - y_true['observed_home'] + y_pred[:,1] - y_true['observed_draw'])
    op2  = np.square(y_pred[:,0] - y_true['observed_home'])

    mean = np.mean(0.5*(op1+op2))

    return mean

def evaluation(test, pred_proba, clf):

    eval = test.to_frame()

    eval = eval.join(pd.DataFrame(
        pred_proba,
        index=eval.index,
        columns=clf.classes_
    ))

    eval['ob_H'] = (eval['result'] == 'H') * 1  # mutiply to 1 to convert boolean in int
    eval['ob_D'] = (eval['result'] == 'D') * 1
    eval['ob_A'] = (eval['result'] == 'A') * 1

    score = 0

    for ev in eval.iterrows():
        e = ev[1]
        predicted = [e['H'], e['D'], e['A']]
        observed = [e['ob_H'], e['ob_D'], e['ob_A']]

        score = score + rps(predicted, observed)

    return score

def evaluation_keras(test, pred_proba, clf):

    eval = test.join(pd.DataFrame(
        pred_proba,
        index=test.index,
        columns=['H', 'D', 'A']
    ))

    score = 0

    for ev in eval.iterrows():
        e = ev[1]
        predicted = [e['H'], e['D'], e['A']]
        observed = [e['observed_home'], e['observed_draw'], e['observed_away']]

        score = score + rps(predicted, observed)

    return score

def get_columns(features, types, minute, transform=False):

    columns = []
    for stat in features:
        for t in types:
            label = 'ts_' + stat + '_' + t + '_' + str(minute)
            col = label
            columns.append(col)

    if transform:
        for stat in features:
            for t in ['home', 'away']:
                label = 'ts_' + stat + '_' + t + '_' + str(minute)

                col = label + '_mean'
                columns.append(col)
                col = label + '_var'
                columns.append(col)
                col = label + '_kut'
                columns.append(col)


    return columns


def learning_by_minute_gnb(data, features, types, name, minute, transform=False):

    columns = get_columns(features, types, minute, transform)

    target = 'result'

    X = data[columns]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    pd.options.mode.chained_assignment = None  # default='warn'
    pred_proba = clf.predict_proba(X_test)

    met = pd.DataFrame()
    met["acc"]  = [clf.score(X_test, y_test)]
    met["rps"]  = [rps_avg(y_test, pred_proba)]

    dirname = PATH_JOBS_RESULTS + name + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filename = dirname + name + "_" + str(minute) + ".csv"
    print(filename)
    met.to_csv(filename)

