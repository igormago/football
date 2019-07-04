from sklearn.naive_bayes import GaussianNB
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
import os
import json
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from core.config import PATH_JOBS_RESULTS

stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']

def rps(y_true, y_pred):

    op1  = (y_pred[0] - y_true[0] + y_pred[1] - y_true[1])**2
    op2  = (y_pred[0] - y_true[0])**2

    return round(0.5*(op1+op2),5)


def rps_loss(y_true, y_pred):

    op1  = K.square(y_pred[0] - y_true[0] + y_pred[1] - y_true[1])
    op2  = K.square(y_pred[0] - y_true[0])

    return K.mean(0.5*(op1+op2),  axis=-1)

def rps_avg(y_true, y_pred):

    op1  = np.square(y_pred[:,0] - y_true['observed_home'] + y_pred[:,1] - y_true['observed_draw'])
    op2  = np.square(y_pred[:,0] - y_true['observed_home'])

    mean = np.mean(0.5*(op1+op2))
    return mean

def rps_avg_gnb(y_true):

    op1  = np.square(y_true['H'] - y_true['observed_home'] + y_true['D'] - y_true['observed_draw'])
    op2  = np.square(y_true['H'] - y_true['observed_home'])

    mean = np.mean(0.5*(op1+op2))
    return mean

def evaluation_gnb(test, pred_proba, classes):

    preds = pd.DataFrame(pred_proba, columns=classes)
    test.loc[:,'H'] = preds['H']
    test.loc[:,'D'] = preds['D']
    test.loc[:,'A'] = preds['A']
    #eval['observed_home'] = (eval['result'] == 'H') * 1  # mutiply to 1 to convert boolean in int
    #eval['observed_draw'] = (eval['result'] == 'D') * 1
    #eval['observed_away'] = (eval['result'] == 'A') * 1

    return test


def evaluation_accuracy(test, pred_proba):

    eval = test

    eval['p_H'] = (pred_proba[:,0] >= pred_proba[:,1]) & (pred_proba[:,0] >= pred_proba[:,2])
    eval['p_D'] = (eval['p_H'] == 0) & (pred_proba[:,1] >= pred_proba[:,2])
    eval['p_A'] = (eval['p_H'] == 0) & (eval['p_D'] == 0)

    hits = eval[(eval['p_H'] == eval['observed_home']) & (eval['p_D'] == eval['observed_draw']) & (eval['p_A'] == eval['observed_away'])]

    return len(hits)/len(test)

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

def gnb_by_minute_total(data, columns, job_label):

    target = 'result'
    print(columns)

    minute_dummies = pd.get_dummies(data['minute'], prefix='minute')

    data = pd.concat([data,minute_dummies], axis=1)
    columns.extend(minute_dummies.columns)

    print(data)
    X = data[columns]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    met = pd.DataFrame()

    y_test = pd.concat([data[[*minute_dummies.columns,'observed_home','observed_draw','observed_away']], y_test], axis=1, join='inner')

    for minute in range(0,96):

        col = 'minute_' + str(minute)
        X_test_minute = X_test[X_test[col] == 1]
        y_test_minute = y_test[y_test[col] == 1]

        #pd.options.mode.chained_assignment = None  # default='warn'
        pred_proba = clf.predict_proba(X_test_minute)
        pred = clf.predict(X_test_minute)

        print(clf.classes_)
        print(pred_proba)
        print(pred)

        y_test_minute['A'] = pred_proba[:, 2]
        y_test_minute['D'] = pred_proba[:, 1]
        y_test_minute['H'] = pred_proba[:, 0]

        met.loc[minute, "rps"] = rps_avg_gnb(y_test_minute)

        y_test_minute = y_test_minute[target]
        met.loc[minute, "acc"]  = clf.score(X_test_minute,y_test_minute)

        dirname = PATH_JOBS_RESULTS + job_label + "/"
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        print(met)

    filename = dirname  + "_all" + ".csv"
    print(filename)
    met.to_csv(filename)

def gnb_by_minute(data, features, types, name, minute, transform=False):

    columns = get_columns(features, types, minute, transform)

    target = 'result'

    X = data[columns]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    pd.options.mode.chained_assignment = None  # default='warn'
    pred_proba = clf.predict_proba(X_test)

    y_true = evaluation_gnb(y_test, pred_proba, clf)
    met = pd.DataFrame()
    met["acc"]  = [clf.score(X_test, y_test)]
    met["rps"]  = [rps_avg(y_true, pred_proba)]

    dirname = PATH_JOBS_RESULTS + name + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filename = dirname + str(minute) + ".csv"
    print(filename)
    met.to_csv(filename)

def knn_by_minute_total(data, columns, job_label, minute):

    target = 'result'

    X = data[columns]
    y = data[target]

    if False:
        dirname = PATH_JOBS_RESULTS + job_label + "/"
        bp_dir = dirname + "bestparams/"
        paramfile = bp_dir + job_label + "_" + str(minute) + ".json"

        with open(paramfile) as outfile:
            params = json.load(outfile)
    else:
        params = {'n_neighbors': 50}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = KNeighborsClassifier(**params)
    clf.fit(X_train, y_train)
    pd.options.mode.chained_assignment = None  # default='warn'
    pred_proba = clf.predict_proba(X_test)

    y_true = evaluation_gnb(y_test, pred_proba, clf.classes_)

    met = pd.DataFrame()
    met["acc"] = [clf.score(X_test, y_test)]
    met["rps"] = [rps_avg_gnb(y_true)]

    dirname = PATH_JOBS_RESULTS + job_label + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filename = dirname + str(minute) + ".csv"
    print(filename)
    met.to_csv(filename)

def xgb_by_minute_total(data, columns, job_label, minute):

    target = 'result'

    X = data[columns]
    y = data[target]

    if False:
        dirname = PATH_JOBS_RESULTS + job_label + "/"
        bp_dir = dirname + "bestparams/"
        paramfile = bp_dir + job_label + "_" + str(minute) + ".json"

        with open(paramfile) as outfile:
            params = json.load(outfile)
    else:
        params = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    pd.options.mode.chained_assignment = None  # default='warn'
    pred_proba = clf.predict_proba(X_test)

    y_true = evaluation_gnb(y_test, pred_proba, clf.classes_)

    met = pd.DataFrame()
    met["acc"] = [clf.score(X_test, y_test)]
    met["rps"] = [rps_avg_gnb(y_true)]

    dirname = PATH_JOBS_RESULTS + job_label + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filename = dirname + str(minute) + ".csv"
    print(filename)
    met.to_csv(filename)

def mlp_by_minute_total(data, columns, job_label, minute):

    target = 'result'

    X = data[columns]
    y = data[target]

    if False:
        dirname = PATH_JOBS_RESULTS + job_label + "/"
        bp_dir = dirname + "bestparams/"
        paramfile = bp_dir + job_label + "_" + str(minute) + ".json"

        with open(paramfile) as outfile:
            params = json.load(outfile)
    else:
        params = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = MLPClassifier(hidden_layer_sizes=(7,))
    clf.fit(X_train, y_train)
    pd.options.mode.chained_assignment = None  # default='warn'
    pred_proba = clf.predict_proba(X_test)

    y_true = evaluation_gnb(y_test, pred_proba, clf.classes_)

    met = pd.DataFrame()
    met["acc"] = [clf.score(X_test, y_test)]
    met["rps"] = [rps_avg_gnb(y_true)]

    dirname = PATH_JOBS_RESULTS + job_label + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filename = dirname + str(minute) + ".csv"
    print(filename)
    met.to_csv(filename)

def knn_by_minute(data, features, types, name, minute, transform=False):

    columns = get_columns(features, types, minute, transform)

    target = ['observed_home', 'observed_draw', 'observed_away']

    X = data[columns]
    y = data[target]

    dirname = PATH_JOBS_RESULTS + name + "/"
    bp_dir = dirname + "bestparams/"
    paramfile = bp_dir + name + "_" + str(minute) + ".json"

    with open(paramfile) as outfile:
        params = json.load(outfile)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = KNeighborsClassifier(**params)
    clf.fit(X_train, y_train)
    pd.options.mode.chained_assignment = None  # default='warn'
    pred_proba = clf.predict(X_test)

    acc = evaluation_accuracy(y_test,pred_proba)

    met = pd.DataFrame()
    met["acc"] = [acc]
    met["rps"] = [rps_avg(y_test, pred_proba)]

    dirname = PATH_JOBS_RESULTS + name + "/"
    filename = dirname + str(minute) + ".csv"

    met.to_csv(filename)

def learning_by_minute_svc(data, features, types, name, minute):

    columns = []

    for stat in features:
        for t in types:
            col = 'ts_' + stat + '_' + t + '_' + str(minute)
            columns.append(col)


    target = 'result'

    X = data[columns]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = SVC(probability=True)
    clf.fit(X_train, y_train)
    pd.options.mode.chained_assignment = None  # default='warn'
    pred_proba = clf.predict_proba(X_test)

    score = evaluation_accuracy(y_test, pred_proba)

    met = pd.DataFrame()
    met["acc"]  = [clf.score(X_test, y_test)]
    met["rps"]  = [score/len(X_test)]

    filename = PATH_JOBS_RESULTS + name + ".csv"
    print(filename)
    met.to_csv(filename)

def learning_by_minute_nn(data, features, types, name, minute, transform=False):

    columns = get_columns(features, types, minute, transform)

    target = ['observed_home', 'observed_draw', 'observed_away']

    X = data[columns]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print(len(columns))
    clf = Sequential()
    clf.add(Dense(30, input_dim=len(columns), activation='relu'))
    clf.add(Dense(30, activation='relu'))
    clf.add(Dense(3, activation='softmax'))

    clf.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    clf.fit(X_train, y_train, epochs=10, batch_size=128)

    pd.options.mode.chained_assignment = None  # default='warn'
    pred_proba = clf.predict(X_test)

    score = evaluation_keras(y_test, pred_proba, clf)

    met = pd.DataFrame()
    met["acc"] = [clf.evaluate(X_test, y_test)[1]]
    met["rps"] = [score / len(X_test)]
    filename = PATH_JOBS_RESULTS + name + ".csv"
    print(filename)
    met.to_csv(filename)

def learning_keras_all_time(data, features, name, loss=False):

    print("entrou3")
    met = pd.DataFrame()
    met['minute'] = range(1, 96)
    met.set_index('minute', inplace=True)
    print("entrou4")

    for i in range(1, 96):

        print(i)
        columns = []

        for stat in features:
            for j in range(i - 1, i):
                for l in ['dif', 'div']:
                    col = 'ts_' + stat + '_' + l + '_' + str(j)
                    columns.append(col)

        target = ['observed_home', 'observed_draw', 'observed_away']

        X = data[columns]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = Sequential()
        clf.add(Dense(len(columns), input_dim=len(columns), activation='relu'))
        clf.add(Dense(7, activation='relu'))
        clf.add(Dense(3, activation='softmax'))

        if loss:
            clf.compile(loss=rps_loss, optimizer='adam', metrics=['accuracy'])
        else:
            clf.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

        clf.fit(X_train, y_train, epochs=10, batch_size=100)

        pd.options.mode.chained_assignment = None  # default='warn'
        pred_proba = clf.predict(X_test)

        score = evaluation_keras(y_test, pred_proba, clf)

        met.loc[i,"acc"] = clf.evaluate(X_test, y_test)[1]
        met.loc[i,"rps"] = score / len(X_test)

    filename = PATH_JOBS_RESULTS + name + ".csv"
    print(filename)
    met.to_csv(filename)

def learning_dnn_lm(data, features, name, loss=False):

    print("entrou3")
    met = pd.DataFrame()
    met['minute'] = range(1, 96)
    met.set_index('minute', inplace=True)
    print("entrou4")

    for i in range(1, 96):

        print(i)
        columns = []

        for stat in features:
            for j in range(i - 1, i):
                for l in ['dif', 'div']:
                    col = 'ts_' + stat + '_' + l + '_' + str(j)
                    columns.append(col)

        target = ['observed_home', 'observed_draw', 'observed_away']

        X = data[columns]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = Sequential()
        clf.add(Dense(len(columns)*2, input_dim=len(columns), activation='relu'))
        clf.add(Dense(len(columns)*2, activation='relu'))
        clf.add(Dense(3, activation='softmax'))

        if loss:
            clf.compile(loss=rps_loss, optimizer='adam', metrics=['accuracy'])
        else:
            clf.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        clf.fit(X_train, y_train, epochs=20, batch_size=32)

        pd.options.mode.chained_assignment = None  # default='warn'
        pred_proba = clf.predict(X_test)

        score = evaluation_keras(y_test, pred_proba, clf)

        met.loc[i,"acc"] = clf.evaluate(X_test, y_test)[1]
        met.loc[i,"rps"] = score / len(X_test)

    filename = PATH_JOBS_RESULTS + name + ".csv"
    print(filename)
    met.to_csv(filename)



def learning_cnn_all_time(data, features, name):

    print("entrou3")
    met = pd.DataFrame()
    met['minute'] = range(1, 96)
    met.set_index('minute', inplace=True)
    print("entrou4")

    for i in range(1, 96):

        print(i)
        columns = []

        for stat in features:
            for j in range(0,i):
                for l in ['dif', 'div']:
                    col = 'ts_' + stat + '_' + l + '_' + str(j)
                    columns.append(col)

        batch_size = 128
        epochs = 10

        # input image dimensions
        img_rows, img_cols = i, len(columns)

        target = ['observed_home', 'observed_draw', 'observed_away']

        X = data[columns]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = Sequential()
        clf.add(Conv2D(filters = 32, kernel_size=(1, 1),
                         activation='relu',
                         input_shape=(1, 14, 1)))
        clf.add(Dense(len(columns), input_dim=len(columns), activation='relu'))
        clf.add(Dense(7, activation='relu'))
        clf.add(Dense(3, activation='softmax'))

        clf.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        clf.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        pd.options.mode.chained_assignment = None  # default='warn'
        pred_proba = clf.predict(X_test)

        score = evaluation_keras(y_test, pred_proba, clf)

        met.loc[i,"acc"] = clf.evaluate(X_test, y_test)[1]
        met.loc[i,"rps"] = score / len(X_test)

