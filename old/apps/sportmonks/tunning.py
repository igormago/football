import keras
import pandas as pd
import numpy as np
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Average, GlobalAveragePooling1D, Masking, Permute, \
    Reshape, multiply, concatenate, CuDNNLSTM
from keras.utils import np_utils
from keras import backend as K, Input, Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from apps.sportmonks import utils, testing
from sklearn.metrics import make_scorer
import os
import json
from xgboost import XGBRegressor, XGBClassifier
from core.config import PATH_JOBS_RESULTS, PATH_JOBS_MODELS
from core.config import PATH_SPORTMONKS_DATAFRAMES, PATH_APP
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import optimizers
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.normalization import BatchNormalization
import pickle
from apps.sportmonks import arch

stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']


def rps_error(y_true, y_pred, **kwargs):

    y = pd.DataFrame()
    y['observed_home'] = (y_true == 'A') * 1
    y['observed_draw'] = (y_true == 'D') * 1
    y['observed_away'] = (y_true == 'H') * 1

    return rps_avg(y, y_pred)


def rps_avg(y_true, y_pred):

    op1 = np.square(y_pred[:,0] - y_true['observed_home'] + y_pred[:,1] - y_true['observed_draw'])
    op2 = np.square(y_pred[:,0] - y_true['observed_home'])

    mean = np.mean(0.5*(op1+op2))
    return mean

def rps_keras(y_true, y_pred):

    op1 = np.square(y_pred[:,0] - y_true[:,0] + y_pred[:,1] - y_true[:,1])
    op2 = np.square(y_pred[:,0] - y_true[:,0])

    mean = np.mean(0.5*(op1+op2))
    return mean


def rps(y_true, y_pred):

    op1 = (y_pred[0] - y_true[0] + y_pred[1] - y_true[1])**2
    op2 = (y_pred[0] - y_true[0])**2

    return round(0.5*(op1+op2),5)


def rps_loss(y_true, y_pred):

    op1 = K.square(y_pred[:,0] - y_true[:,0] + y_pred[:,1] - y_true[:,1])
    op2 = K.square(y_pred[:,0] - y_true[:,0])

    return K.mean(0.5*(op1+op2),  axis=-1)

def evaluation_accuracy(test, pred_proba, clf):

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


def get_columns_transformed(features, types, minute, transform=False):

    columns = ['possession_possession_home_' + str(minute)]
    for stat in features:
        for t in types:
            if stat != 'possession':
                label = t + '_' + stat + '_' + str(minute)
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


def knn_by_minute(data, features, prefix, name, minute, step):

    features = utils.get_features(features, prefix, minute)
    target = 'result'

    rps_score = make_scorer(rps_error, greater_is_better=False, needs_proba=True)

    if step == 1:
        params = {'n_neighbors': np.arange(2,500,1)}
    else:

        dirname = PATH_JOBS_RESULTS + name + "/"
        bp_dir = dirname + "bestparams/"
        paramfile = bp_dir + "1_" + str(minute) + ".json"

        x = 1
        with open(paramfile) as outfile:
            params = json.load(outfile)
            n_neighbors = params['n_neighbors']
            n_start = n_neighbors-x
            n_end = n_neighbors+x

            if n_start < 2:
                n_start = 2

            params = {'n_neighbors': np.arange(n_start, n_end, 1)}

    train_x, train_y, test_x, test_y = utils.split_train_test(data, features, target)

    model = KNeighborsClassifier()

    if step == 1:
        gs = RandomizedSearchCV(model, params, cv=5, scoring=rps_score,
                                verbose=1, return_train_score=True, n_iter=100)
    else:
        gs = GridSearchCV(model, params, cv=5, scoring=rps_score, verbose=1, return_train_score=True)

    gs.fit(train_x, train_y)

    tunned = pd.DataFrame(gs.cv_results_['params'])
    tunned['rps_train'] = gs.cv_results_['mean_train_score'] * -1
    tunned['std_train'] = gs.cv_results_['std_train_score']
    tunned['rps_test'] = gs.cv_results_['mean_test_score'] * -1
    tunned['std_test'] = gs.cv_results_['std_test_score']

    dirname = PATH_JOBS_RESULTS + name + "/"

    tun_dir = dirname + "tunning/"
    bp_dir = dirname + "bestparams/"

    if not os.path.exists(dirname):
        os.mkdir(dirname)
        os.mkdir(tun_dir)
        os.mkdir(bp_dir)

    label = str(step) + "_" + str(minute)
    filename = tun_dir + label + ".csv"
    paramfile = bp_dir + label + ".json"

    bp = gs.best_params_
    bp['n_neighbors'] = int(bp['n_neighbors'])

    with open(paramfile, 'w') as outfile:
        json.dump(bp, outfile)

    tunned.to_csv(filename)

    dirname = PATH_JOBS_MODELS + name + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model_file = dirname + label + ".joblib"
    with open(model_file, "wb") as f:
        pickle.dump(gs.best_estimator_, f)


def xgb_by_minute(data, features, prefix, name, minute, step, odds):

    print(data.columns)
    features = utils.get_features(features, prefix, minute)
    if odds:
        features.append('odds_p_avg_home')
        features.append('odds_p_avg_draw')
        features.append('odds_p_avg_away')
        features.append('odds_p_std_home')
        features.append('odds_p_std_draw')
        features.append('odds_p_std_away')

    target = 'result'

    rps_score = make_scorer(rps_error, greater_is_better=False, needs_proba=True)

    if step == 1:

        n_estimators = np.arange(1, 1000)
        max_depth = [1, 2, 3, 4, 5]
        learning_rate = [0.01, 0.05, 0.1]
        colsample_bytree = [0.6, 0.7, 0.8, 0.9, 1]
        subsample = [0.7, 0.8, 0.9, 1]
        min_child_weight = [1, 2, 3, 4]

    else:

        dirname = PATH_JOBS_RESULTS + name + "/"
        bp_dir = dirname + "bestparams/"
        paramfile = bp_dir + str(minute) + ".json"

        with open(paramfile) as outfile:
            params = json.load(outfile)
            n_neighbors = params['n_neighbors']
            n_start = n_neighbors-10
            n_end = n_neighbors+10

            if n_start < 2:
                n_start = 2

            params = {'n_neighbors': np.arange(n_start, n_end, 1)}

    train_x, train_y, test_x, test_y = utils.split_train_test(data, features, target, odds)

    params = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
                  colsample_bytree=colsample_bytree, subsample=subsample, min_child_weight=min_child_weight)

    model = XGBClassifier(objective="multi:softmax",num_class=3)

    gs = RandomizedSearchCV(model, params, cv=5, scoring=rps_score, verbose=10, return_train_score=True, n_iter=100)
    gs.fit(train_x, train_y)

    tunned = pd.DataFrame(gs.cv_results_['params'])
    tunned['rps_train'] = gs.cv_results_['mean_train_score'] * -1
    tunned['std_train'] = gs.cv_results_['std_train_score']
    tunned['rps_test'] = gs.cv_results_['mean_test_score'] * -1
    tunned['std_test'] = gs.cv_results_['std_test_score']

    dirname = PATH_JOBS_RESULTS + name + "/"

    tun_dir = dirname + "tunning/"
    bp_dir = dirname + "bestparams/"

    if not os.path.exists(dirname):
        os.mkdir(dirname)
        os.mkdir(tun_dir)
        os.mkdir(bp_dir)

    label = str(step) + "_" + str(minute)
    filename = tun_dir + label + ".csv"
    paramfile = bp_dir + label + ".json"

    bp = gs.best_params_
    bp['max_depth'] = int(bp['max_depth'])
    bp['n_estimators'] = int(bp['n_estimators'])

    with open(paramfile, 'w') as outfile:
        json.dump(bp, outfile)

    tunned.to_csv(filename)

    dirname = PATH_JOBS_MODELS + name + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model_file = dirname + label + ".joblib"
    with open(model_file, "wb") as f:
        pickle.dump(gs.best_estimator_, f)


def dnn_by_minute(data, features, prefix, name, config):
    minute = int(config['minute'])

    try:
        optim_num_epochs = int(config['optim_num_epochs'])
    except KeyError:
        optim_num_epochs = 1000

    try:
        optim_batch_size = int(config['optim_batch_size'])
    except KeyError:
        optim_batch_size = 64

    try:
        optim_learning_rate = float(config['optim_learning_rate'])
    except KeyError:
        optim_learning_rate = 0.01

    try:
        cnn_encoder_layer_sizes = list(map(int, config['cnn_encoder_layer_sizes'].split(',')))
    except KeyError:
        cnn_encoder_layer_sizes = [128]

    try:
        cnn_encoder_kernel_sizes = list(map(int, config['cnn_encoder_kernel_sizes'].split(',')))
    except KeyError:
        cnn_encoder_kernel_sizes = [8]

    try:
        method = config['method']
    except KeyError:
        method = 'basic'

    dataset_prefix = PATH_APP + 'apps/dltsc/datasets/trans/'

    X_train = np.load(dataset_prefix + "train_features.npy")
    y_train = np.load(dataset_prefix + "train_labels.npy")

    X_test = np.load(dataset_prefix + "test_features.npy")
    y_test = np.load(dataset_prefix + "test_labels.npy")

    crop = minute + 1

    if crop is not None:
        X_train = X_train[:, 0:crop, :]
        X_test = X_test[:, 0:crop, :]

    y_train_ext = np.expand_dims(y_train, axis=-1)
    y_test_ext = np.expand_dims(y_test, axis=-1)

    y_train_cat = to_categorical(y_train_ext)
    y_test_cat = to_categorical(y_test_ext)

    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train_cat.shape[1]
    optimizer = keras.optimizers.Adam(lr=optim_learning_rate)

    method_to_call = getattr(arch, method)
    model = method_to_call(n_timesteps, n_features, n_outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', rps_loss])

    print(model.summary())

    dirname = PATH_JOBS_RESULTS + name + "/"

    tun_dir = dirname + "tunning/"
    bp_dir = dirname + "bestparams/"

    params_name = config['minute'] + "_" + config['optim_num_epochs'] + "_" + config['optim_batch_size'] + "_" + \
                  config['optim_learning_rate'] + "_" + config['method']

    if not os.path.exists(dirname):
        os.mkdir(dirname)
        os.mkdir(tun_dir)
        os.mkdir(bp_dir)

    dirname = PATH_JOBS_MODELS + name + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model_file = dirname + params_name + ".h5"

    es = EarlyStopping(monitor='val_rps_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(model_file, monitor='val_rps_loss', mode='min', save_best_only=True, verbose=1)

    #
    hist = model.fit(X_train, y_train_cat, validation_split=0.2, epochs=optim_num_epochs,
                     callbacks=[mc, es], batch_size=optim_batch_size, verbose=1)

    saved_model = load_model(model_file, custom_objects={'rps_loss': rps_loss, 'AttentionLSTM': arch.AttentionLSTM})

    eval_train = saved_model.evaluate(X_train, y_train_cat)
    eval_test = saved_model.evaluate(X_test, y_test_cat)

    tunned = pd.DataFrame(hist.history)

    dirname = PATH_JOBS_RESULTS + name + "/"
    tun_dir = dirname + "tunning/"
    bp_dir = dirname + "bestparams/"

    filename = tun_dir + params_name + ".csv"
    paramfile = bp_dir + params_name + ".json"

    if not os.path.exists(dirname):
        os.mkdir(dirname)
        os.mkdir(tun_dir)
        os.mkdir(bp_dir)

    bp = dict()
    bp['eval_train'] = eval_train
    bp['eval_test'] = eval_test

    with open(paramfile, 'w') as outfile:
        json.dump(bp, outfile)

    tunned.to_csv(filename)


def mlp_by_minute(data, features, prefix, name, config, odds):

    cnn_encoder_kernel_sizes, cnn_encoder_layer_sizes, minute, optim_batch_size, optim_learning_rate, \
        optim_num_epochs, method, dropout = get_config(config)

    features = utils.get_features(features, prefix, minute)

    if odds:
        features.append('odds_p_avg_home')
        features.append('odds_p_avg_draw')
        features.append('odds_p_avg_away')
        features.append('odds_p_std_home')
        features.append('odds_p_std_draw')
        features.append('odds_p_std_away')

    target = 'result'

    train_x, train_y, test_x, test_y = utils.split_train_test(data, features, target, odds)

    def get_target(row):
        if row == 'H':
            return 0
        elif row == 'D':
            return 1
        elif row == 'A':
            return 2

    train_y = train_y.apply(lambda row: get_target(row))
    train_y_cat = to_categorical(train_y)

    test_y = test_y.apply(lambda row: get_target(row))
    test_y_cat = to_categorical(test_y)

    model_label = config['minute'] + "_" + config['method'] + "_" + \
                  config['optim_num_epochs'] + "_" + \
                  config['optim_batch_size'] + "_" + \
                  config['optim_learning_rate'] + "_" + \
                  config['cnn_encoder_layer_sizes'] + "_" + config['optim:dropout_rate']

    method_to_call = getattr(arch, method)

    seed = 7
    cv = 10
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    optimizer = keras.optimizers.Adam(lr=optim_learning_rate)

    evaluations = []
    for i, (idx_train, idx_valid) in enumerate(kfold.split(train_x, train_y)):

        xt, yt = train_x.iloc[idx_train], train_y_cat[idx_train]
        xv, yv = train_x.iloc[idx_valid], train_y_cat[idx_valid]

        if method == 'malstm_fcn':
            model = method_to_call(int(minute+1), len(features), 3)
        else:
            model = method_to_call(len(features), cnn_encoder_layer_sizes, dropout)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', rps_loss])

        result = model.fit(xt, yt, validation_data=(xv, yv), epochs=optim_num_epochs,
                       batch_size=optim_batch_size, verbose=2)

        tunned = pd.DataFrame(result.history)
        evaluations.append(tunned)

    tunned_final = evaluations[0]
    for i in evaluations[1:]:
        tunned_final += i
    tunned_final = tunned_final/cv

    tun_dir, bp_dir = create_results_dir(name)

    tunned_file = tun_dir + model_label + ".csv"
    bp_file = bp_dir + model_label + ".json"

    min_train = tunned_final[tunned_final['val_rps_loss'] == tunned_final['val_rps_loss'].min()]

    n_epochs = int(str(min_train.index.values[0]))+1

    bp = dict()
    bp['epochs'] = n_epochs

    with open(bp_file, 'w') as outfile:
        json.dump(bp, outfile)

    tunned.to_csv(tunned_file)

    model = method_to_call(len(features), cnn_encoder_layer_sizes, dropout)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', rps_loss])

    model.fit(train_x, train_y_cat, epochs=n_epochs,
                       batch_size=optim_batch_size, verbose=2)

    model_dir = create_model_dir(name)
    model_file_json = model_dir + model_label + ".json"
    model_file_h5 = model_dir + model_label + ".h5"

    # saving model
    json_model = model.to_json()
    open(model_file_json, 'w').write(json_model)
    model.save_weights(model_file_h5, overwrite=True)

    eval_test = model.evaluate(test_x, test_y_cat)
    print(eval_test)


def create_results_dir(name):

    dirname = PATH_JOBS_RESULTS + name + "/"
    tun_dir = dirname + "tunning/"
    bp_dir = dirname + "bestparams/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        os.mkdir(tun_dir)
        os.mkdir(bp_dir)

    return tun_dir, bp_dir


def create_model_dir(name):
    classifier_dir = PATH_JOBS_MODELS + name + "/"
    if not os.path.exists(classifier_dir):
        os.mkdir(classifier_dir)

    return classifier_dir



def get_config(config):
    minute = int(config['minute'])
    try:
        optim_num_epochs = int(config['optim_num_epochs'])
    except KeyError:
        optim_num_epochs = 1000
    try:
        optim_batch_size = int(config['optim_batch_size'])
    except KeyError:
        optim_batch_size = 64
    try:
        optim_learning_rate = float(config['optim_learning_rate'])
    except KeyError:
        optim_learning_rate = 0.01
    try:
        cnn_encoder_layer_sizes = list(map(int, config['cnn_encoder_layer_sizes'].split(',')))
    except KeyError:
        cnn_encoder_layer_sizes = [128]
    try:
        cnn_encoder_kernel_sizes = list(map(int, config['cnn_encoder_kernel_sizes'].split(',')))
    except KeyError:
        cnn_encoder_kernel_sizes = [8]

    try:
        method = config['method']
    except KeyError:
        method = None

    try:
        dropout = float(config['optim:dropout_rate'])
    except KeyError:
        dropout = 0

    return cnn_encoder_kernel_sizes, cnn_encoder_layer_sizes, \
           minute, optim_batch_size, optim_learning_rate, optim_num_epochs, method, dropout
