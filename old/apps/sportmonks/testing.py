import keras
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from keras import backend as K
import os
import pickle
from keras.models import model_from_json
import json

from apps.sportmonks import utils, arch
from core.config import PATH_JOBS_MODELS, PATH_JOBS_RESULTS

stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']


def rps(y_true, y_pred):

    op1  = (y_pred[0] - y_true[0] + y_pred[1] - y_true[1])**2
    op2  = (y_pred[0] - y_true[0])**2

    return round(0.5*(op1+op2),5)


def rps_loss(y_true, y_pred):

    op1 = K.square(y_pred[:,0] - y_true[:,0] + y_pred[:,1] - y_true[:,1])
    op2 = K.square(y_pred[:,0] - y_true[:,0])

    return K.mean(0.5*(op1+op2),  axis=-1)

def rps_avg(y_true, y_pred):

    op1 = np.square(y_pred[:,0] - y_true['observed_home'] + y_pred[:,1] - y_true['observed_draw'])
    op2 = np.square(y_pred[:,0] - y_true['observed_home'])

    mean = np.mean(0.5*(op1+op2))
    return mean


def rps_avg_gnb(y_true):

    op1 = np.square(y_true['H'] - y_true['observed_home'] + y_true['D'] - y_true['observed_draw'])
    op2 = np.square(y_true['H'] - y_true['observed_home'])

    mean = np.mean(0.5*(op1+op2))
    return mean


def evaluation_gnb(test, pred_proba, classes):

    eval = test.to_frame()

    eval = eval.join(pd.DataFrame(
        pred_proba,
        index=eval.index,
        columns=classes
    ))

    eval['observed_home'] = (eval['result'] == 'H') * 1  # mutiply to 1 to convert boolean in int
    eval['observed_draw'] = (eval['result'] == 'D') * 1
    eval['observed_away'] = (eval['result'] == 'A') * 1

    return eval


def get_features(features, prefix, minute):

    if prefix == "trans":
        columns = ['possession_possession_home_' + str(minute)]
        prefix_trends = ['dif_counter_trends', 'div_counter_trends']
        prefix_cards = ['dif_counter_cards', 'div_counter_cards']

    for stat in features:

        if stat in ['yellow_cards', 'red_cards']:
            for t in prefix_cards:
                label = t + '_' + stat + '_' + str(minute)
                col = label
                columns.append(col)

        elif stat != 'possession':
            for t in prefix_trends:
                label = t + '_' + stat + '_' + str(minute)
                col = label
                columns.append(col)

    return columns


def gnb_by_minute(data, features, prefix, name, minute):

    features = get_features(features, prefix, minute)
    target = 'result'

    train_x, train_y, test_x, test_y = utils.split_train_test(data, features, target)

    dirname = PATH_JOBS_MODELS + name + "/"
    model_file = dirname + str(minute) + ".joblib"
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    pd.options.mode.chained_assignment = None  # default='warn'

    predict(minute, model, name, test_x, test_y, train_x, train_y)


def knn_by_minute(data, features, prefix, name, minute):

    features = get_features(features, prefix, minute)
    target = 'result'

    train_x, train_y, test_x, test_y = utils.split_train_test(data, features, target)

    dirname = PATH_JOBS_MODELS + name + "/"
    model_file = dirname + "2_" + str(minute) + ".joblib"
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    predict_knn(minute, model, name, test_x, test_y, train_x, train_y)


def xgb_by_minute(data, features, prefix, name, minute, odds):

    features = get_features(features, prefix, minute)
    target = 'result'

    if odds:
        features.append('odds_p_avg_home')
        features.append('odds_p_avg_draw')
        features.append('odds_p_avg_away')
        features.append('odds_p_std_home')
        features.append('odds_p_std_draw')
        features.append('odds_p_std_away')

    train_x, train_y, test_x, test_y = utils.split_train_test(data, features, target, odds)

    dirname = PATH_JOBS_MODELS + name + "/"
    model_file = dirname + "1_" + str(minute) + ".joblib"
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    predict_knn(minute, model, name, test_x, test_y, train_x, train_y)


def cnn_by_minute(data, features, prefix, name, minute):

    dirname = PATH_JOBS_RESULTS + name + "/"

    features = get_features(features, prefix, minute)
    target = 'result'

    train_x, train_y, test_x, test_y = utils.split_train_test(data, features, target)

    dirname = PATH_JOBS_MODELS + name + "/"
    model_file = dirname + "2_" + str(minute) + ".joblib"
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    predict_knn(minute, model, name, test_x, test_y, train_x, train_y)


def mlp_by_minute(data, features, prefix, name, config):

    cnn_encoder_kernel_sizes, cnn_encoder_layer_sizes, minute, optim_batch_size, optim_learning_rate, \
    optim_num_epochs, method, dropout = utils.get_config(config)

    features = utils.get_features(features, prefix, minute)
    target = 'result'

    train_x, train_y, test_x, test_y = utils.split_train_test(data, features, target)

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

    model_dir = PATH_JOBS_MODELS + 'mlp_stats' + "/"

    model_label = config['minute'] + "_" + config['method'] + "_" + \
                  config['optim_num_epochs'] + "_" + \
                  config['optim_batch_size'] + "_" + \
                  config['optim_learning_rate'] + "_" + \
                  config['cnn_encoder_layer_sizes'] + "_" + config['optim:dropout_rate']

    model_file_json = model_dir + model_label + ".json"
    model_file_h5 = model_dir + model_label + ".h5"

    # load json and create model
    json_file = open(model_file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_file_h5)

    optimizer = Adam(lr=optim_learning_rate)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', rps_loss])

    eval_train = loaded_model.evaluate(train_x, train_y_cat)
    eval_test = loaded_model.evaluate(test_x, test_y_cat)

    print(eval_train)
    print(eval_test)
    met = pd.DataFrame()
    met["acc_train"] = [eval_train[1]]
    met["rps_train"] = [eval_train[2]]
    met["acc_test"] = [eval_test[1]]
    met["rps_test"] = [eval_test[2]]

    dirname = PATH_JOBS_RESULTS + name + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    results_file = dirname + str(minute) + ".csv"

    met.to_csv(results_file)
    print(met)


def predict(minute, model, name, test_x, test_y, train_x, train_y):

    train_pred_proba = model.predict_proba(train_x)
    test_pred_proba = model.predict_proba(test_x)
    train_y_true = evaluation_gnb(train_y, train_pred_proba, model.classes_)
    test_y_true = evaluation_gnb(test_y, test_pred_proba, model.classes_)
    met = pd.DataFrame()
    met["acc_train"] = [model.score(train_x, train_y)]
    met["rps_train"] = [rps_avg_gnb(train_y_true)]
    met["acc_test"] = [model.score(test_x, test_y)]
    met["rps_test"] = [rps_avg_gnb(test_y_true)]
    dirname = PATH_JOBS_RESULTS + name + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    results_file = dirname + str(minute) + ".csv"
    met.to_csv(results_file)


def predict_knn(minute, model, name, test_x, test_y, train_x, train_y):

    train_pred_proba = model.predict_proba(train_x)
    test_pred_proba = model.predict_proba(test_x)

    train_y_true = evaluation_gnb(train_y, train_pred_proba, model.classes_)
    test_y_true = evaluation_gnb(test_y, test_pred_proba, model.classes_)

    met = pd.DataFrame()
    met["acc_train"] = [model.score(train_x, train_y)]
    met["rps_train"] = [rps_avg_gnb(train_y_true)]
    met["acc_test"] = [model.score(test_x, test_y)]
    met["rps_test"] = [rps_avg_gnb(test_y_true)]

    dirname = PATH_JOBS_RESULTS + name + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    results_file = dirname + str(minute) + ".csv"
    met.to_csv(results_file)

