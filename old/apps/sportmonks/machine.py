import sys
import pandas as pd

path_project = "/home/igorcosta/soccer/"
sys.path.insert(1, path_project)

from apps.sportmonks import learning
from apps.sportmonks import tunning
from core.config import PATH_SPORTMONKS_DATAFRAMES
from apps.sportmonks import trainning

stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']
goals_list = ['goals']


def predicting_gnb_by_minute(minute, group):

    print("...begin...")
    method = 'gnb'

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_v_trans_final.csv'

    types = ['div_counter_trends', 'dif_counter_trends']

    data = pd.read_csv(filename)

    if group == 'S':
        features = stats_list
        name = method + "_stats"
        trainning.gnb_by_minute(data, features, types, name, minute)

    elif group == 'G':
        features = goals_list
        name = method + "_goals"
        trainning.gnb_by_minute(data, features, types, name, minute)

    print("...end...")


def tunning_knn_by_minute(minute, step):

    print("processing...")
    method = 'knn'

    types = ['div_counter_trends', 'dif_counter_trends']

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_v2_final.csv'
    data = pd.read_csv(filename)

    features = stats_list
    name = method + "_stats"
    tunning.knn_by_minute(data, features, types, name, minute, step=step)

    features = goals_list
    name = method + "_goals"
    tunning.knn_by_minute(data, features, types, name, minute, step=step)


def tunning_xgb_by_minute(minute):

    print("processing...")
    method = 'xgb'

    types = ['div_counter_trends', 'dif_counter_trends']

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_v2_final.csv'
    data = pd.read_csv(filename)

    features = stats_list
    name = method + "_stats"
    tunning.xgb_by_minute(data, features, types, name, minute)

    features = goals_list
    name = method + "_goals"
    tunning.xgb_by_minute(data, features, types, name, minute)


def tunning_dnn_by_minute(minute):

    print("processing...")
    method = 'dnn'

    types = ['div_counter_trends', 'dif_counter_trends']

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_v2_final.csv'
    data = pd.read_csv(filename)

    for i in range(0,96):
        features = stats_list
        name = method + "_stats"
        tunning.dnn_by_minute(data, features, types, name, i)

    # features = goals_list
    # name = method + "_goals"
    # tunning.dnn_by_minute(data, features, types, name, minute)


def predicting_knn():

    print("...begin...")
    method = 'knn'

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_v2_final.csv'

    types = ['div_counter_trends', 'dif_counter_trends']

    data = pd.read_csv(filename)

    for minute in range(91, 96):

        features = stats_list
        name = method + "_stats"
        learning.knn_by_minute(data, features, types, name, minute)

        features = goals_list
        name = method + "_goals"
        learning.knn_by_minute(data, features, types, name, minute)
    print("...end...")


def predicting_xgb():
    print("...begin...")
    method = 'xgb'

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_v2_final.csv'

    types = ['div_counter_trends', 'dif_counter_trends']

    data = pd.read_csv(filename)

    for minute in range(0, 96):
        print(minute)
        features = stats_list
        name = method + "_stats"
        learning.xgb_by_minute(data, features, types, name, minute)

        features = goals_list
        name = method + "_goals"
        learning.xgb_by_minute(data, features, types, name, minute)
    print("...end...")


def predicting_dnn():
    print("...begin...")
    method = 'dnn'

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_v2_final.csv'

    types = ['div_counter_trends', 'dif_counter_trends']

    data = pd.read_csv(filename)

    for minute in range(0, 96):
        print(minute)
        features = stats_list
        name = method + "_stats"
        learning.dnn_by_minute(data, features, types, name, minute)

        # features = goals_list
        # name = method + "_goals"
        # learning.xgb_by_minute(data, features, types, name, minute)
    print("...end...")


def tunning_fcn_by_minute(config):

    print("processing...")
    method = 'acnn'

    types = ['div_counter_trends', 'dif_counter_trends']

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_v2_final.csv'
    data = pd.read_csv(filename)

    features = stats_list
    name = method + "_stats"
    tunning.fcn_by_minute(data, features, types, name, config)

    # features = goals_list
    # name = method + "_goals"
    # tunning.dnn_by_minute(data, features, types, name, minute)

config = dict()

classifier = int(sys.argv[1])

if classifier == 'gnb':
    config['group'] = sys.argv[2]
    config['minute'] = int(sys.argv[3])

    predicting_gnb_by_minute(config['group'], config['minute'])
else:
    config['minute'] = int(sys.argv[2])
    config['optim:num_epochs'] = int(sys.argv[3])
    config['optim:batch_size'] = int(sys.argv[4])
    config['optim:learning_rate'] = float(sys.argv[5])
    config['cnn_encoder:layer_sizes'] = list(map(int, sys.argv[6].split(',')))
    config['cnn_encoder:kernel_sizes'] = list(map(int, sys.argv[7].split(',')))


