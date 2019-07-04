import sys
import pandas as pd

path_project = "/home/igorcosta/soccer/"
sys.path.insert(1, path_project)

from apps.sportmonks import tunning
from core.config import PATH_SPORTMONKS_DATAFRAMES

stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks',
              'possession', 'yellow_cards', 'red_cards']
goals_list = ['goals']

def main(argv):

    print("MAIN")
    full_arguments = argv
    argument_list = full_arguments[1:]

    print(argument_list)
    config = dict()
    for arg in argument_list:
        values = arg.split('=')
        config[values[0]] = values[1]

    print(config)
    by_minute_odds(config)


def by_minute(config):

    print("...begin...")

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_v_trans_final.csv'

    data = pd.read_csv(filename)

    classifier = str(config['classifier'])
    minute = int(config['minute'])
    group = str(config['group'])
    step = int(config['step'])

    prefix = "trans"

    if group == 'S':
        features = stats_list
        name = classifier + "_stats"
        if 'knn' in classifier:
            tunning.knn_by_minute(data, features, prefix, name, minute, step)
        elif 'xgb' in classifier:
            tunning.xgb_by_minute(data, features, prefix, name, minute, step)
        elif 'cnn' in classifier:
            tunning.cnn_by_minute(data, features, prefix, name, config)
        elif 'dnn' in classifier:
            tunning.dnn_by_minute(data, features, prefix, name, config)
        elif 'mlp' in classifier:
            tunning.mlp_by_minute(data, features, prefix, name, config)

    elif group == 'G':
        features = goals_list
        name = classifier + "_goals"
        if 'knn' in classifier:
            tunning.knn_by_minute(data, features, prefix, name, minute, step)
        elif 'xgb' in classifier:
            tunning.xgb_by_minute(data, features, prefix, name, minute, step)
        elif 'cnn' in classifier:
            tunning.cnn_by_minute(data, features, prefix, name, config)
        elif 'dnn' in classifier:
            tunning.dnn_by_minute(data, features, prefix, name, config)
        elif 'mlp' in classifier:
            tunning.mlp_by_minute(data, features, prefix, name, config)

    print("...end...")


def by_minute_odds(config):

    print("...begin...")

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_o_trans_final.csv'

    data = pd.read_csv(filename)

    classifier = str(config['classifier'])
    minute = int(config['minute'])
    group = str(config['group'])
    step = int(config['step'])

    prefix = "trans"

    if group == 'S':
        features = stats_list

        name = classifier + "_stats"
        if 'knn' in classifier:
            tunning.knn_by_minute(data, features, prefix, name, minute, step)
        elif 'xgb' in classifier:
            tunning.xgb_by_minute(data, features, prefix, name, minute, step, odds=True)
        elif 'cnn' in classifier:
            tunning.cnn_by_minute(data, features, prefix, name, config)
        elif 'dnn' in classifier:
            tunning.dnn_by_minute(data, features, prefix, name, config)
        elif 'mlp' in classifier:
            tunning.mlp_by_minute(data, features, prefix, name, config, odds=True)

    elif group == 'G':
        features = goals_list

        name = classifier + "_goals"
        if 'knn' in classifier:
            tunning.knn_by_minute(data, features, prefix, name, minute, step)
        elif 'xgb' in classifier:
            tunning.xgb_by_minute(data, features, prefix, name, minute, step, odds=True)
        elif 'cnn' in classifier:
            tunning.cnn_by_minute(data, features, prefix, name, config)
        elif 'dnn' in classifier:
            tunning.dnn_by_minute(data, features, prefix, name, config)
        elif 'mlp' in classifier:
            tunning.mlp_by_minute(data, features, prefix, name, config, odds=True)

    print("...end...")


if __name__ == "__main__":
    main(sys.argv)


