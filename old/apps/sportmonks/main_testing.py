import sys
import pandas as pd

path_project = "/home/igorcosta/soccer/"
sys.path.insert(1, path_project)

from apps.sportmonks import testing
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
    by_minute(config)


def by_minute(config):

    print("...begin...")

    filename = PATH_SPORTMONKS_DATAFRAMES + 'selected_o_trans_final.csv'

    data = pd.read_csv(filename)

    classifier = str(config['classifier'])
    minute = int(config['minute'])
    group = str(config['group'])

    prefix = "trans"

    if group == 'S':
        features = stats_list
        name = classifier + "_stats"

        if 'gnb' in classifier:
            testing.gnb_by_minute(data, features, prefix, name, minute)
        elif 'knn' in classifier:
            testing.knn_by_minute(data, features, prefix, name, minute)
        elif 'xgb' in classifier:
            testing.xgb_by_minute(data, features, prefix, name, minute,  odds=True)
        elif 'cnn' in classifier:
            testing.cnn_by_minute(data, features, prefix, name, minute)
        elif 'mlp' in classifier:
            for i in range(0, 96):
                config['minute'] = str(i)
                testing.mlp_by_minute(data, features, prefix, name, config)


    elif group == 'G':
        features = goals_list
        name = classifier + "_goals"

        if 'gnb' in classifier:
            testing.gnb_by_minute(data, features, prefix, name, minute)
        elif 'knn' in classifier:
            testing.knn_by_minute(data, features, prefix, name, minute)
        elif 'xgb' in classifier:
            testing.xgb_by_minute(data, features, prefix, name, minute,  odds=True)
        elif 'cnn' in classifier:
            testing.cnn_by_minute(data, features, prefix, name, minute)
        elif 'mlp' in classifier:
            for i in range (0,96):
                config['minute'] = str(i)
                testing.mlp_by_minute(data, features, prefix, name, config)


    print("...end...")


if __name__ == "__main__":
    main(sys.argv)


