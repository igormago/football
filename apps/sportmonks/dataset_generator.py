import os
import shutil
import pandas as pd
import pymongo

from pandas.io.json import json_normalize
from argparse import ArgumentParser
from core.config_loader import Config, Database

session = Database.get_session_sportmonks()

ALL = 0
PARTIAL = 1
SINGLE = 2
FINAL = 3


def create_partial_frames(config, selection, projection):

    partial_size = 1000
    idx_from = 0
    idx_to = 0
    num_matches = session.get_collection('matches').count_documents(selection)

    dataset_dir = os.path.join(Config.path('datasets'), config.dataset_id)
    partial_dir = os.path.join(dataset_dir, 'partial')

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        os.makedirs(partial_dir)
    else:
        shutil.rmtree(partial_dir)
        os.makedirs(partial_dir)

    idx_file = 1
    while idx_from < num_matches:

        idx_file_str = str(idx_file).zfill(4)

        idx_to = idx_to + partial_size
        print('Creating partial dataframe number [%s] from index %i to %i ' % (idx_file_str, idx_from, idx_to))

        data = session.get_collection('matches').find(selection, projection).\
            skip(idx_from).limit(partial_size).sort([("date", pymongo.ASCENDING)])

        df = pd.DataFrame(json_normalize(list(data), sep='_'))

        file_name = '_'.join([idx_file_str, str(idx_from), str(idx_to), '.csv'])
        file_path = os.path.join(partial_dir, file_name)
        df.to_csv(file_path)

        idx_from += partial_size
        idx_file += 1


def create_single_dataframe(config):

    partial_dir = os.path.join(Config.path('datasets'), config.dataset_id, 'partial')
    files = sorted(os.listdir(partial_dir))

    print('Concatenating file %s' % files[0])
    file_path = os.path.join(partial_dir, files[0])
    df = pd.read_csv(file_path)

    for f in files[1:]:

        print('Concatenating file %s' % f)
        file_path = os.path.join(partial_dir, f)
        df_temp = pd.read_csv(file_path)
        df = pd.concat([df, df_temp], sort=True)

    df_file_name = 'a.csv'
    df_file_path = os.path.join(Config.path('datasets'), config.dataset_id, df_file_name)
    df.to_csv(df_file_path)


def create_final_dataframe(config):

    file_path = os.path.join(Config.path('datasets'), config.dataset_id, 'df_single.csv')
    df = pd.read_csv(file_path)

    features = list()
    if 'sm_counter_odds' in config.dataset_id:
        features = process_counter(df)

    df = df[features]

    df_file_name = 'df_final.csv'
    df_file_path = os.path.join(Config.path('datasets'), config.dataset_id, df_file_name)
    df.to_csv(df_file_path)


def get_selection_projection(config):

    selection_with_odds = {'selected': 1, 'odds': {'$exists': 1}}
    selection_without_odds = {'selected': 1}

    if config.dataset_id == 'sm_counter':
        projection = {'observed': 1, 'date': 1, 'result': 1, 'counter_trends': 1,
                      'counter_cards': 1, 'possession': 1}

        return selection_without_odds, projection

    elif config.dataset_id == 'sm_trans':
        projection = {'observed': 1, 'date': 1, 'result': 1, 'counter_trends': 1,
                      'counter_cards': 1, 'possession': 1}

        return selection_without_odds, projection

    elif config.dataset_id == 'sm_counter_odds':

        projection = {'observed': 1, 'date': 1, 'result': 1, 'counter_trends': 1, 'counter_cards': 1,
                      'possession': 1, 'odds': 1}

        return selection_with_odds, projection


def process_counter(df):

    def preprocessing_dataframe():

        for i in range(0, 95):
            for feature in subgroups:
                for local in locales:

                    col = '_'.join([group, feature, local, str(i)])
                    next_col = '_'.join([group, feature, local, str(i + 1)])
                    df[next_col].fillna(df[col], inplace=True)
                    features.append(col)
                    if i == 94:
                        features.append(next_col)

    features_basic = ['date', 'observed_away', 'observed_draw', 'observed_home', 'result']
    features_odds = ['odds_p_avg_home', 'odds_p_avg_draw', 'odds_p_avg_away',
                     'odds_p_std_home', 'odds_p_std_draw', 'odds_p_std_away']
    features = features_basic + features_odds
    print(features)

    print('Transforming: cards')
    group = 'accum_cards'
    subgroups = ['yellow_cards', 'red_cards']
    locales = ['home', 'away']
    preprocessing_dataframe()

    print('Transforming: trends')
    group = 'accum_trends'
    subgroups = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']
    preprocessing_dataframe()

    print('Transforming: ball possession')
    group = 'ratio_ball_possession'
    subgroups = ['possession']
    preprocessing_dataframe()

    return features


def create_dataset_complete(config):
    selection, projection = get_selection_projection(config)
    create_partial_frames(config, selection, projection)
    create_single_dataframe(config)
    create_final_dataframe(config)


def main():

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset_id",
                        help="inform the id of dataset")

    parser.add_argument("-a", "--action", dest="action", choices=[PARTIAL, SINGLE, FINAL, ALL],
                        help="1- partial | 2- single | 3- final | 0- all steps ", type=int)

    config = parser.parse_args()

    if config.action == PARTIAL:
        selection, projection = get_selection_projection(config)
        create_partial_frames(config, selection, projection)
    elif config.action == SINGLE:
        create_single_dataframe(config)
    elif config.action == FINAL:
        create_final_dataframe(config)
    elif config.action == ALL:
        create_dataset_complete(config)


if __name__ == "__main__":
    main()
