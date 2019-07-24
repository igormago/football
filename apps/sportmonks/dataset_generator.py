import os
import shutil
import pandas as pd
import pymongo

from core.logger import logging, log_in_out
from pandas.io.json import json_normalize
from argparse import ArgumentParser
from core.config_loader import SysConfig, Database

logger = logging.getLogger('dataset_generator')
session = Database.get_session_sportmonks()

ALL = 0
PARTIAL = 1
SINGLE = 2
FINAL = 3
TRANSFORM = 4


def create_partial_frames(setup, selection, projection):

    partial_size = 1000
    idx_from = 0
    idx_to = 0
    num_matches = session.get_collection('matches').count_documents(selection)

    dataset_dir = os.path.join(SysConfig.path('datasets'), setup.dataset_id)
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


@log_in_out
def create_single_dataframe(setup):

    partial_dir = os.path.join(SysConfig.path('datasets'), setup.dataset_id, 'partial')
    files = sorted(os.listdir(partial_dir))

    print('Concatenating file %s' % files[0])
    file_path = os.path.join(partial_dir, files[0])
    df = pd.read_csv(file_path)

    for f in files[1:]:

        print('Concatenating file %s' % f)
        file_path = os.path.join(partial_dir, f)
        df_temp = pd.read_csv(file_path)
        df = pd.concat([df, df_temp], sort=True)

    df_file_name = 'single.csv'
    df_file_path = os.path.join(SysConfig.path('datasets'), setup.dataset_id, df_file_name)
    df.to_csv(df_file_path)


@log_in_out
def create_final_dataframe(setup):

    file_path = os.path.join(SysConfig.path('datasets'), setup.dataset_id, 'single.csv')
    df = pd.read_csv(file_path)

    features = list()
    if 'sm_accum_odds' in setup.dataset_id:
        features = process_accum(df)
    elif 'sm_comp_odds' in setup.dataset_id:
        features = process_comp(df)

    df = df[features]

    df_file_path = os.path.join(SysConfig.path('datasets'), setup.dataset_id, 'final.csv')
    df.to_csv(df_file_path)


@log_in_out
def transform_minutes_into_feature(setup):

    file_path = os.path.join(SysConfig.path('datasets'), setup.dataset_id, 'final.csv')
    df = pd.read_csv(file_path)

    features_basic = ['id', 'minute_max', 'date', 'observed_away', 'observed_draw', 'observed_home', 'result']
    features_odds = ['odds_p_mean_home', 'odds_p_mean_draw', 'odds_p_mean_away',
                     'odds_p_std_home', 'odds_p_std_draw', 'odds_p_std_away']

    features_trends = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']
    features_cards = ['yellow_cards', 'red_cards']
    features_ratio = ['possession']

    locales = ['home', 'away']

    features_columns = []
    if 'sm_accum_odds' in setup.dataset_id:
        group = 'accum_trends'
        for feature in features_trends:
            for l in locales:
                col = '_'.join([group, feature, l])
                features_columns.append(col)

        group = 'accum_cards'
        for feature in features_cards:
            for l in locales:
                col = '_'.join([group, feature, l])
                features_columns.append(col)

        group = 'ratio_trends'
        for feature in features_ratio:
            col = '_'.join([group, feature])
            features_columns.append(col)

    elif 'sm_comp_odds' in setup.dataset_id:
        comp = ['sub', 'ratio']
        group = 'trends'
        for feature in features_trends + features_ratio:
            for c in comp:
                col = '_'.join([c, group, feature])
                features_columns.append(col)

        group = 'cards'
        for feature in features_cards:
            for c in comp:
                col = '_'.join([c, group, feature])
                features_columns.append(col)

    df_new = pd.DataFrame()
    for i in range(96):
        logger.debug("Extracting minute %i" % i)
        features_columns_by_minute = ['_'.join([a, str(i)]) for a in features_columns]
        df_partial = df[features_basic + features_odds + features_columns_by_minute]
        df_partial.columns = features_basic + features_odds + features_columns
        df_partial.loc[:, 'minute'] = i
        df_new = pd.concat([df_new, df_partial]).sort_index(kind='merge')

    setup.dataset_id = '_'.join([setup.dataset_id, 'min'])
    dataset_dir = os.path.join(SysConfig.path('datasets'), setup.dataset_id)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    df_file_path = os.path.join(dataset_dir, 'final.csv')
    df_new.to_csv(df_file_path)


@log_in_out
def get_selection_projection(setup):

    selection_with_odds = {'selected': True, 'with_odds': True, 'minute_max': {'$gt': 89, '$lt': 96}}
    selection_without_odds = {'selected': 1}

    if setup.dataset_id == 'sm_counter':
        projection = {'observed': 1, 'date': 1, 'result': 1, 'counter_trends': 1,
                      'counter_cards': 1, 'possession': 1}

        return selection_without_odds, projection

    elif setup.dataset_id == 'sm_trans':
        projection = {'observed': 1, 'date': 1, 'result': 1, 'counter_trends': 1,
                      'counter_cards': 1, 'possession': 1}

        return selection_without_odds, projection

    elif setup.dataset_id == 'sm_accum_odds':

        projection = {'id': 1, 'minute_max': 1, 'observed': 1, 'date': 1, 'result': 1, 'accum_trends': 1, 'accum_cards': 1,
                      'ratio_trends': 1, 'odds': 1}

        return selection_with_odds, projection

    elif setup.dataset_id == 'sm_comp_odds':

        projection = {'id': 1, 'minute_max': 1, 'observed': 1, 'date': 1, 'result': 1, 'sub_trends': 1, 'sub_cards': 1,
                      'ratio_trends': 1, 'ratio_cards': 1, 'odds': 1}

        return selection_with_odds, projection


def process_accum(df):

    def processing_by_local():

        for i in range(0, 95):
            for feature in subgroups:
                for local in locales:

                    col = '_'.join([group, feature, local, str(i)])
                    next_col = '_'.join([group, feature, local, str(i + 1)])
                    df[next_col].fillna(df[col], inplace=True)
                    features.append(col)
                    if i == 94:
                        features.append(next_col)

    def processing():

        for i in range(0, 95):
            for feature in subgroups:

                    col = '_'.join([group, feature, str(i)])
                    next_col = '_'.join([group, feature, str(i + 1)])
                    df[next_col].fillna(df[col], inplace=True)
                    features.append(col)
                    if i == 94:
                        features.append(next_col)

    features_basic = ['id', 'minute_max', 'date', 'observed_away', 'observed_draw', 'observed_home', 'result']
    features_odds = ['odds_p_mean_home', 'odds_p_mean_draw', 'odds_p_mean_away',
                     'odds_p_std_home', 'odds_p_std_draw', 'odds_p_std_away']
    features = features_basic + features_odds

    logger.debug('Transforming: cards')
    group = 'accum_cards'
    subgroups = ['yellow_cards', 'red_cards']
    locales = ['home', 'away']
    processing_by_local()

    logger.debug('Transforming: trends')
    group = 'accum_trends'
    subgroups = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']
    processing_by_local()

    logger.debug('Transforming: ball possession')
    group = 'ratio_trends'
    subgroups = ['possession']
    processing()
    return features


def process_comp(df):

    def processing():

        for i in range(0, 95):
            for feature in subgroups:
                for c in comp:
                    col = '_'.join([c, group, feature, str(i)])
                    next_col = '_'.join([c, group, feature, str(i + 1)])
                    df[next_col].fillna(df[col], inplace=True)
                    features.append(col)
                    if i == 94:
                        features.append(next_col)

    features_basic = ['id', 'minute_max', 'date', 'observed_away', 'observed_draw', 'observed_home', 'result']
    features_odds = ['odds_p_mean_home', 'odds_p_mean_draw', 'odds_p_mean_away',
                     'odds_p_std_home', 'odds_p_std_draw', 'odds_p_std_away']
    features = features_basic + features_odds
    comp = ['sub', 'ratio']

    logger.debug('Transforming: cards')
    group = 'cards'
    subgroups = ['yellow_cards', 'red_cards']
    processing()

    logger.debug('Transforming: trends')
    group = 'trends'
    subgroups = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']
    processing()

    return features


def create_dataset_complete(setup):
    selection, projection = get_selection_projection(setup)
    create_partial_frames(setup, selection, projection)
    create_single_dataframe(setup)
    create_final_dataframe(setup)
    transform_minutes_into_feature(setup)


def main():

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset_id",
                        help="inform the id of dataset")

    parser.add_argument("-a", "--action", dest="action", choices=[PARTIAL, SINGLE, FINAL, ALL, TRANSFORM],
                        help="1- partial | 2- single | 3- final | 0- all steps ", type=int)

    parser.add_argument("-t", "--transform", action="store_true", dest="transform", default=False,
                        help="inform to transform minutes in feature")

    setup = parser.parse_args()

    if setup.action == PARTIAL:
        selection, projection = get_selection_projection(setup)
        create_partial_frames(setup, selection, projection)
    elif setup.action == SINGLE:
        create_single_dataframe(setup)
    elif setup.action == FINAL:
        create_final_dataframe(setup)
    elif setup.action == ALL:
        create_dataset_complete(setup)
    elif setup.action == TRANSFORM:
        transform_minutes_into_feature(setup)


if __name__ == "__main__":
    main()
