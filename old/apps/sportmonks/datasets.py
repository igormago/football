import sys
path_project = "/home/igorcosta/soccer/"
sys.path.insert(1, path_project)
import pandas as pd
from core.databases import SportMonksDatabase as DataBase
from datetime import datetime
from pandas.io.json import json_normalize
from core.config import PATH_SPORTMONKS_DATAFRAMES
import numpy as np

session = DataBase.get_session()
stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']


def create_frames(prefix, selection, projection):

    skip = 0
    num_matches = session.get_collection('matches').count(selection)
    print(num_matches)
    print(selection)
    keep = 5000
    file_number = 1

    while skip < num_matches:

        print(skip)
        print(datetime.now())
        data = session.get_collection('matches').find(selection, projection).skip(skip).limit(keep)
        skip = skip + keep
        print(datetime.now())
        df = pd.DataFrame(json_normalize(list(data), sep='_'))
        print(datetime.now())

        name = prefix + "_" + str(file_number) + ".csv"
        file_number += 1
        filename = PATH_SPORTMONKS_DATAFRAMES + name
        df.to_csv(filename)
        print(datetime.now())


def create_partial_dataframe(prefix):

    first_file = prefix + "_1.csv"
    filename = PATH_SPORTMONKS_DATAFRAMES + first_file
    df = pd.read_csv(filename)

    file_number = 2
    have_other_file = True
    while have_other_file:

        print(file_number)
        try:
            name = prefix + '_' + str(file_number) + '.csv'
            filename = PATH_SPORTMONKS_DATAFRAMES + name
            df_temp = pd.read_csv(filename)
            df = pd.concat([df, df_temp], sort=True)

        except Exception as ex:
            have_other_file = False
        file_number += 1

    name = prefix + "_partial.csv"
    filename = PATH_SPORTMONKS_DATAFRAMES + name
    df.to_csv(filename)


def create_final_dataframe(prefix, odds=False):

    name = prefix + "_partial.csv"
    filename = PATH_SPORTMONKS_DATAFRAMES + name
    df = pd.read_csv(filename)

    if 'counter' in prefix:
        features = process_counter(df)
    elif 'trans' in prefix:
        features = process_trans(df, odds)
    else:
        features = process_ts(df)

    df = df[df['odds_p_avg_home'].notnull()]
    df = df[df['odds_p_avg_draw'].notnull()]
    df = df[df['odds_p_avg_away'].notnull()]
    df = df[df['odds_p_std_home'].notnull()]
    df = df[df['odds_p_std_draw'].notnull()]
    df = df[df['odds_p_std_away'].notnull()]

    df_final = df[features]

    name = prefix + "_final.csv"
    filename = PATH_SPORTMONKS_DATAFRAMES + name
    df_final = df_final.sort_values(by='date')
    df_final.to_csv(filename)


def process_counter(df):

    features = ['date', 'observed_away', 'observed_draw', 'observed_home', 'result']
    for i in range(0, 95):
        for j in ['yellow_cards', 'red_cards']:
            for z in ['home', 'away']:
                t = 'counter_cards'
                col = t + "_" + j + "_" + z + "_" + str(i)
                next_col = t + "_" + j + "_" + z + "_" + str(i+1)
                df[next_col].fillna(df[col], inplace=True)
                features.append(col)
                if i == 94:
                    features.append(next_col)

    for i in range(0, 95):
        for j in ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']:
            for z in ['home', 'away']:
                t = 'counter_trends'
                col = t + "_" + j + "_" + z + "_" + str(i)
                next_col = t + "_" + j + "_" + z + "_" + str(i + 1)
                df[next_col].fillna(df[col], inplace=True)
                features.append(col)
                if i == 94:
                    features.append(next_col)

    for i in range(0, 95):
        for z in ['possession_home', 'possession_away']:
            j = 'possession'
            col = j + "_" + z + "_" + str(i)
            next_col = j + "_" + z + "_" + str(i + 1)
            df[next_col].fillna(df[col], inplace=True)
            features.append(col)
            if i == 94:
                features.append(next_col)

    return features


def process_ts(df):

    features = ['date', 'observed_away', 'observed_draw', 'observed_home', 'result']
    for i in range(0, 95):
        for j in ['yellow_cards', 'red_cards']:
            for z in ['home', 'away']:
                t = 'cards'
                col = t + "_" + j + "_" + z + "_" + str(i)
                next_col = t + "_" + j + "_" + z + "_" + str(i + 1)
                df[next_col].fillna(df[col], inplace=True)
                features.append(col)
                if i == 94:
                    features.append(next_col)

    for i in range(0, 95):
        for j in ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']:
            for z in ['home', 'away']:
                t = 'trends'
                col = t + "_" + j + "_" + z + "_" + str(i)
                next_col = t + "_" + j + "_" + z + "_" + str(i + 1)
                df[next_col].fillna(df[col], inplace=True)
                features.append(col)
                if i == 94:
                    features.append(next_col)

    for i in range(0, 95):
        for z in ['possession_home', 'possession_away']:
            j = 'possession'
            col = j + "_" + z + "_" + str(i)
            next_col = j + "_" + z + "_" + str(i + 1)
            df[next_col].fillna(df[col], inplace=True)
            features.append(col)
            if i == 94:
                features.append(next_col)

    return features


def process_trans(df, odds=False):

    features = ['date', 'observed_away', 'observed_draw', 'observed_home', 'result']
    if odds:
        features.append('odds_p_avg_home')
        features.append('odds_p_avg_draw')
        features.append('odds_p_avg_away')
        features.append('odds_p_std_home')
        features.append('odds_p_std_draw')
        features.append('odds_p_std_away')

    for i in range(0, 95):
        for t in ['dif', 'div']:
            for z in ['yellow_cards', 'red_cards']:
                j = 'counter_cards'
                col = t + "_" + j + "_" + z + "_" + str(i)
                next_col = t + "_" + j + "_" + z + "_" + str(i + 1)
                df[next_col].fillna(df[col], inplace=True)
                features.append(col)
                if i == 94:
                    features.append(next_col)

    for i in range(0, 95):
        for t in ['dif', 'div']:
            for z in ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']:
                j = 'counter_trends'
                col = t + "_" + j + "_" + z + "_" + str(i)
                next_col = t + "_" + j + "_" + z + "_" + str(i + 1)
                df[next_col].fillna(df[col], inplace=True)
                features.append(col)
                if i == 94:
                    features.append(next_col)

    for i in range(0, 95):
        for z in ['possession_home', 'possession_away']:
            j = 'possession'
            col = j + "_" + z + "_" + str(i)
            next_col = j + "_" + z + "_" + str(i + 1)
            df[next_col].fillna(df[col], inplace=True)
            features.append(col)
            if i == 94:
                features.append(next_col)

    return features


def create_numpy_dataframe(prefix):

    name = prefix + "_final.csv"
    filename = PATH_SPORTMONKS_DATAFRAMES + name
    df = pd.read_csv(filename)

    df_train = df[:40000]
    df_test = df[40001:]

    if prefix == 'selected_v_counter':
        print("1")
        X_train = numpy_counter(df_train)
        print("2")
        X_test = numpy_counter(df_test)

    elif prefix == 'selected_v_trans':
        X_train = numpy_trans(df_train)
        X_test = numpy_trans(df_test)

    else:
        X_train = numpy_ts(df_train)
        X_test = numpy_ts(df_test)

    y_train = numpy_target(df_train)
    y_test = numpy_target(df_test)

    np_files = dict()
    np_files['train_features'] = np.asarray(X_train, dtype=np.float32)
    np_files['train_labels'] = np.asarray(y_train, dtype=np.float32)
    np_files['test_features'] = np.asarray(X_test, dtype=np.float32)
    np_files['test_labels'] = np.asarray(y_test, dtype=np.float32)

    for key in np_files:
        name = prefix + "_" + key + ".npy"
        filename = PATH_SPORTMONKS_DATAFRAMES + name
        np.save(filename, np_files[key])


def numpy_counter(df):

    total = list()

    for idx, row in enumerate(df.iterrows()):

        obj = row[1]
        row_array = list()

        for j in ['yellow_cards', 'red_cards']:
            for z in ['home', 'away']:
                t = 'counter_cards'
                farray = list()
                for i in range(0, 96):
                    col = t + "_" + j + "_" + z + "_" + str(i)
                    farray.append(obj[col])
                row_array.append(farray)

        for j in ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']:
            for z in ['home', 'away']:
                t = 'counter_trends'
                farray = list()
                for i in range(0, 96):
                    col = t + "_" + j + "_" + z + "_" + str(i)
                    farray.append(obj[col])
                row_array.append(farray)

        for z in ['possession_home', 'possession_away']:
            j = 'possession'
            farray = list()
            for i in range(0, 96):
                col = j + "_" + z + "_" + str(i)
                farray.append(obj[col])
            row_array.append(farray)

        row_array = list(map(list, zip(*row_array)))

        total.append(row_array)

    return total


def numpy_ts(df):

    total = list()

    for idx, row in enumerate(df.iterrows()):

        obj = row[1]
        row_array = list()

        for j in ['yellow_cards', 'red_cards']:
            for z in ['home', 'away']:
                t = 'cards'
                farray = list()
                for i in range(0, 96):
                    col = t + "_" + j + "_" + z + "_" + str(i)
                    farray.append(obj[col])
                row_array.append(farray)

        for j in ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']:
            for z in ['home', 'away']:
                t = 'trends'
                farray = list()
                for i in range(0, 96):
                    col = t + "_" + j + "_" + z + "_" + str(i)
                    farray.append(obj[col])
                row_array.append(farray)

        for z in ['possession_home', 'possession_away']:
            j = 'possession'
            farray = list()
            for i in range(0, 96):
                col = j + "_" + z + "_" + str(i)
                farray.append(obj[col])
            row_array.append(farray)

        row_array = list(map(list, zip(*row_array)))

        total.append(row_array)

    return total

def numpy_trans(df):

    total = list()

    for idx, row in enumerate(df.iterrows()):

        obj = row[1]
        row_array = list()

        for t in ['dif', 'div']:
            for z in ['yellow_cards', 'red_cards']:
                farray = list()
                for i in range(0, 96):
                    j = 'counter_cards'
                    col = t + "_" + j + "_" + z + "_" + str(i)
                    farray.append(obj[col])
                row_array.append(farray)

        for t in ['dif', 'div']:
            for z in ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']:
                farray = list()
                for i in range(0, 96):
                    j = 'counter_trends'
                    col = t + "_" + j + "_" + z + "_" + str(i)
                    farray.append(obj[col])
                row_array.append(farray)

        for z in ['possession_home']:
            farray = list()
            for i in range(0, 96):
                j = 'possession'
                col = j + "_" + z + "_" + str(i)
                farray.append(obj[col])
            row_array.append(farray)

        row_array = list(map(list, zip(*row_array)))

        total.append(row_array)

    return total


def numpy_target(df):

    total = list()

    for idx, row in enumerate(df.iterrows()):

        obj = row[1]
        if obj['result'] == 'H':
            total.append(0)
        elif obj['result'] == 'D':
            total.append(1)
        else:
            total.append(2)

    return total

def run():

    filter_ts = {'observed': 1, 'date': 1, 'result': 1, 'trends': 1, 'cards': 1, 'possession': 1}
    filter_counter = {'observed': 1, 'date': 1, 'result': 1, 'counter_trends': 1, 'counter_cards': 1, 'possession': 1}
    filter_trans = {'observed': 1, 'date': 1, 'result': 1, 'dif_counter_trends': 1, 'dif_counter_cards': 1,
                    'div_counter_trends': 1, 'div_counter_cards': 1,
                    'possession': 1}

    filter_trans_o = {'observed': 1, 'date': 1, 'result': 1, 'dif_counter_trends': 1, 'dif_counter_cards': 1,
                    'div_counter_trends': 1, 'div_counter_cards': 1,
                    'possession': 1, 'odds': 1}

    selection = {'selected':1, 'odds': {'$exists':1}}

    #create_final_dataframe('selected_v_counter')
    #create_frames('selected_o_trans', selection, filter_trans_o)
    create_final_dataframe('selected_o_trans', odds=True)
    #create_final_dataframe('selected_o_trans', odds= True)

    #create_numpy_dataframe('selected_v_ts')


run()