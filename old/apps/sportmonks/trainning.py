from sklearn.naive_bayes import GaussianNB
import os
import pickle

from apps.sportmonks import testing, utils
from core.config import PATH_JOBS_MODELS


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

    model = GaussianNB()
    model.fit(train_x, train_y)

    dirname = PATH_JOBS_MODELS + name + "/"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model_file = dirname + str(minute) + ".joblib"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    evaluate = True
    if evaluate:
        testing.predict(minute, model, name, test_x, test_y, train_x, train_y)


