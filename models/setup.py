from argparse import ArgumentParser
from core.logger import logging

logger = logging.getLogger('utils')


class Setup:

    @staticmethod
    def load():

        parser = ArgumentParser()
        parser.add_argument("-c", "--classifier", dest="classifier",
                            help="inform the name of classifier")

        parser.add_argument("-d", "--dataset", dest="dataset_id",
                            help="inform the name of dataset")

        parser.add_argument("-au", "--tuning",
                            action="store_true", dest="is_to_tune", default=False,
                            help="tunning the model")

        parser.add_argument("-ar", "--training",
                            action="store_true", dest="is_to_train", default=False,
                            help="training the model")

        parser.add_argument("-t", "--minute", dest="minute", default=0, help="inform the minute", type=int)

        parser.add_argument("-f", "--features", dest="features", help="inform the method to get features")

        parser.add_argument("-gt", "--grid_type", dest="grid_type", help="inform the type of grid: 'grid' or'random'")

        parser.add_argument("-gp", "--grid_params", dest="grid_params", help="inform the method to get grid params")

        parser.add_argument("-ts", "--train_size", dest="train_size", default=40000, help="inform the train size", type=int)

        parser.add_argument("-tf", "--by_minute",
                            action="store_true", dest="is_minute_feature", default=False,
                            help="use minute as a feature")

        parser.add_argument("-env", "--virtualenv", dest="virtualenv", default='TEST',
                            help="define the virtualenv")

        setup = parser.parse_args()

        return setup


