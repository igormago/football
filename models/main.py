from argparse import ArgumentParser

from prediction import training

TUNING = 1
TRAINING = 2
TESTING = 3


def main():

    parser = ArgumentParser()
    parser.add_argument("-c", "--classifier", dest="classifier",
                        help="inform the name of classifier")

    parser.add_argument("-a", "--action", dest="action", choices=[TUNING, TRAINING, TESTING],
                        help="1- tuning | 2- training | 3 = test ", type=int)

    parser.add_argument("-q", "--quiet",
                        action="store_false", dest="verbose", default=True,
                        help="don't print status messages to stdout")

    config = parser.parse_args()

    if config.action == TUNING:
        training.run(config)
    elif config.action == TRAINING:
        pass
    elif config.action == TESTING:
        print("entrou")


if __name__ == "__main__":
    main()
