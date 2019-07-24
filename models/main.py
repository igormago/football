import sys
path_project = "/home/igorcosta/football/"
sys.path.insert(1, path_project)  # to Slurm

from models.setup import Setup
from models.datasets import DatasetFactory
from models.model import ModelFactory

from core.logger import log_in_out, logging
logger = logging.getLogger('Main')


@log_in_out
def main():

    setup = Setup.load()
    logger.info("Setup: " + str(setup))
    dataset = DatasetFactory.load(setup)
    logger.info("Dataset: " + str(dataset))

    model = ModelFactory.load(setup)
    model.train(dataset)
    model.test(dataset)


if __name__ == "__main__":
    main()
