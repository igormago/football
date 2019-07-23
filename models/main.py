import sys
path_project = "/home/igorcosta/football/"
sys.path.insert(1, path_project)  # to Slurm

from core.logger import log_in_out
from models import utils
from models.Dataset import Dataset
from models.Model import Model

from core.logger import log_in_out, logging
logger = logging.getLogger('Main')


@log_in_out
def main():

    setup = utils.get_setup()
    logger.info("Setup: " + str(setup))
    data = Dataset(setup)
    model = Model(setup)
    model.train(data)
    model.test(data)


if __name__ == "__main__":
    main()
