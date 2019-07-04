import logging

from logging import handlers
from core.config import PATH_LOGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh = handlers.TimedRotatingFileHandler(filename=PATH_LOGS + 'futebol.log',
                               when="midnight", interval=1, backupCount=15)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


