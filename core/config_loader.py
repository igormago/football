# config_loader.py
from configparser import SafeConfigParser, ConfigParser, ExtendedInterpolation
import os

from pymongo import MongoClient


def get_parser():

    parser = ConfigParser(interpolation=ExtendedInterpolation())

    file_path = os.path.dirname(os.path.abspath(__file__))
    file_conf = os.path.join(file_path, 'config.ini')
    parser.read(file_conf)
    print("entrou")
    return parser


class Config:
    """Interact with configuration variables."""

    parser = get_parser()

    @classmethod
    def path(cls, key):
        """Get PATH values from config.ini."""
        return cls.parser.get('PATH', key)

    @classmethod
    def database(cls, key):
        """Get PATH values from config.ini."""
        return cls.parser.get('DATABASE', key)


class Database:
    """Interact with configuration variables."""
    sm_session = MongoClient(host=Config.database('mongo_host'),
                             port=int(Config.database('mongo_port'))).get_database('sportmonks')

    @classmethod
    def get_session_sportmonks(cls):
        return cls.sm_session
