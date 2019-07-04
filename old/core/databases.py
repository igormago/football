from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pymongo import MongoClient

client = MongoClient(host='localhost',port=27017, connect=False)

from core.config import DataBase_BETEXPLORER, DataBase_SOCCER, DataBase_OLD_BETFAIR, DataBase_CENTRAL

class SessionFactory():
    def __init__(self, engine):
        self.engine = create_engine(engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def getSession(self):
        return self.session


class BetExplorerDatabase:

    SESSION = MongoClient(host='localhost', port=27017).get_database('betexplorer')

    @staticmethod
    def open_new_session():
        return MongoClient(host='localhost', port=27017, maxPoolSize=200).get_database('betexplorer')

    @classmethod
    def get_session(cls):
        return cls.SESSION

class CartolaDatabase:

    SESSION = MongoClient(host='localhost', port=27017).get_database('cartola')

    @staticmethod
    def open_new_session():
        return MongoClient(host='localhost', port=27017, maxPoolSize=200).get_database('cartola')

    @classmethod
    def get_session(cls):
        return cls.SESSION

class BetafairhdDatabase:

    SESSION = MongoClient(host='localhost', port=27017).get_database('betfairhd')

    @staticmethod
    def open_new_session():
        return MongoClient(host='localhost', port=27017, maxPoolSize=200).get_database('betfairhd')

    @classmethod
    def get_session(cls):
        return cls.SESSION

class FootballdataDatabase:

    SESSION = MongoClient(host='localhost', port=27017).get_database('footballdata')

    @staticmethod
    def open_new_session():
        return MongoClient(host='localhost', port=27017, maxPoolSize=200).get_database('footballdata')

    @classmethod
    def get_session(cls):
        return cls.SESSION

class SportMonksDatabase:

    SESSION = MongoClient(host='localhost', port=27018).get_database('sportmonks')

    @staticmethod
    def open_new_session():
        return MongoClient(host='localhost', port=27018, maxPoolSize=200).get_database('sportmonks')

    @classmethod
    def get_session(cls):
        return cls.SESSION

session_betexplorer = SessionFactory(DataBase_BETEXPLORER).getSession()
session_old_betfair = SessionFactory(DataBase_OLD_BETFAIR).getSession()
session_soocer = SessionFactory(DataBase_SOCCER).getSession()
session_central = SessionFactory(DataBase_CENTRAL).getSession()
session_premierleague = client.premierleague
session_footballdata = client.footballdata
session_betfair = client.betfair

Base = declarative_base()
