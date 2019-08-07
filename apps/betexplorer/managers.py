import os
from core.config import PATH_BETEXPLORER_CHAMPIONSHIPS, PATH_BETEXPLORER_MATCHES, PATH_BETEXPLORER_DATAFRAMES
from modules.betexplorer2.models import Season
from modules.betexplorer2.notations import ChampionshipNotation, SeasonNotation, MatchNotation
from core.logger import logger
from abc import ABC
from pandas.io.json import json_normalize
import pandas as pd


class Manager(ABC):
    """ Abstract Manager """

    @classmethod
    def get_filename(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def get_file(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def is_file_exists(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def create_file(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def _create_file(cls, filename, content):
        """ Writes the content to the file """
        file = open(filename, 'wb')
        file.write(content)
        file.close()

        logger.info('File Created: ' + filename)

    @classmethod
    def _is_file_exists(cls, filename):
        """ Writes the content to the file """
        return os.path.exists(filename)

    @classmethod
    def _get_file(cls, filename):
        """ Returns the file """
        return open(filename, 'r')


class ChampionshipManager:
    @staticmethod
    def create_dir(championship):
        """ Creates directories for championships and matches """

        dir_name = PATH_BETEXPLORER_CHAMPIONSHIPS + championship[ChampionshipNotation.NAME]
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        dir_name = PATH_BETEXPLORER_MATCHES + championship[ChampionshipNotation.NAME]
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)


class SeasonManager(Manager):
    """ Season Manager """

    @classmethod
    def get_filename(cls, season):
        """ Returns the local file name to a season """
        return PATH_BETEXPLORER_CHAMPIONSHIPS + "/" + season[SeasonNotation.CHAMPIONSHIP][ChampionshipNotation.NAME] + "/" + \
               season[SeasonNotation.NAME] + '.html'

    @classmethod
    def get_file(cls, season):
        """ Returns the file """
        return super()._get_file(cls.get_filename(season))

    @classmethod
    def is_file_exists(cls, season):
        """checks if the file exist"""
        return super()._is_file_exists(cls.get_filename(season))

    @classmethod
    def create_file(cls, season, content):
        """ Writes the content to the file """
        super()._create_file(cls.get_filename(season), content)

    @classmethod
    def create_dir(cls, season):
        directory = cls.get_dir(season)

        if not os.path.exists(directory):
            os.mkdir(directory)

    @staticmethod
    def get_dir(season):
        seasonModel = Season(season)
        return PATH_BETEXPLORER_MATCHES + seasonModel.get_championship_name() + "/" + \
               str(seasonModel.get_initial_year()) + "/"


class MatchOddsManager(Manager):
    """ Manager to Odds File """

    @classmethod
    def get_filename(cls, match, odds_type):
        """ Returns the local file name to odds from a match """
        filename = SeasonManager.get_dir(match[MatchNotation.SEASON]) + match[MatchNotation.ID_SITE] + '-' + odds_type + '.json'

        return filename

    @classmethod
    def get_file(cls, match, odds_type):
        """ Returns the file """
        return super()._get_file(cls.get_filename(match, odds_type))

    @classmethod
    def is_file_exists(cls, match, odds_type):
        """checks if the file exist"""
        return super()._is_file_exists(cls.get_filename(match, odds_type))

    @classmethod
    def create_file(cls, match, odds_type, content):
        """ Writes the content to the file """
        super()._create_file(cls.get_filename(match, odds_type), content)


class DataframeManager(Manager):
    @classmethod
    def is_file_exists(cls, title):
        """checks if the file exist"""
        return super()._is_file_exists(cls.get_filename(title))

    @classmethod
    def get_filename(cls, title):
        """ Returns the local file name to odds from a match """
        filename = PATH_BETEXPLORER_DATAFRAMES + title + '.csv'

        return filename

    @classmethod
    def create_file(cls, title, content):
        """ Writes the content to the file """

        rows = list(content)
        df = json_normalize(rows)
        df.to_csv(cls.get_filename(title))

    @classmethod
    def get_file(cls, title):
        return super()._get_file(cls.get_filename(title))
