import requests
from modules.betexplorer2.notations import SeasonNotation, MatchNotation, OddsTypeNotation
from modules.betexplorer2.managers import SeasonManager, MatchOddsManager
from modules.betexplorer2.config import URL_BETEXPLORER, URL_BETEXPLORER_MATCHES
from core.logger import logger


class SeasonCrawler(object):
    """ Crawler to Season """

    @classmethod
    def get_url(cls, season):
        """ Return the site URL """
        return URL_BETEXPLORER + season[SeasonNotation.PATH] + '/results'

    @classmethod
    def request(cls, season, replace=False):
        """ Request the season file to the site """
        logger.debug('request(): ' + str(season))

        if not SeasonManager.is_file_exists(season) or replace:

            download = False
            r = None
            while not download:

                try:
                    logger.debug('Requests: ' + cls.get_url(season))
                    r = requests.get(cls.get_url(season))
                    download = True
                except Exception as e:
                    logger.exception(e)

            SeasonManager.create_file(season, r.content)


class MatchOddsCrawler(object):
    """ Crawler to Odds from Matches """

    @staticmethod
    def get_url():
        """ Get the site URL to crawler odds from matches """
        return URL_BETEXPLORER_MATCHES

    @classmethod
    def request(cls, match, replace=False):
        """ Request the season file to the site """
        for odds_type in OddsTypeNotation.list():

            if not MatchOddsManager.is_file_exists(match, odds_type) or replace:

                params = dict()
                params['p'] = '0'
                params['e'] = match[MatchNotation.ID_SITE]

                headers = dict()
                headers['Referer'] = 'notnull'
                params['b'] = odds_type

                done = False
                while not done:

                    try:
                        r = requests.get(cls.get_url(), params=params, headers=headers, timeout=50)
                        MatchOddsManager.create_file(match, odds_type, r.content)
                        done = True
                    except Exception as e:
                        logger.exception(e)
                        done = False
                        pass
