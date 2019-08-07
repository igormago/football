import multiprocessing as mp

from core.database import BetExplorerDatabase as Database
from core.logger import logger
from modules.betexplorer2.controllers import \
    ChampionshipController, SeasonController, MatchController, OddsController


def reset_database():

    logger.info('Service: reset_database()')
    session = Database.get_session()

    MatchController(session).drop()
    SeasonController(session).drop()
    ChampionshipController(session).drop()


def create_championships():

    logger.info('Service: create_championships()')
    session = Database.get_session()
    ChampionshipController(session).initialize()


def create_seasons(initial_year, final_year):

    logger.info('Service: create_seasons()')
    session = Database.get_session()
    SeasonController(session).initialize(initial_year, final_year)


def request_seasons(replace=False):

    logger.info('Service: request_seasons()')
    session = Database.get_session()

    champs = list(ChampionshipController(session).list())
    args = [(champs[idx], replace) for idx, item in enumerate(champs)]

    p = mp.Pool(mp.cpu_count())
    p.map(_requests_season, args)


def _requests_season(args):

    logger.info('request_season(): ' + str(args))
    new_session = Database.open_new_session()
    SeasonController(new_session).crawler(*args)


def create_matches(replace=False):

    logger.info('Service: create_matches()')
    session = Database.SESSION

    if replace:
        MatchController(session).drop()

    MatchController(session).initialize()
    seasons = list(SeasonController(session).list())

    p = mp.Pool(mp.cpu_count())
    p.map(_create_matches, seasons)


def _create_matches(season):

    session = Database.open_new_session()

    logger.info('extract_matches(): ' + str(season))
    MatchController(session).create_matches(season)
    season = SeasonController(session).update_summary_fields(season)
    MatchController(session).update_rounds(season)


def request_matches(replace=False):

    logger.info('Service: request_matches()')
    session = Database.get_session()

    seasons = list(SeasonController(session).list())
    args = [(seasons[i], replace) for i, item_a in enumerate(seasons)]

    p = mp.Pool(mp.cpu_count())
    p.map(_request_matches, args)


def _request_matches(args):

    session = Database.open_new_session()
    SeasonController(session).crawler_matches(*args)


def create_odds(replace=False):
    logger.info('Service: extract_odds()')

    session = Database.SESSION
    seasons = list(SeasonController(session).list())

    if replace:
        OddsController(session).reset()

    p = mp.Pool(mp.cpu_count())
    p.map(_create_odds, seasons)


def _create_odds(season):

    logger.info('_extract_odds(): ' + str(season))
    session = Database.open_new_session()
    MatchController(session).create_odds(season)
    MatchController(session).process_summary_odds(season)
    #MatchController(session).process_favorites(season)



