import multiprocessing as mp

from core.database import BetExplorerDatabase as Database
from core.logger import logger

from modules.betexplorer2.controllers import ChampionshipController, SeasonController, \
    MatchController, OddsController, PerformanceController, MarkovChainController
from modules.betexplorer2.dataframes import DataframeController


def reset_database():

    logger.info('reset_database() ')

    session = Database.get_session()
    ChampionshipController(session).drop()
    SeasonController(session).drop()
    MatchController(session).drop()
    PerformanceController(session).drop()


def create_championships():
    """ Create championships """
    logger.info('create_championships() ')
    session = Database.get_session()
    ChampionshipController(session).initialize()


def create_seasons(initial_year, final_year):
    """ Create seasons """
    logger.info('create_seasons() ')
    session = Database.get_session()
    SeasonController(session).initialize(initial_year, final_year)


def request_seasons(replace=False):
    logger.info('request_seasons() ')
    session = Database.get_session()

    championships = ChampionshipController(session).list()

    args = [(c, replace) for c in championships]
    pool = mp.Pool(mp.cpu_count())
    pool.map(_request_season, args)


def _request_season(args):
    logger.info('_request_season() ' + str(args))
    session = Database.open_new_session()

    SeasonController(session).crawler(*args)


def create_matches(replace=False):
    logger.info('create_matches() ')
    session = Database.get_session()
    MatchController(session).initialize()

    if replace:
        MatchController(session).drop()

    seasons = SeasonController(session).list()

    pool = mp.Pool(mp.cpu_count())
    pool.map(_create_matches, seasons)


def _create_matches(season):

    logger.info('create_matches(): ' + str(season))
    session = Database.open_new_session()

    MatchController(session).create_matches(season)
    SeasonController(session).update_summary_fields(season)
    MatchController(session).process_results(season)
    MatchController(session).update_rounds(season)


def request_matches(replace=False):
    logger.info('request_matches() ')
    session = Database.get_session()

    seasons = SeasonController(session).list()
    args = [(s, replace) for s in seasons]

    pool = mp.Pool(mp.cpu_count())
    pool.map(_request_matches, args)


def _request_matches(args):

    logger.info('create_matches(): ' + str(args))
    session = Database.open_new_session()

    MatchController(session).crawler(*args)


def create_odds(replace=False):

    session = Database.get_session()

    if replace:
        OddsController(session).unset_odds()

    seasons = SeasonController(session).list()

    pool = mp.Pool(mp.cpu_count())
    pool.map(_create_odds, seasons)


def _create_odds(season):

    logger.info('_create_odds(): ' + str(season))
    session = Database.open_new_session()

    OddsController(session).create_odds(season)
    OddsController(session).process_summary_odds(season)
    OddsController(session).process_favorites(season)


def process_hits(replace=False):

    session = Database.get_session()

    if replace:
        OddsController(session).unset_hits()

    seasons = SeasonController(session).list()

    pool = mp.Pool(mp.cpu_count())
    pool.map(_process_hits, seasons)


def _process_hits(season):

    logger.info('_process_hits(): ' + str(season))
    session = Database.open_new_session()

    MatchController(session).process_hits(season)


def create_dataframe(replace=False):

    session = Database.get_session()

    DataframeController(session).create_dataframe(replace)
    #DataframeController(session).create_dataframe_complete()


def create_performances(replace=False):

    session = Database.get_session()

    if replace:
        PerformanceController(session).drop()

    PerformanceController(session).initialize()
    seasons = SeasonController(session).list()

    pool = mp.Pool(mp.cpu_count())
    pool.map(_create_performances, seasons)


def _create_performances(season):

    logger.info('_create_performances: ' + str(season))
    session = Database.open_new_session()

    PerformanceController(session).create_performances(season)

def create_markov_chains(replace=False):

    session = Database.get_session()

    if replace:
        MarkovChainController(session).drop()

    MarkovChainController(session).initialize()
    seasons = SeasonController(session).list()

    pool = mp.Pool(mp.cpu_count())
    pool.map(_create_markov_chains, seasons)

def _create_markov_chains(season):

    logger.info('_create_markov_chains: ' + str(season))
    session = Database.open_new_session()

    MarkovChainController(session).create_markov_chains(season)