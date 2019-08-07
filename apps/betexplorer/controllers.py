import pymongo
from modules.betexplorer2.notations import ChampionshipNotation, SeasonNotation, MatchNotation, OddsNotation, \
    OddsFavoritesNotation, \
    PerformanceNotation, OddsTypeNotation, MarkovChainNotation
from modules.betexplorer2.models import Season, Performance, Match
from modules.betexplorer2.repositories import ChampionshipRepository, SeasonRepository, MatchRepository, \
    PerformanceRepository, MarkovChainRepository
from modules.betexplorer2.utils import ChampionshipUtil
from modules.betexplorer2.scrapers import SeasonScraper, MatchOddsScraper
from modules.betexplorer2.crawlers import SeasonCrawler, MatchOddsCrawler
from modules.betexplorer2.business import MatchBusiness, SeasonBusiness, PerformanceBusiness, MarkovChainBusiness
from modules.betexplorer2.managers import ChampionshipManager, SeasonManager, DataframeManager
from pymongo.errors import DuplicateKeyError
from core.logger import logger
from abc import ABC
from modules.betexplorer2.markov import Markov


class Controller(ABC):
    def __init__(self, session):
        self.session = session


class ChampionshipController(Controller):
    """ Championship Controller """

    def create_indexes(self):
        return ChampionshipRepository(self.session).create_index(ChampionshipNotation.NAME, unique=True)

    def drop(self):
        return ChampionshipRepository(self.session).drop()

    def list(self, filter=None, projection=None):
        return ChampionshipRepository(self.session).list(filter, projection)

    def initialize(self):

        self.create_indexes()
        for champ in ChampionshipUtil.list_championships():
            try:
                ChampionshipRepository(self.session).insert(champ)
                logger.debug('Championship created: ' + str(champ))
            except DuplicateKeyError as e:
                logger.warning(e)
                pass

            ChampionshipManager.create_dir(champ)
            logger.debug('Championship DIR created: ' + str(champ))


class SeasonController(Controller):
    """ Season Controller """

    def create_indexes(self):

        return SeasonRepository(self.session).create_index(
            [(SeasonNotation.CHAMPIONSHIP + '.' + ChampionshipNotation.NAME, pymongo.ASCENDING),
             (SeasonNotation.INITIAL_YEAR, pymongo.ASCENDING),
             (SeasonNotation.FINAL_YEAR, pymongo.ASCENDING)],
            unique=True
        )

    def drop(self):
        return SeasonRepository(self.session).drop()

    def list_by_championship(self, championship=None):
        return ChampionshipRepository(self.session).get_seasons(championship)

    def list(self, filter=None, projection=None):
        return SeasonRepository(self.session).list(filter, projection)

    def initialize(self, initial_year, final_year):
        """ Creates seasons """
        self.create_indexes()

        for champ in ChampionshipRepository(self.session).list():

            for year in range(initial_year, final_year):

                season = SeasonBusiness.create(champ,year)

                try:
                    SeasonRepository(self.session).insert(season)
                    logger.debug('Season created: ' + str(season))
                except DuplicateKeyError as e:
                    logger.warning(e)
                    pass

                SeasonManager.create_dir(season)
                logger.debug('Season DIR created: ' + str(season))

    def crawler(self, championship, replace=False):

        seasons = ChampionshipRepository(self.session).get_seasons(championship)

        for season in seasons:
            SeasonCrawler.request(season, replace)

    def update_summary_fields(self, season):

        num_teams = SeasonRepository(self.session).get_number_of_teams(season)
        SeasonBusiness.update_summary_fields(season, num_teams)

        fields_to_update = {key: season[key] for key
                         in [SeasonNotation.NUM_TEAMS, SeasonNotation.NUM_ROUNDS, SeasonNotation.NUM_MATCHES_PER_ROUND]}

        SeasonRepository(self.session).update(season, fields_to_update)
        return season


class MatchController(Controller):
    """ Match Controller """

    def create_indexes(self):
        return MatchRepository(self.session).create_index(MatchNotation.ID_SITE, unique=True)

    def initialize(self):
        self.create_indexes()

    def drop(self):
        return MatchRepository(self.session).drop()

    def list_by_season(self, season=None):
        return SeasonRepository(self.session).get_matches(season)

    def create_matches(self, season):

        matches = SeasonScraper.extract_matches(season)

        for match in matches:
            try:
                MatchRepository(self.session).insert(match)
            except DuplicateKeyError as error:
                logger.debug(error)
                pass

    def process_results(self, season):

        matches = SeasonRepository(self.session).get_matches(season)
        for match in matches:
            try:

                match[MatchNotation.RESULTS] = dict()

                MatchBusiness.set_result_money_line(match)
                MatchBusiness.set_result_both_teams_to_score(match)

                fields_to_update = {MatchNotation.RESULTS: match[MatchNotation.RESULTS]}
                MatchRepository(self.session).update(match, fields_to_update)

            except Exception as e:
                logger.warning(e)
                pass

    def update_rounds(self, season):

        matches = list(SeasonRepository(self.session).get_matches(season).sort(MatchNotation.DATE))
        max_matches_per_round = season[SeasonNotation.NUM_MATCHES_PER_ROUND]

        matches = MatchBusiness.set_round_groups(matches, max_matches_per_round)

        for match in matches:

            fields = {MatchNotation.NUMBER: match[MatchNotation.NUMBER],
                      MatchNotation.ROUND_GROUP: match[MatchNotation.ROUND_GROUP]}

            MatchRepository(self.session).update(match, fields)

    def crawler(self, season, replace=False):

        matches = SeasonRepository(self.session).get_matches(season)

        for match in matches:
            MatchOddsCrawler.request(match, replace)

    def process_hits(self, season):

        matches = SeasonRepository(self.session).get_matches(season)
        for match in matches:
            match[MatchNotation.HITS] = dict()

            MatchBusiness.process_hits(match,
                                       OddsTypeNotation.BOTH_TEAMS_TO_SCORE,
                                       OddsFavoritesNotation.FAVORITE,
                                       )
            MatchBusiness.process_hits(match,
                                       OddsTypeNotation.MONEY_LINE,
                                       OddsFavoritesNotation.FAVORITE)

            if bool(match[MatchNotation.HITS]):

                fields_to_update = {MatchNotation.HITS: match[MatchNotation.HITS]}
                MatchRepository(self.session).update(match, fields_to_update)


class OddsController(Controller):
    """ Odds Controller """

    def unset_odds(self):

        fields = {MatchNotation.ODDS: True}
        MatchRepository(self.session).unset(dict(), fields)

    def unset_hits(self):

        fields = {MatchNotation.HITS: True}
        MatchRepository(self.session).unset(dict(), fields)

    def create_odds(self, season):
        """ Create Odds """
        for match in SeasonRepository(self.session).get_matches(season):
            odds_list = MatchOddsScraper.extract(match)
            if bool(odds_list):
                fields = {MatchNotation.odds_list(): odds_list}
                MatchRepository(self.session).update(match, fields)
            logger.debug('Created odds: ' + str(match))

    def process_summary_odds(self, season):
        """ Process summary odds """

        for match in SeasonRepository(self.session).get_matches(season):
            summary = MatchBusiness.get_resume_odds(match)
            if bool(summary):
                fields = {MatchNotation.odds_summary(): summary}
                MatchRepository(self.session).update(match, fields)

    def process_favorites(self, season):
        """ Processes favorites """
        for match in SeasonRepository(self.session).get_matches(season):
            favorites = MatchBusiness.get_favorites(match)
            if bool(favorites):
                fields = {MatchNotation.odds_favorites(): favorites}
                MatchRepository(self.session).update(match, fields)


class PerformanceController(Controller):
    """ Performance Controller """

    def create_indexes(self):
        indexes = [(PerformanceNotation.SEASON_ID, pymongo.ASCENDING),
                   (PerformanceNotation.TEAM, pymongo.ASCENDING),
                   (PerformanceNotation.LOCAL, pymongo.ASCENDING),
                   (PerformanceNotation.NEXT_MATCH, pymongo.ASCENDING),
                   (PerformanceNotation.MATCHES_PLAYED, pymongo.ASCENDING),
                   (PerformanceNotation.LAST_MATCHES_NUM, pymongo.ASCENDING)]

        PerformanceRepository(self.session).create_index(PerformanceNotation.NEXT_MATCH)
        return PerformanceRepository(self.session).create_index(indexes, unique=True)

    def initialize(self):
        self.create_indexes()

    def drop(self):
        return PerformanceRepository(self.session).drop()

    def create_performances(self, season):

        teams = SeasonRepository(self.session).get_teams(season)

        for team in teams:

            matches = list(SeasonRepository(self.session).get_matches_by_team(season, team))

            for local in PerformanceNotation.Local.list():
                performances = PerformanceBusiness.create(season, team, matches, local)

                PerformanceRepository(self.session).insert_many(performances)

class MarkovChainController(Controller):

    def create_indexes(self):
        indexes = [(MarkovChainNotation.SEASON_ID, pymongo.ASCENDING),
                   (MarkovChainNotation.TEAM, pymongo.ASCENDING),
                   (MarkovChainNotation.LOCAL, pymongo.ASCENDING),
                   (MarkovChainNotation.NEXT_MATCH, pymongo.ASCENDING),
                   (MarkovChainNotation.MATCHES_PLAYED, pymongo.ASCENDING),
                   (MarkovChainNotation.LAST_MATCHES_NUM, pymongo.ASCENDING)]

        MarkovChainRepository(self.session).create_index(MarkovChainNotation.NEXT_MATCH)
        MarkovChainRepository(self.session).create_index(indexes, unique=True)

    def initialize(self):
        self.create_indexes()

    def drop(self):
        return MarkovChainRepository(self.session).drop()


    def create_markov_chains(self, season):

        teams = SeasonRepository(self.session).get_teams(season)

        for team in teams:

            matches = list(SeasonRepository(self.session).get_matches_by_team(season, team))

            for local in PerformanceNotation.Local.list():

                chains = MarkovChainBusiness.create(season, team, matches, local)
                MarkovChainRepository(self.session).insert_many(chains)

