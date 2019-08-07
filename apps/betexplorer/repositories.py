from modules.betexplorer2.notations import ChampionshipNotation, SeasonNotation, MatchNotation, PerformanceNotation, \
    MarkovChainNotation
from abc import ABC
from pymongo import database
import pymongo



class Repository(ABC):
    # not implemented

    ID = '_id'

    @property
    def collection(self):
        raise NotImplementedError

    def __init__(self, session):
        assert isinstance(session, database.Database)
        self.cursor = session[self.collection]
        self.session = session

    def insert_many(self, models):
        return self.cursor.insert_many(models)

    def insert(self, model):
        return self.cursor.insert_one(model)

    def drop(self):
        return self.cursor.drop()

    def list(self, filter=None, projection=None):
        return self.cursor.find(filter, projection)

    def get(self, model):
        return self.cursor.find({self.ID: model[self.ID]})

    def create_index(self, keys, **kwargs):
        return self.cursor.create_index(keys, **kwargs)

    def update(self, model, fields):
        return self.cursor.update_one(
            {self.ID: model[self.ID]}, {'$set': fields}
        )

    def replace(self, new_model):
        return self.cursor.replace_one(
            {self.ID: new_model[self.ID]}, new_model
        )

    def unset(self, filter_fields, unset_fields):
        return self.cursor.update_many(
            filter_fields, {'$unset': unset_fields}
        )


class ChampionshipRepository(Repository):
    """ Repository for championships """
    collection = 'championships'

    def get_by_name(self, name):
        return self.cursor.find_one({ChampionshipNotation.NAME: name})

    def get_seasons(self, championship):
        filter = {SeasonNotation.championship_id(): championship[ChampionshipNotation.ID]}
        return SeasonRepository(self.session).list(filter=filter)


class SeasonRepository(Repository):
    """ Repository for seasons """
    collection = 'seasons'

    def get_number_of_teams(self, season):
        return len(self.get_teams(season))

    def get_teams(self, season):
        filter = {MatchNotation.season_id(): season[self.ID]}
        return MatchRepository(self.session).list(filter=filter).distinct(MatchNotation.HOME_TEAM)

    def get_matches_by_team(self, season, team, local=None):

        if local == PerformanceNotation.Local.HOME:
            filter = {MatchNotation.season_id(): season[SeasonNotation.ID], MatchNotation.HOME_TEAM: team}
        elif local == PerformanceNotation.Local.AWAY:
            filter = {MatchNotation.season_id(): season[SeasonNotation.ID], MatchNotation.AWAY_TEAM: team}
        else:
            filter = {MatchNotation.season_id(): season[SeasonNotation.ID],
                      '$or': [{MatchNotation.HOME_TEAM: team}, {MatchNotation.AWAY_TEAM: team}]}

        return MatchRepository(self.session).list(filter=filter).sort(MatchNotation.NUMBER, pymongo.ASCENDING)

    def get_matches(self, season):
        filter = {MatchNotation.season_id(): season[SeasonNotation.ID]}
        return MatchRepository(self.session).list(filter=filter)


class MatchRepository(Repository):
    """ Repository for matches """
    collection = 'matches'

    def update_odds_list(self, match):
        return self.cursor.update_one(
            {self.ID: match[self.ID]}, {'$set': {MatchNotation.ODDS: match[MatchNotation.ODDS]}}
        )

    def update_results(self, match):
        return self.cursor.update_one(
            {self.ID: match[self.ID]}, {'$set': {MatchNotation.RESULTS: match[MatchNotation.RESULTS]}}
        )

    def get_by_id_site(self, id_site):
        return self.cursor.find_one({MatchNotation.ID_SITE: id_site})

class PerformanceRepository(Repository):
    collection = 'performances'

    def get_by_next_match(self, match):

        filter = {PerformanceNotation.NEXT_MATCH: match[self.ID]}
        projection = {PerformanceNotation.ID: 0, PerformanceNotation.NEXT_MATCH: 0,
                      PerformanceNotation.SEASON_ID: 0}
        return super().list(filter, projection)

class MarkovChainRepository(Repository):

    collection = 'markov_chains'

    def get_by_next_match(self, match):

        filter = {MarkovChainNotation.NEXT_MATCH: match[self.ID]}
        projection = {MarkovChainNotation.ID: 0, MarkovChainNotation.NEXT_MATCH: 0,
                      MarkovChainNotation.SEASON_ID: 0}
        return super().list(filter, projection)