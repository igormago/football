from abc import ABC
from modules.betexplorer2.notations import MatchNotation, OddsTypeNotation, OddsFavoritesNotation, \
    OddsSummaryNotation, PerformanceNotation, SeasonNotation, ChampionshipNotation


class Model(ABC):

    def __init__(self, data):
        if not data:
            self.data = dict()
        else:
            self.data = data

    def get_data(self):
        return self.data


class Championship(Model):

    def __init__(self, data=None):
        super().__init__(data)


class Season(Model):

    def __init__(self, data=None):
        super().__init__(data)

    def get_initial_year(self):
        return self.data[SeasonNotation.INITIAL_YEAR]

    def get_championship_name(self):
        return self.data[SeasonNotation.CHAMPIONSHIP][ChampionshipNotation.NAME]


class Match(Model):

    def __init__(self, data):
        super().__init__(data)

    def get_money_line_result(self):
        return self.data[MatchNotation.RESULTS][OddsTypeNotation.MONEY_LINE]

    def get_both_teams_to_score_result(self):
        return self.data[MatchNotation.RESULTS][OddsTypeNotation.BOTH_TEAMS_TO_SCORE]

    def get_result(self, odds_type):
        return self.data[MatchNotation.RESULTS][odds_type]

    def get_odds_list(self, odds_type=None):
        if not odds_type:
            return self.data[MatchNotation.ODDS][MatchNotation.Odds.LIST]
        else:
            return self.data[MatchNotation.ODDS][MatchNotation.Odds.LIST][odds_type]

    def get_odds_summary(self, odds_type=None, summary_type=None):
        if not odds_type:
            return self.data[MatchNotation.ODDS][MatchNotation.Odds.SUMMARY]
        else:
            if not summary_type:
                return self.data[MatchNotation.ODDS][
                    MatchNotation.Odds.SUMMARY][odds_type]
            else:
                return self.data[MatchNotation.ODDS][
                    MatchNotation.Odds.SUMMARY][odds_type][summary_type]

    def get_money_line_probability(self, odd_type=None):
        return self.data[MatchNotation.ODDS][MatchNotation.Odds.SUMMARY][OddsTypeNotation.MONEY_LINE][odd_type][
            OddsSummaryNotation.AVERAGE]

    def get_odds_favorites(self, odds_type=None):
        if not odds_type:
            return self.data[MatchNotation.ODDS][MatchNotation.Odds.FAVORITES]
        else:
            return self.data[MatchNotation.ODDS][MatchNotation.Odds.FAVORITES][odds_type]

    def has_odds(self):
        return MatchNotation.ODDS in self.data

    def has_hits(self, description=None):
        if description:
            return MatchNotation.ODDS in self.data and \
                   description in self.data[MatchNotation.ODDS]
        else:
            return MatchNotation.ODDS in self.data

    def has_odds_list(self):
        return self.has_odds() and \
               MatchNotation.Odds.LIST in self.data[MatchNotation.ODDS]

    def has_odds_summary(self, odds_type=None):

        if not odds_type:
            return self.has_odds() and \
                   MatchNotation.Odds.SUMMARY in self.data[MatchNotation.ODDS]
        else:
            return self.has_odds() and \
                   MatchNotation.Odds.SUMMARY in self.data[MatchNotation.ODDS] and \
                   odds_type in self.data[MatchNotation.ODDS][MatchNotation.Odds.SUMMARY]

    def has_odds_favorites(self, odds_type=None):
        if not odds_type:
            return self.has_odds() and \
                   MatchNotation.Odds.FAVORITES in self.data[MatchNotation.ODDS]
        else:
            return self.has_odds() and \
                   MatchNotation.Odds.FAVORITES in self.data[MatchNotation.ODDS] and \
                   odds_type in self.data[MatchNotation.ODDS][MatchNotation.Odds.FAVORITES]

    def get_favorite_column(self, odds_type):
        return self.get_odds_favorites(odds_type)[OddsFavoritesNotation.FAVORITE][OddsFavoritesNotation.COLUMN]

    def get_hits(self, odds_type=None):
        if not odds_type:
            return self.data[MatchNotation.HITS]
        else:
            return self.data[MatchNotation.HITS][odds_type]


class Performance(Model):
    def has_probabilities(self, odds_type=None):
        if not odds_type:
            return PerformanceNotation.PROBABILITIES in self.data
        else:
            return PerformanceNotation.PROBABILITIES in self.data and \
                   odds_type in self.data[PerformanceNotation.PROBABILITIES]

    def get_probabilities(self, odds_type=None):

        if not odds_type:
            return self.data[PerformanceNotation.PROBABILITIES]
        else:
            return self.data[PerformanceNotation.PROBABILITIES][odds_type]
