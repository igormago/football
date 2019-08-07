from copy import deepcopy

from modules.betexplorer2.markov import Markov
from modules.betexplorer2.models import Match, Performance
from modules.betexplorer2.notations import ChampionshipNotation, MatchNotation, SeasonNotation, OddsNotation, \
    OddsSummaryNotation, \
    OddsFavoritesNotation, OddsTypeNotation, PerformanceNotation, MarkovChainNotation
from modules.betexplorer2.utils import ChampionshipUtil

import numpy as np
import operator


class ObjectBusiness:

    def __init__(self, model):
        self.model = model

class SeasonBusiness(ObjectBusiness):

    def __init__(self, model):
        super().__init__(model)

    class SeasonType:
        RR1 = 'RR1'
        RR2 = 'RR2'

    @staticmethod
    def create(championship, year):

        def set_championship():
            season[SeasonNotation.CHAMPIONSHIP] = championship

        def set_years():
            season[SeasonNotation.INITIAL_YEAR] = year
            if championship_name in brazilian_championships_names:
                season[SeasonNotation.FINAL_YEAR] = year
            else:
                season[SeasonNotation.FINAL_YEAR] = year + 1

        def set_type():
            if championship_name in brazilian_championships_names:
                season[SeasonNotation.TYPE] = SeasonBusiness.SeasonType.RR1
            else:
                season[SeasonNotation.TYPE] = SeasonBusiness.SeasonType.RR2

        def set_path():
            if championship_name in brazilian_championships_names:
                path = championship[ChampionshipNotation.PATHS][0] + "-" + str(year)
            else:
                print(championship_name, [ChampionshipUtil.SPAIN_A[ChampionshipNotation.NAME]])
                if championship_name == ChampionshipUtil.SPAIN_A[ChampionshipNotation.NAME] and \
                        int(year) > 2015:
                    path = championship[ChampionshipNotation.PATHS][1] + "-" + str(year) + "-" + str(year + 1)
                else:
                    print('0' + championship_name + str(year))
                    path = championship[ChampionshipNotation.PATHS][0] + "-" + str(year) + "-" + str(year + 1)
            season[SeasonNotation.PATH] = path

        def set_name():
            if championship_name in brazilian_championships_names:
                season[SeasonNotation.NAME] = championship_name + " " + \
                                                 str(season[SeasonNotation.INITIAL_YEAR])
            else:
                season[SeasonNotation.NAME] = championship_name + " " + \
                                                 str(season[SeasonNotation.INITIAL_YEAR]) + "-" + \
                                                 str(season[SeasonNotation.FINAL_YEAR])

        season = dict()
        championship_name = championship[ChampionshipNotation.NAME]
        brazilian_championships_names = [ChampionshipUtil.BRAZIL_A[ChampionshipNotation.NAME],
                                         ChampionshipUtil.BRAZIL_B[ChampionshipNotation.NAME]]
        set_championship()
        set_years()
        set_type()
        set_path()
        set_name()

        return season

    @staticmethod
    def update_summary_fields(season, num_teams):

        season[SeasonNotation.NUM_TEAMS] = int(num_teams)
        season[SeasonNotation.NUM_ROUNDS] = int((num_teams - 1) * 2)
        season[SeasonNotation.NUM_MATCHES_PER_ROUND] = int(num_teams / 2)

        return season


class MatchBusiness:

    def process_hits(match, odds_type, column):

        match_model = Match(match)

        if match_model.has_odds_favorites(odds_type):

            if match_model.has_hits(odds_type):
                hits_by_type = match_model.get_hits(odds_type)
            else:
                hits_by_type = dict()

            description = column
            hits_by_type[description] = MatchBusiness.get_bookmaker_hit(match, odds_type)

            if bool(hits_by_type):
                match[MatchNotation.HITS][odds_type] = hits_by_type


    @staticmethod
    def set_result_money_line(match):

        def _get_result():
            try:
                if match[MatchNotation.HOME_GOALS] > match[MatchNotation.AWAY_GOALS]:
                    return OddsNotation.MoneyLine.HOME
                elif match[MatchNotation.HOME_GOALS] < match[MatchNotation.AWAY_GOALS]:
                    return OddsNotation.MoneyLine.AWAY
                else:
                    return OddsNotation.MoneyLine.DRAW
            except TypeError:
                return None

        match[MatchNotation.RESULTS][OddsTypeNotation.MONEY_LINE] = _get_result()

    @staticmethod
    def set_result_both_teams_to_score(match):

        def _get_result():
            try:
                if match[MatchNotation.HOME_GOALS] > 0 and match[MatchNotation.AWAY_GOALS] > 0:
                    return OddsNotation.BothTeamsToScore.YES
                else:
                    return OddsNotation.BothTeamsToScore.NO
            except TypeError:
                return None

        match[MatchNotation.RESULTS][OddsTypeNotation.BOTH_TEAMS_TO_SCORE] = _get_result()

    @staticmethod
    def get_resume_odds(match):

        def _extract_resume(odds_values):

            summary_values = dict()
            summary_values[OddsSummaryNotation.COUNT] = len(odds_values)
            summary_values[OddsSummaryNotation.AVERAGE] = round(float(np.average(odds_values)), 4)
            # noinspection PyTypeChecker
            summary_values[OddsSummaryNotation.MEDIAN] = round(float(np.median(odds_values)), 4)
            summary_values[OddsSummaryNotation.MAXIMUM] = round(float(np.max(odds_values)), 4)
            summary_values[OddsSummaryNotation.MINIMUM] = round(float(np.min(odds_values)), 4)
            # noinspection PyTypeChecker
            summary_values[OddsSummaryNotation.STD_DEVIATION] = round(float(np.std(odds_values)), 4)

            return summary_values

        def _process_summary(odds_type, fields):

            odds_summary = dict()

            # check if exist odds of a type
            if odds_type in odds:

                odds_list = odds[odds_type]

                type_summary_original = dict()
                type_summary_normalized = dict()
                odds_original = dict()

                for field in fields:
                    type_summary_original[field] = list()
                    type_summary_normalized[field] = list()
                    odds_original[field] = list()

                for odd_i in odds_list:

                    active = True
                    for field in fields:
                        active = active and odd_i[OddsNotation.get_active_field(field)]

                    if active:
                        total = 0

                        prob_original = dict()
                        for field in fields:
                            prob_original[field] = 1 / odd_i[field]
                            total = total + prob_original[field]
                            odds_original[field].append(odd_i[field])

                        for field in fields:
                            type_summary_original[field].append(prob_original[field])

                        prob_normalized = dict()
                        for field in fields:
                            prob_normalized[field] = prob_original[field] / total

                        for field in fields:
                            type_summary_normalized[field].append(prob_normalized[field])

                valid = True
                for field in fields:
                    valid = valid and (len(type_summary_original[field]) > 0)

                if valid:
                    total_avg = 0
                    for field in fields:
                        odds_summary[field] = dict()
                        transformed = _extract_resume(type_summary_normalized[field])
                        odds_summary[field][OddsSummaryNotation.Type.TRANSFORMED] = transformed
                        original = _extract_resume(odds_original[field])
                        odds_summary[field][OddsSummaryNotation.Type.ORIGINAL] = original
                        total_avg += round(float(np.average(type_summary_original[field])), 4)
                    odds_summary[OddsSummaryNotation.OVERROUND] = round(total_avg - 1, 4)

            return odds_summary

        summary = dict()

        if Match(match).has_odds_list():

            odds = Match(match).get_odds_list()

            ml_resume = _process_summary(OddsTypeNotation.MONEY_LINE, OddsNotation.MoneyLine.list())
            bts_resume = _process_summary(OddsTypeNotation.BOTH_TEAMS_TO_SCORE, OddsNotation.BothTeamsToScore.list())

            if bool(ml_resume):
                summary[OddsTypeNotation.MONEY_LINE] = ml_resume

            if bool(bts_resume):
                summary[OddsTypeNotation.BOTH_TEAMS_TO_SCORE] = bts_resume

        return summary

    @staticmethod
    def get_favorites(match):

        def _process_favorites(odds_type, fields, fav_orders):

            favorite_of_type = dict()

            if odds_type in summary:

                summary_of_type = summary[odds_type]

                odds_avg = dict()

                for f in fields:
                    odds_avg[f] = summary_of_type[f][OddsSummaryNotation.Type.ORIGINAL][OddsSummaryNotation.AVERAGE]

                sort_values = sorted(odds_avg.items(), key=operator.itemgetter(1), reverse=False)

                for idx, field in zip(fav_orders, sort_values):
                    favorite_of_type[idx] = dict()
                    field = field[0]
                    favorite_of_type[idx][OddsFavoritesNotation.COLUMN] = field
                    value_field = summary_of_type[field][OddsSummaryNotation.Type.ORIGINAL]
                    favorite_of_type[idx][OddsFavoritesNotation.VALUE] = value_field

            return favorite_of_type

        favorites = dict()
        match_model = Match(match)

        if match_model.has_odds_summary():

            summary = match_model.get_odds_summary()

            ml_favorites = _process_favorites(OddsTypeNotation.MONEY_LINE, OddsNotation.MoneyLine.list(),
                                              OddsFavoritesNotation.list_fmu())
            bts_favorites = _process_favorites(OddsTypeNotation.BOTH_TEAMS_TO_SCORE, OddsNotation.BothTeamsToScore.list(),
                                               OddsFavoritesNotation.list_fu())

            if bool(ml_favorites):
                favorites[OddsTypeNotation.MONEY_LINE] = ml_favorites

            if bool(bts_favorites):
                favorites[OddsTypeNotation.BOTH_TEAMS_TO_SCORE] = bts_favorites

        return favorites

    @staticmethod
    def set_round_groups(matches, max_matches_per_round):

        mpr = 0
        number = 1
        round_group = 1

        for match in matches:
            match[MatchNotation.NUMBER] = number
            match[MatchNotation.ROUND_GROUP] = round_group
            number += 1
            mpr += 1

            if mpr == max_matches_per_round:
                mpr = 0
                round_group += 1

        return matches


    @staticmethod
    def get_bookmaker_hit(match, result_type):

        match_model = Match(match)
        return match_model.get_result(result_type) == match_model.get_favorite_column(result_type)

class PerformanceBusiness:

    @classmethod
    def create(cls, season, team, matches, local):

        performances = list()

        performance = dict()
        performance[PerformanceNotation.SEASON_ID] = season[SeasonNotation.ID]
        performance[PerformanceNotation.TEAM] = team
        performance[PerformanceNotation.LOCAL] = local
        performance[PerformanceNotation.GOALS_FOR] = 0
        performance[PerformanceNotation.GOALS_AGAINST] = 0
        performance[PerformanceNotation.POINTS] = 0
        performance[PerformanceNotation.WINS] = 0
        performance[PerformanceNotation.LOSES] = 0
        performance[PerformanceNotation.DRAWS] = 0
        performance[PerformanceNotation.MATCHES_PLAYED] = 0
        performance[PerformanceNotation.MATCHES_PLAYED_HOME] = 0
        performance[PerformanceNotation.MATCHES_PLAYED_AWAY] = 0
        performance[PerformanceNotation.LAST_MATCHES_NUM] = 0

        performance[PerformanceNotation.MATCHES_SCORED_GOALS] = 0
        performance[PerformanceNotation.MATCHES_CONCEDED_GOALS] = 0

        performance[PerformanceNotation.NEXT_MATCH] = matches[0][MatchNotation.ID]

        performances.append(performance)
        performance = deepcopy(performance)

        for idx, m in enumerate(matches):

            try:
                performance[PerformanceNotation.NEXT_MATCH] = matches[idx+1][MatchNotation.ID]
            except:
                performance[PerformanceNotation.NEXT_MATCH] = None

            performance[PerformanceNotation.MATCHES_PLAYED] += 1

            if m[MatchNotation.HOME_TEAM] == team:
                performance[PerformanceNotation.MATCHES_PLAYED_HOME] += 1
            else:
                performance[PerformanceNotation.MATCHES_PLAYED_AWAY] += 1

            if (local == PerformanceNotation.Local.OVERALL) or\
                (local == PerformanceNotation.Local.HOME and m[MatchNotation.HOME_TEAM] == team) or\
                (local == PerformanceNotation.Local.AWAY and m[MatchNotation.AWAY_TEAM] == team):

                performance[PerformanceNotation.LAST_MATCHES_NUM] += 1

                result_ml = Match(m).get_money_line_result()
                cls.calculate_performance(m, performance, team, result_ml)

            performances.append(performance)
            performance = deepcopy(performance)

        return performances


    @staticmethod
    def calculate_performance(match, performance, team, result_ml):

        if match[MatchNotation.HOME_TEAM] == team:

            try:
                performance[PerformanceNotation.GOALS_FOR] += match[MatchNotation.HOME_GOALS]
                if match[MatchNotation.HOME_GOALS] > 0:
                    performance[PerformanceNotation.MATCHES_SCORED_GOALS] += 1
            except TypeError:
                performance[PerformanceNotation.GOALS_FOR] += 0
                performance[PerformanceNotation.MATCHES_SCORED_GOALS] += 0

            try:
                performance[PerformanceNotation.GOALS_AGAINST] += match[MatchNotation.AWAY_GOALS]
                if match[MatchNotation.AWAY_GOALS] > 0:
                    performance[PerformanceNotation.MATCHES_CONCEDED_GOALS] += 1
            except TypeError:
                performance[PerformanceNotation.GOALS_AGAINST] += 0
                performance[PerformanceNotation.MATCHES_CONCEDED_GOALS] += 0

            if result_ml == OddsNotation.MoneyLine.HOME:
                performance[PerformanceNotation.POINTS] += 3
                performance[PerformanceNotation.WINS] += 1
            elif result_ml == OddsNotation.MoneyLine.DRAW:
                performance[PerformanceNotation.POINTS] += 1
                performance[PerformanceNotation.DRAWS] += 1
            else:
                performance[PerformanceNotation.LOSES] += 1
        else:

            try:
                performance[PerformanceNotation.GOALS_FOR] += match[MatchNotation.AWAY_GOALS]
                if match[MatchNotation.AWAY_GOALS] > 0:
                    performance[PerformanceNotation.MATCHES_CONCEDED_GOALS] += 1
            except TypeError:
                performance[PerformanceNotation.GOALS_FOR] += 0

            try:
                performance[PerformanceNotation.GOALS_AGAINST] += match[MatchNotation.HOME_GOALS]
                if match[MatchNotation.HOME_GOALS] > 0:
                    performance[PerformanceNotation.MATCHES_SCORED_GOALS] += 1
            except TypeError:
                performance[PerformanceNotation.GOALS_AGAINST] += 0

            if result_ml == OddsNotation.MoneyLine.AWAY:
                performance[PerformanceNotation.POINTS] += 3
                performance[PerformanceNotation.WINS] += 1
            elif result_ml == OddsNotation.MoneyLine.DRAW:
                performance[PerformanceNotation.POINTS] += 1
                performance[PerformanceNotation.DRAWS] += 1
            else:
                performance[PerformanceNotation.LOSES] += 1

class MarkovChainBusiness:

    @classmethod
    def create(cls, season, team, matches, local):

        markov_chains = list()

        markov = dict()

        markov[MarkovChainNotation.SEASON_ID] = season[SeasonNotation.ID]
        markov[MarkovChainNotation.TEAM] = team
        markov[MarkovChainNotation.LOCAL] = local
        markov[MarkovChainNotation.MATCHES_PLAYED] = 0
        markov[MarkovChainNotation.MATCHES_PLAYED_HOME] = 0
        markov[MarkovChainNotation.MATCHES_PLAYED_AWAY] = 0
        markov[MarkovChainNotation.LAST_MATCHES_NUM] = 0

        markov[MarkovChainNotation.NEXT_MATCH] = matches[0][MatchNotation.ID]

        mc_scored = Markov(2)
        mc_conceded = Markov(2)

        chains = {MarkovChainNotation.SCORED: mc_scored.get_dict(),
                  MarkovChainNotation.CONCEDED: mc_conceded.get_dict()}

        markov[MarkovChainNotation.CHAINS] = chains

        markov_chains.append(markov)
        markov = deepcopy(markov)

        for idx, match in enumerate(matches):

            try:
                markov[MarkovChainNotation.NEXT_MATCH] = matches[idx+1][MatchNotation.ID]
            except Exception:
                markov[MarkovChainNotation.NEXT_MATCH] = None

            markov[MarkovChainNotation.MATCHES_PLAYED] += 1
            if match[MatchNotation.HOME_TEAM] == team:
                markov[PerformanceNotation.MATCHES_PLAYED_HOME] += 1
            else:
                markov[PerformanceNotation.MATCHES_PLAYED_AWAY] += 1

            if (local == PerformanceNotation.Local.OVERALL) or\
                (local == PerformanceNotation.Local.HOME and match[MatchNotation.HOME_TEAM] == team) or\
                (local == PerformanceNotation.Local.AWAY and match[MatchNotation.AWAY_TEAM] == team):

                cls.calculate_markov(match, team, mc_scored, mc_conceded)

                markov[MarkovChainNotation.LAST_MATCHES_NUM] += 1

                chains = {MarkovChainNotation.SCORED: mc_scored.get_dict(),
                          MarkovChainNotation.CONCEDED: mc_conceded.get_dict()}

                markov[MarkovChainNotation.CHAINS] = chains

            markov_chains.append(markov)
            markov = deepcopy(markov)

        return markov_chains

    @staticmethod
    def calculate_markov(match, team, markov_scored, markov_conceded):

        if match[MatchNotation.HOME_TEAM] == team:
            try:
                if match[MatchNotation.HOME_GOALS] > 0:
                    markov_scored.visit(1)
                else:
                    markov_scored.visit(0)
            except TypeError:
                markov_scored.visit(0)

            try:
                if match[MatchNotation.AWAY_GOALS] > 0:
                    markov_conceded.visit(1)
                else:
                    markov_conceded.visit(0)
            except TypeError:
                markov_conceded.visit(0)
        else:
            try:
                if match[MatchNotation.AWAY_GOALS] > 0:
                    markov_scored.visit(1)
                else:
                    markov_scored.visit(0)
            except TypeError:
                markov_scored.visit(0)

            try:
                if match[MatchNotation.HOME_GOALS] > 0:
                    markov_conceded.visit(1)
                else:
                    markov_conceded.visit(0)
            except TypeError:
                markov_conceded.visit(0)