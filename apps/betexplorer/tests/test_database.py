import unittest
from core.database import BetExplorerDatabase as database
from modules.betexplorer2.models import Match
from modules.betexplorer2.notations import MatchNotation, OddsNotation, OddsTypeNotation, OddsSummaryNotation, \
    OddsFavoritesNotation, PerformanceNotation
from modules.betexplorer2.repositories import MatchRepository, ChampionshipRepository, SeasonRepository, \
    PerformanceRepository


class TestDatabaseCount(unittest.TestCase):

    def test_championships(self):
        ''' Checks the number of championships in the database '''
        session = database.get_session()
        champs = ChampionshipRepository(session).list()
        self.assertEqual(champs.count(),9)

    def test_seasons(self):
        ''' Checks the number of seasons in the database '''
        session = database.get_session()
        seasons = SeasonRepository(session).list()

        self.assertEqual(seasons.count(), 99)


class TestMatches(unittest.TestCase):

    def test_match_fields(self):
        ''' Checks the fields of matches '''

        session = database.get_session()

        match = MatchRepository(session).get_by_id_site('IXDncKQQ')

        self.assertEqual(match[MatchNotation.HOME_TEAM], 'Fluminense')
        self.assertEqual(match[MatchNotation.AWAY_TEAM], 'Guarani')
        self.assertEqual(match[MatchNotation.HOME_GOALS], 1)
        self.assertEqual(match[MatchNotation.AWAY_GOALS], 0)
        self.assertEqual(match[MatchNotation.RESULTS][OddsTypeNotation.MONEY_LINE],
                         OddsNotation.MoneyLine.HOME)

        matchModel = Match(match)
        self.assertEqual(matchModel.get_money_line_result(), OddsNotation.MoneyLine.HOME)
        self.assertEqual(matchModel.get_both_teams_to_score_result(), OddsNotation.BothTeamsToScore.NO)

        match = MatchRepository(session).get_by_id_site('l0MZkxJ0')

        self.assertEqual(match[MatchNotation.HOME_TEAM], 'Goias')
        self.assertEqual(match[MatchNotation.AWAY_TEAM], 'Corinthians')
        self.assertEqual(match[MatchNotation.HOME_GOALS], 1)
        self.assertEqual(match[MatchNotation.AWAY_GOALS], 1)
        self.assertEqual(match[MatchNotation.RESULTS][OddsTypeNotation.MONEY_LINE],
                         OddsNotation.MoneyLine.DRAW)

        matchModel = Match(match)
        self.assertEqual(matchModel.get_money_line_result(), OddsNotation.MoneyLine.DRAW)
        self.assertEqual(matchModel.get_both_teams_to_score_result(), OddsNotation.BothTeamsToScore.YES)

        match = MatchRepository(session).get_by_id_site('zJAzNd9i')

        self.assertEqual(match[MatchNotation.HOME_TEAM], 'Portuguesa')
        self.assertEqual(match[MatchNotation.AWAY_TEAM], 'Sao Paulo')
        self.assertEqual(match[MatchNotation.HOME_GOALS], 2)
        self.assertEqual(match[MatchNotation.AWAY_GOALS], 3)
        self.assertEqual(match[MatchNotation.RESULTS][OddsTypeNotation.MONEY_LINE],
                         OddsNotation.MoneyLine.AWAY)

        matchModel = Match(match)
        self.assertEqual(matchModel.get_money_line_result(), OddsNotation.MoneyLine.AWAY)
        self.assertEqual(matchModel.get_both_teams_to_score_result(), OddsNotation.BothTeamsToScore.YES)

    def test_match_odds(self):
        ''' Checks the odds of matches '''

        session = database.get_session()

        match = MatchRepository(session).get_by_id_site('8KEmQeVQ')

        matchModel = Match(match)
        odds_ml = matchModel.get_odds_list(OddsTypeNotation.MONEY_LINE)
        odds_bts = matchModel.get_odds_list(OddsTypeNotation.BOTH_TEAMS_TO_SCORE)

        self.assertEqual(len(odds_ml), 25)
        self.assertEqual(len(odds_bts), 13)

        ml_summary = matchModel.get_odds_summary(OddsTypeNotation.MONEY_LINE)
        bts_summary = matchModel.get_odds_summary(OddsTypeNotation.BOTH_TEAMS_TO_SCORE)

        self.assertIsNotNone(ml_summary)
        self.assertIsNotNone(bts_summary)

        ml_favorites = matchModel.get_odds_favorites(OddsTypeNotation.MONEY_LINE)
        bts_favorites = matchModel.get_odds_favorites(OddsTypeNotation.BOTH_TEAMS_TO_SCORE)

        self.assertIsNotNone(ml_favorites)
        self.assertIsNotNone(bts_favorites)
        self.assertEqual(matchModel.get_favorite_column(OddsTypeNotation.MONEY_LINE),
                         OddsNotation.MoneyLine.HOME)

        self.assertEqual(matchModel.get_favorite_column(OddsTypeNotation.BOTH_TEAMS_TO_SCORE),
                         OddsNotation.BothTeamsToScore.NO)

        ml_hits = matchModel.get_hits(OddsTypeNotation.MONEY_LINE)
        bts_hits = matchModel.get_hits(OddsTypeNotation.BOTH_TEAMS_TO_SCORE)

        self.assertEqual(ml_hits[OddsFavoritesNotation.FAVORITE],True)
        self.assertEqual(bts_hits[OddsFavoritesNotation.FAVORITE],False)





if __name__ == '__main__':
    unittest.main()
