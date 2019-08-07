import unittest
from core.database import BetExplorerDatabase as database
from modules.betexplorer2.models import Match
from modules.betexplorer2.notations import MatchNotation, OddsNotation, OddsTypeNotation, OddsSummaryNotation, \
    OddsFavoritesNotation, PerformanceNotation
from modules.betexplorer2.repositories import MatchRepository, ChampionshipRepository, SeasonRepository, \
    PerformanceRepository

class TestPerformances(unittest.TestCase):

    def test_first_match(self):

        session = database.get_session()

        match = MatchRepository(session).get_by_id_site('OvMLBTBD')
        p_filter = {PerformanceNotation.NEXT_MATCH: match[MatchNotation.ID]}

        performances = PerformanceRepository(session).list(filter=p_filter)

        self.assertEqual(performances.count(), 6)

        for p in performances:
            self.assertEqual(p[PerformanceNotation.POINTS],0)
            self.assertEqual(p[PerformanceNotation.WINS], 0)
            self.assertEqual(p[PerformanceNotation.DRAWS], 0)
            self.assertEqual(p[PerformanceNotation.LOSES], 0)
            self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED], 0)
            self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED_HOME], 0)
            self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED_AWAY], 0)
            self.assertEqual(p[PerformanceNotation.LAST_MATCHES_NUM], 0)
            self.assertEqual(p[PerformanceNotation.GOALS_FOR], 0)
            self.assertEqual(p[PerformanceNotation.GOALS_AGAINST], 0)
            self.assertIn(p[PerformanceNotation.TEAM],['Lazio','Torino'])


    def test_performance(self):

        session = database.get_session()

        match = MatchRepository(session).get_by_id_site('bTr4iQt3')

        p_filter = {PerformanceNotation.NEXT_MATCH: match[MatchNotation.ID],
                    PerformanceNotation.TEAM: 'Lazio'}

        performances = PerformanceRepository(session).list(filter=p_filter)

        self.assertEqual(performances.count(), 3)

        for p in performances:
            print(p)
            if (p[PerformanceNotation.LOCAL] == PerformanceNotation.Local.OVERALL):
                self.assertEqual(p[PerformanceNotation.POINTS], 3)
                self.assertEqual(p[PerformanceNotation.WINS], 0)
                self.assertEqual(p[PerformanceNotation.DRAWS], 3)
                self.assertEqual(p[PerformanceNotation.LOSES], 1)
                self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED], 4)
                self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED_HOME], 2)
                self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED_AWAY], 2)
                self.assertEqual(p[PerformanceNotation.LAST_MATCHES_NUM], 4)
                self.assertEqual(p[PerformanceNotation.GOALS_FOR], 3)
                self.assertEqual(p[PerformanceNotation.GOALS_AGAINST], 4)

            elif p[PerformanceNotation.LOCAL] == PerformanceNotation.Local.HOME:
                self.assertEqual(p[PerformanceNotation.POINTS], 2)
                self.assertEqual(p[PerformanceNotation.WINS], 0)
                self.assertEqual(p[PerformanceNotation.DRAWS], 2)
                self.assertEqual(p[PerformanceNotation.LOSES], 0)
                self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED], 4)
                self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED_HOME], 2)
                self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED_AWAY], 2)
                self.assertEqual(p[PerformanceNotation.LAST_MATCHES_NUM], 2)
                self.assertEqual(p[PerformanceNotation.GOALS_FOR], 2)
                self.assertEqual(p[PerformanceNotation.GOALS_AGAINST], 2)

            elif p[PerformanceNotation.LOCAL] == PerformanceNotation.Local.AWAY:
                self.assertEqual(p[PerformanceNotation.POINTS], 1)
                self.assertEqual(p[PerformanceNotation.WINS], 0)
                self.assertEqual(p[PerformanceNotation.DRAWS], 1)
                self.assertEqual(p[PerformanceNotation.LOSES], 1)
                self.assertEqual(p[PerformanceNotation.MATCHES_PLAYED], 4)
                self.assertEqual(p[PerformanceNotation.LAST_MATCHES_NUM], 2)
                self.assertEqual(p[PerformanceNotation.GOALS_FOR], 1)
                self.assertEqual(p[PerformanceNotation.GOALS_AGAINST], 2)

if __name__ == '__main__':
    unittest.main()
