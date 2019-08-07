import unittest
from core.database import BetExplorerDatabase as database
from modules.betexplorer2.models import Match
from modules.betexplorer2.notations import MatchNotation, OddsNotation, OddsTypeNotation, OddsSummaryNotation, \
    OddsFavoritesNotation, PerformanceNotation, MarkovChainNotation
from modules.betexplorer2.repositories import MatchRepository, ChampionshipRepository, SeasonRepository, \
    PerformanceRepository, MarkovChainRepository


class TestMarkovChains(unittest.TestCase):

    def test_first_match(self):

        session = database.get_session()

        match = MatchRepository(session).get_by_id_site('OvMLBTBD')
        p_filter = {PerformanceNotation.NEXT_MATCH: match[MatchNotation.ID]}

        markov_chains = MarkovChainRepository(session).list(filter=p_filter)

        self.assertEqual(markov_chains.count(), 6)

        for mc in markov_chains:

            self.assertEqual(mc[PerformanceNotation.MATCHES_PLAYED], 0)
            self.assertEqual(mc[PerformanceNotation.MATCHES_PLAYED_HOME], 0)
            self.assertEqual(mc[PerformanceNotation.MATCHES_PLAYED_AWAY], 0)
            self.assertEqual(mc[PerformanceNotation.LAST_MATCHES_NUM], 0)
            self.assertIn(mc[PerformanceNotation.TEAM],['Lazio','Torino'])

            chains = mc[MarkovChainNotation.CHAINS]

            self.assertEqual(len(chains),2)


    def test_markov_chain(self):

        session = database.get_session()

        match = MatchRepository(session).get_by_id_site('UPfBnP5G')

        p_filter = {PerformanceNotation.NEXT_MATCH: match[MatchNotation.ID],
                    PerformanceNotation.TEAM: 'Napoli'}

        markov_chains = MarkovChainRepository(session).list(filter=p_filter)

        self.assertEqual(markov_chains.count(), 3)

        for mc in markov_chains:

            if (mc[MarkovChainNotation.LOCAL] == MarkovChainNotation.Local.OVERALL):
                scored = mc[MarkovChainNotation.CHAINS][MarkovChainNotation.SCORED]
                conceded = mc[MarkovChainNotation.CHAINS][MarkovChainNotation.CONCEDED]

                self.assertEqual(scored['0_0'], 0)
                self.assertEqual(scored['0_1'], 1)
                self.assertEqual(scored['1_0'], 0.2857)
                self.assertEqual(scored['1_1'], 0.7143)
                self.assertEqual(scored['current_state'], 0)

                self.assertEqual(conceded['0_0'], 0.75)
                self.assertEqual(conceded['0_1'], 0.25)
                self.assertEqual(conceded['1_0'], 0.2)
                self.assertEqual(conceded['1_1'], 0.8)
                self.assertEqual(conceded['current_state'], 1)

            if (mc[MarkovChainNotation.LOCAL] == MarkovChainNotation.Local.HOME):
                scored = mc[MarkovChainNotation.CHAINS][MarkovChainNotation.SCORED]
                conceded = mc[MarkovChainNotation.CHAINS][MarkovChainNotation.CONCEDED]

                self.assertEqual(scored['0_0'], 0)
                self.assertEqual(scored['0_1'], 1)
                self.assertEqual(scored['1_0'], 0)
                self.assertEqual(scored['1_1'], 1)
                self.assertEqual(scored['current_state'], 1)

                self.assertEqual(conceded['0_0'], 0.5)
                self.assertEqual(conceded['0_1'], 0.5)
                self.assertEqual(conceded['1_0'], 0.5)
                self.assertEqual(conceded['1_1'], 0.5)
                self.assertEqual(conceded['current_state'], 1)

            if (mc[MarkovChainNotation.LOCAL] == MarkovChainNotation.Local.AWAY):
                scored = mc[MarkovChainNotation.CHAINS][MarkovChainNotation.SCORED]
                conceded = mc[MarkovChainNotation.CHAINS][MarkovChainNotation.CONCEDED]

                self.assertEqual(scored['0_0'], 0)
                self.assertEqual(scored['0_1'], 1)
                self.assertEqual(scored['1_0'], 0.6667)
                self.assertEqual(scored['1_1'], 0.3333)
                self.assertEqual(scored['current_state'], 0)

                self.assertEqual(conceded['0_0'], 0.5)
                self.assertEqual(conceded['0_1'], 0.5)
                self.assertEqual(conceded['1_0'], 0)
                self.assertEqual(conceded['1_1'], 1)
                self.assertEqual(conceded['current_state'], 1)

if __name__ == '__main__':
    unittest.main()
