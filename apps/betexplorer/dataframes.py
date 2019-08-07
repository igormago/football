from modules.betexplorer2.controllers import Controller
from modules.betexplorer2.managers import DataframeManager
from modules.betexplorer2.notations import MatchNotation, PerformanceNotation
from modules.betexplorer2.repositories import MatchRepository, PerformanceRepository, MarkovChainRepository


class DataframeController(Controller):
    """ Dataframe Controller """

    def create_dataframe(self, replace=False):

        if replace or not DataframeManager.is_file_exists('Total'):

            filter = {}

            projection = {MatchNotation.ID: 1,
                          MatchNotation.ID_SITE: 1,
                          MatchNotation.RESULTS: 1,
                          MatchNotation.HOME_TEAM: 1,
                          MatchNotation.AWAY_TEAM: 1,
                          MatchNotation.HOME_GOALS: 1,
                          MatchNotation.AWAY_GOALS: 1,
                          MatchNotation.season_initial_year(): 1,
                          MatchNotation.championship_name(): 1,
                          MatchNotation.odds_summary(): 1,
                          MatchNotation.odds_favorites(): 1,
                          MatchNotation.HITS: 1,
                          MatchNotation.ROUND_GROUP: 1}

            matches = MatchRepository(self.session).list(filter,projection)

            content = list()

            for m in matches:

                filter = {PerformanceNotation.NEXT_MATCH: m[MatchNotation.ID]}
                projection = {PerformanceNotation.ID: 0,
                              PerformanceNotation.NEXT_MATCH: 0,
                              PerformanceNotation.SEASON_ID: 0,
                              PerformanceNotation.LAST_MATCHES_NUM: 0}

                performances = PerformanceRepository(self.session).list(filter, projection)
                perf_home = dict()
                perf_away = dict()

                for p in performances:
                    if p[PerformanceNotation.TEAM] == m[MatchNotation.HOME_TEAM]:
                        if p[PerformanceNotation.LOCAL] == PerformanceNotation.Local.OVERALL:
                            perf_home[PerformanceNotation.Local.OVERALL] = p
                        elif p[PerformanceNotation.LOCAL] == PerformanceNotation.Local.HOME:
                            perf_home[PerformanceNotation.Local.HOME] = p
                        else:
                            perf_home[PerformanceNotation.Local.AWAY] = p
                    else:
                        if p[PerformanceNotation.LOCAL] == PerformanceNotation.Local.OVERALL:
                            perf_away[PerformanceNotation.Local.OVERALL] = p
                        elif p[PerformanceNotation.LOCAL] == PerformanceNotation.Local.HOME:
                            perf_away[PerformanceNotation.Local.HOME] = p
                        else:
                            perf_away[PerformanceNotation.Local.AWAY] = p

                    del p[PerformanceNotation.TEAM]
                    del p[PerformanceNotation.LOCAL]

                markov_chains = MarkovChainRepository(self.session).list(filter, projection)
                mc_home = dict()
                mc_away = dict()

                for mc in markov_chains:
                    if mc[PerformanceNotation.TEAM] == m[MatchNotation.HOME_TEAM]:
                        if mc[PerformanceNotation.LOCAL] == PerformanceNotation.Local.OVERALL:
                            mc_home[PerformanceNotation.Local.OVERALL] = mc
                        elif mc[PerformanceNotation.LOCAL] == PerformanceNotation.Local.HOME:
                            mc_home[PerformanceNotation.Local.HOME] = mc
                        else:
                            mc_home[PerformanceNotation.Local.AWAY] = mc

                    else:
                        if mc[PerformanceNotation.LOCAL] == PerformanceNotation.Local.OVERALL:
                            mc_away[PerformanceNotation.Local.OVERALL] = mc
                        elif mc[PerformanceNotation.LOCAL] == PerformanceNotation.Local.HOME:
                            mc_away[PerformanceNotation.Local.HOME] = mc
                        else:
                            mc_away[PerformanceNotation.Local.AWAY] = mc

                    del mc[PerformanceNotation.TEAM]
                    del mc[PerformanceNotation.LOCAL]

                perf = {'m': m, 'ph': perf_home, 'pa': perf_away, 'kh': mc_home, 'ka': mc_away}
                content.append(perf)

            DataframeManager.create_file('Total', content)



