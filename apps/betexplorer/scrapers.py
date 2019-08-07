import json
from modules.betexplorer2.notations import MatchNotation, OddsNotation, BookmakerNotation, OddsTypeNotation
from modules.betexplorer2.managers import SeasonManager, MatchOddsManager
from modules.betexplorer2.utils import Util
from bs4 import BeautifulSoup
from datetime import datetime
from core.logger import logger
from abc import ABC, abstractmethod


class Scraper(object):

    def __init__(self, data):
        self.data = data


class SeasonScraper(Scraper):
    """ Scraper for seasons file """

    @classmethod
    def extract_matches(cls, season):

        def _create_match():

            def _extract_teams(td):

                a = td.find('a')
                href = a['href'].split("/")
                team_home_name = td.getText().split(" - ")[0].strip()
                team_away_name = td.getText().split(" - ")[1].strip()

                match[MatchNotation.ID_SITE] = href[-2]
                match[MatchNotation.HOME_TEAM] = team_home_name
                match[MatchNotation.AWAY_TEAM] = team_away_name

            def _extract_goals(td):

                try:
                    match[MatchNotation.HOME_GOALS] = int(td.getText().split(':')[0])

                    goals_away = td.getText().split(':')[1]

                    if len(goals_away.split()) > 1:
                        match[MatchNotation.AWAY_GOALS] = int(goals_away.split()[0])
                        match[MatchNotation.OBSERVATION] = goals_away.split()[1]
                    else:
                        match[MatchNotation.AWAY_GOALS] = int(goals_away)

                except ValueError:
                    match[MatchNotation.HOME_GOALS] = None
                    match[MatchNotation.AWAY_GOALS] = None
                    match[MatchNotation.OBSERVATION] = td.getText()

            def _extract_odds(td_home, td_draw, td_away):
                """ Extract odds"""

                def _extract_odd(td):

                    try:
                        items = str(td).split('"')
                        odd = items[-2]
                    except IndexError:
                        odd = None

                    return odd

                match[MatchNotation.ODDS_HOME] = Util.get_float(_extract_odd(td_home))
                match[MatchNotation.ODDS_DRAW] = Util.get_float(_extract_odd(td_draw))
                match[MatchNotation.ODDS_AWAY] = Util.get_float(_extract_odd(td_away))

            def _extract_date(td):
                date_string = td.text
                match[MatchNotation.DATE] = datetime.strptime(date_string, "%d.%m.%Y")

            match = dict()
            match[MatchNotation.SEASON] = season
            match[MatchNotation.ROUND] = int(round_num)

            tds = tr.find_all('td')

            _extract_teams(tds[0])
            _extract_goals(tds[1])
            _extract_odds(tds[2], tds[3], tds[4])
            _extract_date(tds[5])

            return match

        # begin extract_matches()
        file = SeasonManager.get_file(season)
        content = file.read()
        soup = BeautifulSoup(content, 'html.parser')

        r = soup.find('table')

        r = r.find_all("tr")
        round_num = 0

        matches = list()

        for tr in r:
            th = tr.find('th')

            if th is not None:
                round_num = th.getText().split(".")[0]
            else:
                m = _create_match()
                matches.append(m)
                logger.debug('match created: ' + str(m))

        return matches


class MatchOddsScraper(Scraper):
    """ Scraper for odds from matches"""

    def __init__(self, data):
        super().__init__(data)

    @classmethod
    def extract(cls, match):

        odds_list = dict()

        ml_odds = _MoneyLineScraper.extract(match)
        bts_odds = _BothTeamsToScoreScraper.extract(match)
        # _DoubleChance(self.data).extract()
        # _DrawNoBet(self.data).extract()
        # _OverUnderScraper(self.data).extract()
        # _AsianHandicapScraper(self.data).extract()

        if ml_odds:
            odds_list[OddsTypeNotation.MONEY_LINE] = ml_odds

        if bts_odds:
            odds_list[OddsTypeNotation.BOTH_TEAMS_TO_SCORE] = bts_odds

        return odds_list


class _OddsScraper(ABC):

    def __init__(self, data):
        self.data = data

    @property
    @abstractmethod
    def type(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def odds_keys(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def pos_initial_td(self):
        raise NotImplementedError

    @classmethod
    def get_odds_rows(cls, match, odds_type):

        file = MatchOddsManager().get_file(match, odds_type)
        content = file.read()
        content = json.loads(content)

        soup = BeautifulSoup(content['odds'], 'html.parser')
        bodies = soup.find_all('tbody')

        if bodies is not None:
            for body in bodies:
                trs = body.find_all('tr')
                return trs

    @staticmethod
    def extract_bookmaker(odds, tr):
        bookmaker = dict()

        bookmaker[BookmakerNotation.NAME] = tr.find('a').getText().strip()
        bookmaker[BookmakerNotation.ID_SITE] = int(tr['data-bid'])

        odds[OddsNotation.BOOKMAKER] = bookmaker

    @staticmethod
    def _extract_multiple_odds(odds, tds, odds_keys, pos_initial_td):

        idx = 0
        for i in range(pos_initial_td, len(tds)):
            try:
                odds[odds_keys[idx]] = Util.get_float(tds[i]['data-odd'])
            except KeyError:
                odds[odds_keys[idx]] = None
            idx += 1

        idx = 0
        for i in range(pos_initial_td, len(tds)):
            try:
                odds[OddsNotation.get_active_field(odds_keys[idx])] = tds[i]['class'].count("inactive") == 0
            except KeyError:
                odds[OddsNotation.get_active_field(odds_keys[idx])] = None
            idx += 1

    @classmethod
    def extract(cls, match):

        trs = cls.get_odds_rows(match, cls.type)

        odds_list = list()
        if trs:
            for tr in trs:
                odds = dict()
                cls.extract_bookmaker(odds, tr)
                cls.extract_odds(odds, tr)
                odds_list.append(odds)

        return odds_list

    @classmethod
    def extract_odds(cls, odds, tr):

        tds = tr.find_all('td')
        assert isinstance(cls.odds_keys, list)
        # noinspection PyTypeChecker
        cls._extract_multiple_odds(odds, tds, cls.odds_keys, cls.pos_initial_td)


class _MoneyLineScraper(_OddsScraper):
    type = OddsTypeNotation.MONEY_LINE
    odds_keys = OddsNotation.MoneyLine.list()
    pos_initial_td = 4


class _BothTeamsToScoreScraper(_OddsScraper):
    type = OddsTypeNotation.BOTH_TEAMS_TO_SCORE
    odds_keys = OddsNotation.BothTeamsToScore.list()
    pos_initial_td = 4


class _DoubleChance(_OddsScraper):
    type = OddsTypeNotation.DOUBLE_CHANCE
    odds_keys = OddsNotation.DoubleChance.list()
    pos_initial_td = 4


class _DrawNoBet(_OddsScraper):
    type = OddsTypeNotation.DRAW_NO_BET
    odds_keys = OddsNotation.DrawNoBet.list()
    pos_initial_td = 4


class _OverUnderScraper(_OddsScraper):
    type = OddsTypeNotation.OVER_UNDER
    odds_keys = OddsNotation.OverUnder.list()
    pos_initial_td = 5

    def extract_odds(self, odds, tr):
        tds = tr.find_all('td')
        odds[OddsNotation.OverUnder.GOALS] = Util.get_float(tds[4].getText())
        self._extract_multiple_odds(odds, tds, self.odds_keys, self.pos_initial_td)


class _AsianHandicapScraper(_OddsScraper):
    type = OddsTypeNotation.ASIAN_HANDICAP
    odds_keys = OddsNotation.AsianHandicap.list()
    pos_initial_td = 5

    def extract_odds(self, odds, tr):
        tds = tr.find_all('td')
        odds[OddsNotation.AsianHandicap.HANDICAP] = tds[4].getText()
        self._extract_multiple_odds(odds, tds, self.odds_keys, self.pos_initial_td)
