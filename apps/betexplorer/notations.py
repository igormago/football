def concat(*args):
    """
    Returns label of field
    :param args: a, b, c
    :return: a.b.c
    """

    field = args[0]
    for v in args[1:]:
        field = field + '.' + v

    return field


class ObjectNotation(object):
    ID = '_id'

    def __init__(self, data):
        self.data = data

    @staticmethod
    def concat(*args):
        """
        Returns label of field    
        :param args: a, b, c 
        :return: a.b.c
        """

        field = args[0]
        for v in args[1:]:
            field = field + '.' + v

        return field


class ChampionshipNotation(ObjectNotation):
    """ Championship JSON Fields """

    COUNTRY = 'co'
    NAME = 'nm'
    PATHS = 'pt'


class SeasonNotation(ObjectNotation):
    """ Season JSON Fields """

    COLLECTION = 'seasons'

    CHAMPIONSHIP = 'ch'
    INITIAL_YEAR = 'yi'
    FINAL_YEAR = 'yf'
    NAME = 'nm'
    PATH = 'pt'
    TYPE = 'tp'
    NUM_TEAMS = 'nt'
    NUM_ROUNDS = 'nr'
    NUM_MATCHES_TOTAL = 'nmt'
    NUM_MATCHES_PER_ROUND = 'nmr'

    @classmethod
    def championship_id(cls):
        return cls.concat(cls.CHAMPIONSHIP, ChampionshipNotation.ID)


class MatchNotation(ObjectNotation):
    """ Match JSON Fields """

    COLLECTION = 'matches'

    SEASON = 'ss'
    ID_SITE = 'id'
    ROUND = 'rd'
    HOME_TEAM = 'th'
    AWAY_TEAM = 'ta'
    HOME_GOALS = 'gh'
    AWAY_GOALS = 'ga'
    ODDS_HOME = 'oh'
    ODDS_DRAW = 'od'
    ODDS_AWAY = 'oa'
    DATE = 'dt'
    OBSERVATION = 'ob'
    ODDS = 'odds'
    RESULTS = 'res'
    NUMBER = 'nm'
    ROUND_GROUP = 'rg'
    HITS = 'hit'

    class Odds(ObjectNotation):
        LIST = 'l'
        SUMMARY = 's'
        FAVORITES = 'f'

    class Hits(ObjectNotation):
        STRATEGY = 'st'

    @classmethod
    def season_id(cls):
        return cls.concat(cls.SEASON, SeasonNotation.ID)

    @classmethod
    def odds_list(cls):
        return cls.concat(cls.ODDS, cls.Odds.LIST)

    @classmethod
    def odds_summary(cls):
        return cls.concat(cls.ODDS, cls.Odds.SUMMARY)

    @classmethod
    def odds_favorites(cls):
        return cls.concat(cls.ODDS, cls.Odds.FAVORITES)

    @classmethod
    def season_initial_year(cls):
        return cls.concat(cls.SEASON, SeasonNotation.INITIAL_YEAR)

    @classmethod
    def championship_name(cls):
        return cls.concat(cls.SEASON, SeasonNotation.CHAMPIONSHIP, ChampionshipNotation.NAME)

    @classmethod
    def bts_result(cls):
        return cls.concat(cls.RESULTS, OddsTypeNotation.BOTH_TEAMS_TO_SCORE)

class OddsSummaryNotation(ObjectNotation):

    COUNT = 'cnt'
    AVERAGE = 'avg'
    MEDIAN = 'med'
    MAXIMUM = 'max'
    MINIMUM = 'min'
    STD_DEVIATION = 'std'
    OVERROUND = 'ovr'

    class Type:
        ORIGINAL = 'o'
        TRANSFORMED = 't'

    @classmethod
    def list(cls):
        return [cls.AVERAGE, cls.MAXIMUM, cls.MINIMUM, cls.STD_DEVIATION]


class OddsTypeNotation(ObjectNotation):

    MONEY_LINE = '1x2'
    OVER_UNDER = 'ou'
    ASIAN_HANDICAP = 'ah'
    DRAW_NO_BET = 'ha'
    DOUBLE_CHANCE = 'dc'
    BOTH_TEAMS_TO_SCORE = 'bts'

    @classmethod
    def list(cls):
        return [cls.MONEY_LINE, cls.OVER_UNDER, cls.ASIAN_HANDICAP,
                cls.DRAW_NO_BET, cls.DOUBLE_CHANCE, cls.BOTH_TEAMS_TO_SCORE]


class OddsNotation(ObjectNotation):

    BOOKMAKER = 'bk'
    ACTIVE = '_'

    @classmethod
    def get_active_field(cls, value):
        return value + cls.ACTIVE

    class AsianHandicap:

        HOME = 'h'
        AWAY = 'a'
        HANDICAP = 'hc'

        @classmethod
        def list(cls):
            return [cls.HOME, cls.AWAY]

    class OverUnder:

        OVER = 'o'
        UNDER = 'u'
        GOALS = 'g'

        @classmethod
        def list(cls):
            return [cls.OVER, cls.UNDER]

    class MoneyLine:

        HOME = 'h'
        DRAW = 'd'
        AWAY = 'a'

        @classmethod
        def list(cls):
            return [cls.HOME, cls.DRAW, cls.AWAY]

    class DoubleChance:

        HOME_DRAW = 'hd'
        HOME_AWAY = 'ha'
        DRAW_AWAY = 'da'

        @classmethod
        def list(cls):
            return [cls.HOME_DRAW, cls.HOME_AWAY, cls.DRAW_AWAY]

    class BothTeamsToScore:

        YES = 'y'
        NO = 'n'

        @classmethod
        def list(cls):
            return [cls.YES, cls.NO]

    class DrawNoBet:

        HOME = 'h'
        AWAY = 'a'

        @classmethod
        def list(cls):
            return [cls.HOME, cls.AWAY]


class OddsFavoritesNotation(ObjectNotation):

    FAVORITE = 'f'
    MEDIUM = 'm'
    UNDERDOG = 'u'

    VALUE = 'v'
    COLUMN = 'c'

    @classmethod
    def list_fmu(cls):
        return [cls.FAVORITE, cls.MEDIUM, cls.UNDERDOG]

    @classmethod
    def list_fu(cls):
        return [cls.FAVORITE, cls.UNDERDOG]

    @classmethod
    def type_and_favorite(cls, odds_type, favorite):
        return cls.concat(MatchNotation.odds_favorites(), odds_type, favorite)


class BookmakerNotation(ObjectNotation):
    ID_SITE = 'id'
    NAME = 'nm'


class PerformanceNotation(ObjectNotation):

    class Local:

        HOME = 'h'
        AWAY = 'a'
        OVERALL = 'o'

        @classmethod
        def list(cls):
            return [cls.HOME, cls.AWAY, cls.OVERALL]

    SEASON_ID = 'ss'
    TEAM = 'tm'
    LOCAL = 'loc'
    GOALS_FOR = 'gf'
    GOALS_AGAINST = 'ga'
    GOALS_BALANCE = 'gb'

    POINTS = 'pt'
    NEXT_MATCH = 'nm'
    WINS = 'w'
    LOSES = 'l'
    DRAWS = 'd'

    MATCHES_PLAYED = 'mp'
    MATCHES_PLAYED_HOME = 'mph'
    MATCHES_PLAYED_AWAY = 'mpa'
    LAST_MATCHES_NUM = 'lmn'

    MATCHES_SCORED_GOALS = 'msg'
    MATCHES_CONCEDED_GOALS = 'mcg'

    PROBABILITIES = 'pr'
    MARKOV_CHAINS = 'mc'

    class Probabilities:

        VALUE = 'v'
        COUNT = 'c'

    class MoneyLine:

        PROB_FOR = 'pf'
        PROB_AGAINST = 'pa'
        PROB_DRAW = 'pd'

    class BothTeamsToScore:
        PROB_YES = 'py'
        PROB_NO = 'pn'


class MarkovChainNotation(PerformanceNotation):

    CHAINS = 'ch'

    SCORED = 's'
    CONCEDED = 'c'

    YES = 'y'
    NO = 'n'


class DataFrameNotation:

    MATCHES = 'm'
    PERFORMANCE_HOME = 'ph'
    PERFORMANCE_AWAY = 'pa'

    @classmethod
    def from_matches(cls,notation):
        return concat(cls.MATCHES, notation)

    @classmethod
    def odds_summary(cls):
        return cls.concat(cls.MATCHES, MatchNotation.odds_summary(), )