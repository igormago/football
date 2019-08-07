from modules.betexplorer2.notations import ChampionshipNotation


class Util:

    @staticmethod
    def get_float(value):
        try:
            return float(value)
        except ValueError:
            return None
        except TypeError:
            return None


class ChampionshipUtil:

    BRAZIL_A = {ChampionshipNotation.NAME: 'Brazil A',
                ChampionshipNotation.PATHS: ['brazil/serie-a'],
                ChampionshipNotation.COUNTRY: 'Brazil'}

    BRAZIL_B = {ChampionshipNotation.NAME: 'Brazil B',
                ChampionshipNotation.PATHS: ['brazil/serie-b'],
                ChampionshipNotation.COUNTRY: 'Brazil'}

    ENGLAND_A = {ChampionshipNotation.NAME: 'England A',
                 ChampionshipNotation.PATHS: ['england/premier-league'],
                 ChampionshipNotation.COUNTRY: 'England'}

    SPAIN_A = {ChampionshipNotation.NAME: 'Spain A',
               ChampionshipNotation.PATHS: ['spain/primera-division', 'spain/laliga'],
               ChampionshipNotation.COUNTRY: 'Spain'}

    GERMANY_A = {ChampionshipNotation.NAME: 'Germany A',
                 ChampionshipNotation.PATHS: ['germany/bundesliga'],
                 ChampionshipNotation.COUNTRY: 'Germany'}

    ITALY_A = {ChampionshipNotation.NAME: 'Italy A',
               ChampionshipNotation.PATHS: ['italy/serie-a'],
               ChampionshipNotation.COUNTRY: 'Italy'}

    PORTUGAL_A = {ChampionshipNotation.NAME: 'Portugal A',
                  ChampionshipNotation.PATHS: ['portugal/primeira-liga'],
                  ChampionshipNotation.COUNTRY: 'Portugal'}

    FRANCE_A = {ChampionshipNotation.NAME: 'France A',
                ChampionshipNotation.PATHS: ['france/ligue-1'],
                ChampionshipNotation.COUNTRY: 'France'}

    NETHERLANDS_A = {ChampionshipNotation.NAME: 'Netherlands A',
                     ChampionshipNotation.PATHS: ['netherlands/eredivisie'],
                     ChampionshipNotation.COUNTRY: 'Netherlands'}

    @classmethod
    def list_championships(cls):
        return [cls.BRAZIL_A, cls.BRAZIL_B, cls.ENGLAND_A, cls.SPAIN_A,
                cls.GERMANY_A, cls.ITALY_A, cls.PORTUGAL_A, cls.FRANCE_A, cls.NETHERLANDS_A]
