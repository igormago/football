from modules.betexplorer2.utils import Util
from modules.betexplorer2.notations import ChampionshipNotation, SeasonNotation, MatchNotation, OddsNotation
from core.database import BetExplorerDatabase as Database
from pandas.io.json import json_normalize
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
from core.logger import logger


def get_dataframe():

    session = Database.get_session()
    filter = {Util.get_field(MatchNotation.SEASON, SeasonNotation.YEAR_BEGIN): {"$gt":2012},
              Util.get_field(MatchNotation.FAVORITES, OddsNotation.Type.BOTH_TEAMS_TO_SCORE): {'$exists':1}
              }


    fields = {MatchNotation.SEASON:1, MatchNotation.RESULTS:1, MatchNotation.ODDS_SUMMARY:1}
    #
    # matches = session.matches.find({'CHAMP.YR':{"$gt":2012},'HIT.BTS_FAV':{'$exists':1}},
    #                                {'CHAMP':1,'FMU':1,'DT':1,'BTSR':1,'ODDS.RESUME.bts':1, 'round':1})

    logger.info('Consultando... ')
    matches = session.matches.list(filter, fields)
    logger.info('Passando pra lista... ')
    rows = list(matches)
    logger.info('Normalizando... ')
    df = json_normalize(rows)
    logger.info('Fim')
    df.to_csv('teste.csv')
    logger.info('Lendo... ')
    df = pd.read_csv('teste.csv')
    logger.info('Lido... ')

def learn():

    df = pd.read_csv('teste.csv')
    print(df)

def learning():

    df = get_dataframe()

    champs = session.matches.list({}).distinct("CHAMP.PT")
    champs.drop('spain/laliga')
    champs.sort()

    for c in champs:

        # param_grid = {'Cs': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        #               'penalty':['l2'],
        #               'max_iter':[1,10,100,1000]}

        param_grid = {'cv':[8,10]}
        clf = GridSearchCV(LogisticRegressionCV(), param_grid)

        matches = df[df['CHAMP.PT'] == c]

        train = matches[matches['CHAMP.YR'] < 2015]
        test = matches[matches['CHAMP.YR'] >= 2015]

        target = 'BTSR'

        features = ['ODDS.RESUME.bts.NO.avg','ODDS.RESUME.bts.YES.avg',
                    'ODDS.RESUME.bts.NO.std', 'ODDS.RESUME.bts.YES.std',
                    'ODDS.RESUME.bts.NO.max', 'ODDS.RESUME.bts.YES.max',
                    'ODDS.RESUME.bts.NO.min', 'ODDS.RESUME.bts.YES.min']

        X = train[features]
        Y = train[target]

        clf.fit(X,Y)

        X = test[features]
        Y = test[target]

        print(c,clf.score(X,Y)*100)

def learning2():

    df = get_dataframe()

    champs = session.matches.list({}).distinct("CHAMP.PT")
    champs.drop('spain/laliga')
    champs.sort()

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'penalty':['l1','l2'],
                  'max_iter':[1,10,100,1000]}

    clf = GridSearchCV(LogisticRegression(), param_grid)

    matches = df

    train = matches[matches['CHAMP.YR'] < 2015]
    test = matches[matches['CHAMP.YR'] >= 2015]

    target = 'BTSR'

    features = ['ODDS.RESUME.bts.NO.avg','ODDS.RESUME.bts.YES.avg']

    X = train[features]
    Y = train[target]

    clf.fit(X,Y)

    for c in champs:

        test = matches[matches['CHAMP.PT'] == c]

        X = test[features]
        Y = test[target]

        print(c,clf.score(X,Y)*100)

def learning3():

    df = get_dataframe()

    champs = session.matches.list({}).distinct("CHAMP.PT")
    champs.drop('spain/laliga')
    champs.sort()


    for c in champs:

        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      'penalty':['l2'],
                      'max_iter':[1,10,100,1000]}

        clf = GridSearchCV(LogisticRegression(), param_grid)

        matches = df[df['CHAMP.PT'] == c]

        round = 1
        year = 2015

        train = matches[matches['CHAMP.YR'] < 2015].sort_values('DT')
        test = matches[matches['CHAMP.YR'] >= 2015].sort_values('DT')

        while True:

            print(test[test['round'] == round, test['CHAMP.YR'] == year])
            train = train.append(test[['RD' == round and 'CHAMP.YR' == year]])
            test = test['RD' == round+1, 'CHAMP.YR' == year]

            if len(train) == 0:
                if year < 2016:
                    year = year +1
                else:
                    break

            print(len(train), len(test))
            target = 'BTSR'

            features = ['ODDS.RESUME.bts.NO.avg','ODDS.RESUME.bts.YES.avg']

            X = train[features]
            Y = train[target]

            clf.fit(X,Y)

            X = test[features]
            Y = test[target]

            print(c,clf.score(X,Y)*100)



get_dataframe()
learn()