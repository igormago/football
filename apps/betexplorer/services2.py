from modules.betexplorer2.notations import MatchNotation, PerformanceNotation, Odd
from core.database import session_betexplorer_2 as session
import numpy as np


def set_round_groups():
    DataBase_matches = session.matches
    matches = DataBase_matches.list()
    bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

    for idx, m in enumerate(matches):

        res_ml = MatchNotation.set_money_line_result(m)
        res_bts = MatchNotation.set_both_teams_to_score_result(m)

        bulk_matches.list({'_id': m['_id']}).update({'$set': {MatchNotation.MONEY_LINE_RESULT: res_ml,
                                                              MatchNotation.BOTH_TEAMS_TO_SCORE_RESULT: res_bts}})

        if idx % 300 == 0:
            bulk_matches.execute()
            bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

    bulk_matches.execute()


def set_results():
    DataBase_matches = session.matches
    matches = DataBase_matches.list()
    bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

    for idx, m in enumerate(matches):

        res_ml = MatchNotation.set_money_line_result(m)
        res_bts = MatchNotation.set_both_teams_to_score_result(m)

        bulk_matches.list({'_id': m['_id']}).update({'$set': {MatchNotation.MONEY_LINE_RESULT: res_ml,
                                                              MatchNotation.BOTH_TEAMS_TO_SCORE_RESULT: res_bts}})

        if idx % 300 == 0:
            bulk_matches.execute()
            bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

    bulk_matches.execute()


def sumarize_odds():
    DataBase_matches = session.matches
    matches = DataBase_matches.list()
    bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

    for idx, m in enumerate(matches):

        resume = {}
        odds = m[MatchNotation.ODDS]
        ml = odds[Odd.MONEY_LINE]
        lhome, ldraw, laway = [], [], []

        for i in ml:

            if i[Odd.ACTIVE_HOME] and i[Odd.ACTIVE_DRAW] and i[Odd.ACTIVE_AWAY]:
                total = i[Odd.HOME] + i[Odd.DRAW] + i[Odd.AWAY]

                pHome = i[Odd.HOME] / total
                pDraw = i[Odd.DRAW] / total
                pAway = i[Odd.AWAY] / total

                lhome.append(pHome)
                ldraw.append(pDraw)
                laway.append(pAway)

        values = {}

        values[Odd.HOME] = extract_resume(lhome)
        values[Odd.DRAW] = extract_resume(ldraw)
        values[Odd.AWAY] = extract_resume(laway)

        resume[Odd.MONEY_LINE] = values

        bts = odds[Odd.BOTH_TEAMS_TO_SCORE]
        lYes, lNo = [], []

        for i in bts:

            if i[Odd.ACTIVE_YES] and i[Odd.ACTIVE_NO]:
                total = i[Odd.YES] + i[Odd.NO]

                pYes = i[Odd.YES] / total
                pNo = i[Odd.NO] / total

                lYes.append(pYes)
                lNo.append(pNo)

        values = {}
        values[Odd.YES] = extract_resume(lYes)
        values[Odd.NO] = extract_resume(lNo)

        resume[Odd.BOTH_TEAMS_TO_SCORE] = values

        update_field = MatchNotation.ODDS + '.' + Odd.RESUME

        bulk_matches.list({'_id': m['_id']}).update({'$set': {update_field: resume}})

        if idx % 300 == 0:
            bulk_matches.execute()
            bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

    bulk_matches.execute()


def extract_resume(values):
    dict = {}
    dict['count'] = len(values)

    if dict['count'] > 0:
        dict['avg'] = round(float(np.average(values)), 4)
        dict['max'] = round(float(np.max(values)), 4)
        dict['min'] = round(float(np.min(values)), 4)
        dict['std'] = round(float(np.std(values)), 4)
    else:
        dict['avg'], dict['max'], dict['min'], dict['std'] = None, None, None, None

    return dict


def evaluate_favorite_ML():
    DataBase_matches = session.matches
    matches = DataBase_matches.list()
    bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

    for idx, m in enumerate(matches):

        try:
            ml_fmu = Odd.set_money_line_FMU(m)
            ml = {}

            for idx2, i in enumerate(Odd.FMU_LIST):
                ml[i] = {}
                ml[i][Odd.ODD] = ml_fmu[idx2]
                ml[i][Odd.VALUE] = ml_fmu[idx2 + 3]

            bulk_matches.list({'_id': m['_id']}).update({'$set': {Odd.ML_FMU: ml}})

            if idx % 300 == 0:
                print(idx)
                bulk_matches.execute()
                bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

        except Exception as e:
            pass

    bulk_matches.execute()


def evaluate_favorite_BTS():
    DataBase_matches = session.matches
    matches = DataBase_matches.list()
    bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

    for idx, m in enumerate(matches):

        try:

            bts_fu = Odd.set_both_teams_to_score_FU(m)
            bts = {}

            for idx2, i in enumerate(Odd.FU_LIST):
                bts[i] = {}
                bts[i][Odd.ODD] = bts_fu[idx2]
                bts[i][Odd.VALUE] = bts_fu[idx2 + 2]

            bulk_matches.list({'_id': m['_id']}).update({'$set': {Odd.BTS_FU: bts}})
            if idx % 300 == 0:
                print(idx)
                bulk_matches.execute()
                bulk_matches = DataBase_matches.initialize_unordered_bulk_op()

        except Exception as e:
            pass

    bulk_matches.execute()

    #
    # fmu = Odd.get_FMU(m)
    #
    # data = {}
    #
    # for idx, i in enumerate(Odds.ML_FMU):
    #
    #     data[i] = {}
    #     data[i][Odds.ODD] = fmu[idx]
    #     data[i][Odds.VALUE] = fmu[idx+3]
    #
    # session.matches.update({'_id': m['_id']},
    #                        {'$set': {Odds.FMU: data}}, upsert=False)



# sumarize_odds()
# evaluate_favorite_ML()
# evaluate_favorite_BTS()
evaluate_hits()
