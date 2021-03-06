from argparse import ArgumentParser

from pandas.io.json import json_normalize
from pymongo import UpdateOne, InsertOne, IndexModel, ASCENDING
from core.logger import logging, log_in_out
import numpy as np
import pandas as pd
import json
import sys
import math
from core.config_loader import Database

session = Database.get_session_sportmonks()

# create logger with 'spam_application'
logger = logging.getLogger('dataset_preprocessing')

EXTRACT = 1
CHECK = 2
PROCESS = 3
ALL = 4

SELECTED = 'selected'
STATUS_ERROR = 'status_error'
COL_MATCHES = 'matches'
COL_FIXTURES = 'fixtures'


class PreProcessingException(Exception):
    pass


class RequiredKeyException(PreProcessingException):

    def __init__(self):
        self.msg = 'REQUIRED_KEY_EXCEPTION'
        self.code = 'RQK'


class NoneMinutesException(PreProcessingException):

    def __init__(self):
        self.msg = 'NONE_MINUTES'
        self.code = 'NMI'


class TeamIdentifierException(PreProcessingException):

    def __init__(self):
        self.msg = 'TEAM_IDENTIFIER'
        self.code = 'TID'


class AmountNullEventException(PreProcessingException):

    def __init__(self):
        self.msg = 'AMOUNT_NULL_EVENT_EXCEPTION'
        self.code = 'ANG'


class TrendAmountInvalidEventException(PreProcessingException):

    def __init__(self):
        self.msg = 'AMOUNT_NEGATIVE_EVENT_EXCEPTION'
        self.code = 'ANU'


class TrendMinuteInvalidException(PreProcessingException):

    def __init__(self):
        self.msg = 'TREND_MINUTE_INVALID_EXCEPTION'
        self.code = 'MIV'


class CardMinuteInvalidException(PreProcessingException):

    def __init__(self):
        self.msg = 'CARD_MINUTE_INVALID_EXCEPTION'
        self.code = 'CMI'


class TrendGoalsIsNotEqualToScoreboardException(PreProcessingException):

    def __init__(self):
        self.msg = 'TREND_GOALS_NOT_EQUAL_TO_SCOREBOARD'
        self.code = 'GNS'


def is_to_submit(idx, batch_size=5000):
    return idx >= batch_size and idx % batch_size == 0


def get_columns(stats_list):
    columns = list()
    for s in stats_list:
        columns.append(s + '_home')
        columns.append(s + '_away')
    return columns


def get_matches(config):

    if config.limit is None:
        matches = session.get_collection(COL_MATCHES).find({'selected': True}, no_cursor_timeout=True).skip(config.skip)
    else:
        matches = session.get_collection(COL_MATCHES).find({'selected': True}, no_cursor_timeout=True).skip(config.skip).limit(config.limit)
    return matches


def stat_proportion(row, stat):

    if row[stat + "_home"] == row[stat + "_away"]:
        return 0.5
    else:
        return row[stat + "_home"] / (row[stat + "_away"] + row[stat + "_home"])


@log_in_out
def extract_matches(config):

    """
    Initial insertion in the collection matches
    """
    if config.limit is not None:
        fixtures = session.get_collection(COL_FIXTURES).find({'trends.data.type': 'possession'}).limit(config.limit)
    else:
        fixtures = session.get_collection(COL_FIXTURES).find({'trends.data.type': 'possession'})

    requests = list()

    if config.drop:
        logger.info('Removing the collection %s ' % COL_MATCHES)
        session.get_collection(COL_MATCHES).drop()

    index_1 = IndexModel([("id", ASCENDING)], unique=1, name="id_unique")
    index_2 = IndexModel([("date", ASCENDING)], name="date_ascending")
    session.get_collection(COL_MATCHES).create_indexes([index_1, index_2])

    for idx, f in enumerate(fixtures):

        logger.debug('[%i] Extracting Match: %i ' % (idx, f['id']))
        match = extract_single_match(f)

        requests.append(InsertOne(match))

        if is_to_submit(idx, config.bulk_size):
            logger.info('Saving in the database block [%i-%i]' % (idx - config.bulk_size, idx))
            save_db(requests, config)
            requests = list()

    save_db(requests, config)


def extract_single_match(f):

    match = dict()
    match['id'] = f['id']
    try:
        try:
            match['team_home_id'] = f['localteam_id']
            match['team_away_id'] = f['visitorteam_id']
            match['team_home_leg_id'] = f['localTeam']['data']['legacy_id']
            match['team_away_leg_id'] = f['visitorTeam']['data']['legacy_id']
            match['league_id'] = f['league_id']
            match['season_id'] = f['season_id']
            match['score_home'] = f['scores']['localteam_score']
            match['score_away'] = f['scores']['visitorteam_score']
            match['date'] = f['time']['starting_at']['date']
            match['time'] = f['time']['starting_at']['time']
            match['observed'] = dict()
        except KeyError:
            raise RequiredKeyException

        if match['score_home'] > match['score_away']:
            match['result'] = 'H'
            observed = {'home': 1, 'draw': 0, 'away': 0}
            match['observed'] = observed

        elif match['score_home'] == match['score_away']:
            match['result'] = 'D'
            observed = {'home': 0, 'draw': 1, 'away': 0}
            match['observed'] = observed
        else:
            match['result'] = 'A'
            observed = {'home': 0, 'draw': 0, 'away': 1}
            match['observed'] = observed

        if f['time']['minute'] is None:
            raise NoneMinutesException
        else:
            if f['time']['injury_time'] is None:
                match['minutes'] = f['time']['minute']
            else:
                match['minutes'] = f['time']['minute'] + f['time']['injury_time']

            match['minute_max'] = match['minutes']

            match[SELECTED] = True

    except (RequiredKeyException, NoneMinutesException) as ex:
        match[STATUS_ERROR] = ex.code
        match[SELECTED] = False

    return match


@log_in_out
def check_all_matches(config):
    """
    status_error = 2, if the team_id of trends is different of team_home_id and team_away_id
    """
    matches = get_matches(config)

    requests = list()
    index_1 = IndexModel([("selected", ASCENDING)], name="selected")
    index_2 = IndexModel([("status_error", ASCENDING)], name="status_error")
    session.get_collection(COL_MATCHES).create_indexes([index_1, index_2])

    for idx, m in enumerate(matches):

        logger.debug('[%i] Checking Match: %i ' % (idx, m['id']))
        update = check_single_match(m)
        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if is_to_submit(idx, config.bulk_size):
            logger.info('Saving in the database block [%i-%i]' % (idx-config.bulk_size, idx))
            save_db(requests, config)
            requests = list()

    save_db(requests, config)


def check_single_match(match):

    update = {}
    fixture = session.get_collection(COL_FIXTURES).find_one({'id': match['id']})
    trends = fixture['trends']['data']
    cards = fixture['cards']['data']
    home_ids = [match['team_home_id'], match['team_home_leg_id']]
    away_ids = [match['team_away_id'], match['team_away_leg_id']]
    update['minute_max'] = match['minute_max']

    try:
        for trend in trends:
            check_teams(away_ids, home_ids, trend)

        for card in cards:
            check_teams(away_ids, home_ids, card)

        goals_home = 0
        goals_away = 0

        for trend in trends:

            df = check_trend(trend, update)

            if trend['type'] == 'goals':
                if trend['team_id'] in home_ids:
                    goals_home = df.iloc[-1]['amount']
                elif trend['team_id'] in away_ids:
                    goals_away = df.iloc[-1]['amount']
                else:
                    raise TeamIdentifierException

        if goals_home != match['score_home'] or goals_away != match['score_away']:
            raise TrendGoalsIsNotEqualToScoreboardException

        check_cards(cards, update)

        update['selected'] = True

    except (TeamIdentifierException,
            TrendMinuteInvalidException,
            TrendAmountInvalidEventException,
            TrendGoalsIsNotEqualToScoreboardException) as ex:
        logger.debug('Exception [%s] raised. Match: %i' % (ex.code, match['id']))
        update['status_error'] = ex.code
        update['selected'] = False

    return update


@log_in_out
def check_match_by_id(config):

    match = session.get_collection(COL_MATCHES).find_one({'id': config.match_id})

    update = check_single_match(match)
    requests = list()
    requests.append(UpdateOne({'_id': match['_id']}, {'$set': update}))
    save_db(requests, config)


def check_teams(home_ids, away_ids, target):

    try:
        team_id = int(target['team_id'])
        if team_id not in home_ids and team_id not in away_ids:
            logger.debug('Exception: Invalid Team in trends')
            raise TeamIdentifierException
    except TypeError:
        raise TeamIdentifierException


def check_trend(trend, update):

    analyses = trend['analyses']
    df = pd.DataFrame.from_dict(analyses)
    df.minute = df.minute.astype(int)
    df.amount = df.amount.astype(int)
    df.set_index('minute', inplace=True)
    df.sort_index(inplace=True)
    df.astype('int64')

    if df.index.values.item(0) < 0 or df.index.values.item(-1) > 150:
        raise TrendMinuteInvalidException
    elif any(df['amount'] < 0):
        raise TrendAmountInvalidEventException

    if df.index.values.item(-1) > update['minute_max']:
        update['minute_max'] = df.index.values.item(-1)

    return df


def check_cards(cards, update):

    def get_real_minute(row):
        if row['extra_minute'] is not None and not np.isnan(row['extra_minute']):
            return int(row['minute'] + row['extra_minute'] - 1)
        else:
            return int(row['minute'] - 1)

    if len(cards) > 0:

        df = pd.DataFrame.from_records(cards)
        df = df[df['minute'].notna()]

        if len(df) > 0:
            df['minute'] = df['minute'].astype(int)
            df['minute'] = df.apply(get_real_minute, axis=1)

            df.set_index('minute', inplace=True)
            df.sort_index(inplace=True)

            if len(df) > 0:

                if df.index.values.item(0) < 0:
                    raise CardMinuteInvalidException

                if df.index.values.item(-1) > update['minute_max']:
                    update['minute_max'] = df.index.values.item(-1)


def save_db(requests, config):

    if config.save and len(requests) > 0:
        session.get_collection(COL_MATCHES).bulk_write(requests)


@log_in_out
def process_trends(config):

    matches = get_matches(config)
    requests = list()

    for idx, m in enumerate(matches):

        logger.debug('[%i] Processing Match Trends: %i ' % (idx, m['id']))
        update = process_trends_by_match(m)
        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if is_to_submit(idx, config.bulk_size):
            logger.info('Saving in the database block [%i-%i]' % (idx - config.bulk_size, idx))
            save_db(requests, config)
            requests = list()

    save_db(requests, config)


def process_trends_by_match(match):

    update = dict()

    stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']
    columns = get_columns(stats_list)
    minute_max = match['minute_max']

    accum = create_events_dataframe(columns, minute_max)
    deaccum = create_events_dataframe(columns, minute_max)

    fixture = session.get_collection(COL_FIXTURES).find_one({'id': match['id']})
    trends = fixture['trends']['data']
    home_ids = [match['team_home_id'], match['team_home_leg_id']]
    away_ids = [match['team_away_id'], match['team_away_leg_id']]
    for trend in trends:

        team_id = int(trend['team_id'])

        if team_id in home_ids:
            add_trend('home', trend, accum)
        elif team_id in away_ids:
            add_trend('away', trend, accum)
        else:
            raise Exception

    for col in accum.columns:
        fill_empty_values(col, accum, deaccum)

    sub = pd.DataFrame()
    ratio = pd.DataFrame()

    for col in stats_list:
        home = '_'.join([col, 'home'])
        away = '_'.join([col, 'away'])
        sub[col] = accum[home] - accum[away]
        ratio[col] = round(accum.apply(lambda row: stat_proportion(row, col), axis=1), 3)

    ratio['ball_possession'] = accum['possession_home']
    stats_accum = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']
    col_accum = get_columns(stats_accum)

    accum_json = accum[col_accum].to_json()
    accum_dict = json.loads(accum_json)
    update['accum_trends'] = accum_dict

    deaccum_json = deaccum[col_accum].to_json()
    deaccum_dict = json.loads(deaccum_json)
    update['deaccum_trends'] = deaccum_dict

    ratio_json = ratio.to_json()
    ratio_dict = json.loads(ratio_json)
    update['ratio_trends'] = ratio_dict

    sub_json = sub.to_json()
    sub_dict = json.loads(sub_json)
    update['sub_trends'] = sub_dict

    return update


@log_in_out
def process_cards(config):

    matches = get_matches(config)

    requests = list()

    for idx, m in enumerate(matches):

        logger.debug('[%i] Processing Match Cards: %i ' % (idx, m['id']))
        update = process_cards_by_match(m)
        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if is_to_submit(idx, config.bulk_size):
            logger.info('Saving in the database block [%i-%i]' % (idx - config.bulk_size, idx))
            save_db(requests, config)
            requests = list()

    save_db(requests, config)


def process_cards_by_match(m):

    update = dict()
    stats_list = ['yellow_cards', 'red_cards']
    columns = get_columns(stats_list)

    minute_max = m['minute_max']
    deaccum = create_events_dataframe(columns, minute_max)

    fixture = session.get_collection(COL_FIXTURES).find_one({'id': m['id']})
    cards = fixture['cards']['data']
    home_ids = [m['team_home_id'], m['team_home_leg_id']]
    away_ids = [m['team_away_id'], m['team_away_leg_id']]

    for card_event in cards:

        team_id = int(card_event['team_id'])

        if team_id in home_ids:
            add_card('home', card_event, deaccum)
        elif team_id in away_ids:
            add_card('away', card_event, deaccum)
        else:
            raise Exception

    deaccum.fillna(0, inplace=True)
    accum = deaccum.cumsum()

    sub = pd.DataFrame()
    ratio = pd.DataFrame()

    for col in stats_list:
        home = '_'.join([col, 'home'])
        away = '_'.join([col, 'away'])
        sub[col] = accum[home] - accum[away]
        ratio[col] = round(accum.apply(lambda row: stat_proportion(row, col), axis=1), 3)

    accum_events_json = accum.to_json()
    accum_events_dict = json.loads(accum_events_json)
    update['accum_cards'] = accum_events_dict

    deaccum_events_json = deaccum.to_json()
    deaccum_events_dict = json.loads(deaccum_events_json)
    update['deaccum_cards'] = deaccum_events_dict

    ratio_json = ratio.to_json()
    ratio_dict = json.loads(ratio_json)
    update['ratio_cards'] = ratio_dict

    sub_json = sub.to_json()
    sub_dict = json.loads(sub_json)
    update['sub_cards'] = sub_dict

    return update


def add_trend(local, trend, accum_events):

    trends = trend['analyses']
    col = trend['type'] + "_" + local

    for trend in trends:

        minute = int(trend['minute'])
        amount = int(trend['amount'])
        accum_events.loc[minute, col] = amount


def add_card(local, card_event, deaccum_events):

    def get_minute_of_card(event):

        normal_minute = event['minute']
        if normal_minute == 90:
            extra_minute = event['extra_minute']
            if extra_minute is not None:
                normal_minute = normal_minute + extra_minute

        return normal_minute - 1

    try:
        minute = get_minute_of_card(card_event)
    except TypeError:
        return

    if card_event['type'] == 'yellowcard':
        col = 'yellow_cards_' + local
    else:
        col = 'red_cards_' + local

    if np.isnan(deaccum_events.loc[minute, col]):
        deaccum_events.loc[minute, col] = 1
    else:
        deaccum_events.loc[minute, col] += 1


def create_events_dataframe(columns, minutes):

    df = pd.DataFrame(columns=columns)
    df['minute'] = range(0, minutes + 1)
    df.set_index('minute', inplace=True)
    return df


def fill_empty_values(col, accum_events, deaccum_events):

    if np.isnan(accum_events.loc[0, col]):
        accum_events.loc[0, col] = 0
        lvv = 0
    else:
        lvv = accum_events.loc[0, col]

    for idx, value in accum_events[col][1:].iteritems():
        if np.isnan(value):
            accum_events.loc[idx, col] = lvv
        lvv = accum_events.loc[idx, col]

    if 'possession' not in col:
        deaccum_events.loc[0, col] = accum_events.loc[0, col]
        for idx, value in accum_events[col][1:].iteritems():
            deaccum_events.loc[idx, col] = accum_events.loc[idx, col] - accum_events.loc[idx - 1, col]


def process_matches(config):

    process_trends(config)
    process_cards(config)
    process_odds(config)


@log_in_out
def process_match_by_id(config):
    match = session.get_collection(COL_MATCHES).find_one({'id': config.match_id})

    update = process_cards_by_match(match)
    requests = list()
    requests.append(UpdateOne({'_id': match['_id']}, {'$set': update}))
    save_db(requests, config)

    update = process_trends_by_match(match)
    requests = list()
    requests.append(UpdateOne({'_id': match['_id']}, {'$set': update}))
    save_db(requests, config)


def process_odds(config):

    def get_summary_by_column():

        if tp == 'o':
            col = 'value'
        else:
            col = 'prob'

        method_to_call = getattr(df[col], method)
        result = method_to_call()

        return round(result, 5)

    matches = get_matches(config)
    requests = list()

    for idx, m in enumerate(matches):

        logger.debug('[%i] Processing Match Odds: %i ' % (idx, m['id']))
        fixture = session.get_collection('fixtures').find_one({'id': m['id'], 'odds.data.id': 1},
                                                              {'odds.data.$': 1})
        update = dict()
        odds_summary = dict()

        try:
            od_1 = fixture['odds']['data']
        except TypeError:
            logger.debug('[%i] Match without odds: %i ' % (idx, m['id']))
            update['with_odds'] = False
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))
            continue

        odds_list = list()
        for d1 in od_1:
            od_2 = d1['bookmaker']['data']
            for d2 in od_2:
                odds = d2['odds']['data']
                for o in odds:
                    odds_list.append(o)

        df = pd.DataFrame.from_dict(json_normalize(odds_list), orient='columns')
        df['value'] = pd.to_numeric(df['value'])

        df['prob'] = 1 / df['value']

        home = df[df['label'] == '1']
        draw = df[df['label'] == 'X']
        away = df[df['label'] == '2']

        odds_summary['n_books'] = len(home)

        for label, df in zip(['home', 'draw', 'away'], [home, draw, away]):
            for method in ['max', 'min', 'mean', 'std']:
                for tp in ['o', 'p']:

                    field = '_'.join(([tp, method, label]))
                    odds_summary[field] = get_summary_by_column()

        if (math.isnan(odds_summary['o_mean_home']) or odds_summary['o_mean_home'] < 1) or \
                (math.isnan(odds_summary['o_mean_draw']) or odds_summary['o_mean_draw'] < 1) or \
                (math.isnan(odds_summary['o_mean_away']) or odds_summary['o_mean_away'] < 1):

            logger.debug('[%i] Exception in odds values of Match: %i ' % (idx, m['id']))
            update['with_odds'] = False
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))
            continue

        raw = [odds_summary['p_mean_home'], odds_summary['p_mean_draw'], odds_summary['p_mean_away']]
        pred_final = [float(i) / sum(raw) for i in raw]

        odds_summary['p_norm_home'] = round(pred_final[0], 5)
        odds_summary['p_norm_draw'] = round(pred_final[1], 5)
        odds_summary['p_norm_away'] = round(pred_final[2], 5)

        idx_prediction = np.argmax(pred_final)

        if idx_prediction == 0:
            fav = 'home'
        elif idx_prediction == 1:
            fav = 'draw'
        else:
            fav = 'away'

        for method in ['max', 'min', 'mean', 'std']:
            for tp in ['o', 'p']:

                field = '_'.join(([tp, method, fav]))
                field_fav = '_'.join(([tp, method, 'fav']))
                odds_summary[field_fav] = odds_summary[field]

        update['with_odds'] = True
        update['odds'] = odds_summary

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if is_to_submit(idx, config.bulk_size):
            logger.info('Saving in the database block [%i-%i]' % (idx - config.bulk_size, idx))
            save_db(requests, config)
            requests = list()

    save_db(requests, config)


def main():

    parser = ArgumentParser()

    parser.add_argument("-a", "--action", dest="action", choices=[EXTRACT, CHECK, PROCESS, ALL],
                        help="1- extract | 2- single | 3- final ", type=int)

    parser.add_argument("-d", "--drop", action="store_true", dest="drop", default=False,
                        help="drop the collection of matches")

    parser.add_argument("-s", "--save", action="store_true", dest="save", default=False,
                        help="save the changes in the database")

    parser.add_argument("-m", "--match", dest="match_id",
                        help="inform a valid match ID ", type=int)

    parser.add_argument("-l", "--limit", dest="limit",
                        help="inform a number of matches", type=int)

    parser.add_argument("-b", "--bulk_size", dest="bulk_size", default=1000,
                        help="the bulk size to save in the database", type=int)

    parser.add_argument("-k", "--method", dest="method",
                        help="invoke a method by name")

    parser.add_argument("-i", "--skip", dest="skip", default=0,
                        help="the number of matches to skip", type=int)

    config = parser.parse_args()

    if config.method is not None:
        method_to_call = getattr(sys.modules[__name__], config.method)
        method_to_call(config)
    elif config.action == EXTRACT:
        extract_matches(config)
    elif config.action == CHECK:
        if config.match_id is not None:
            check_match_by_id(config)
        else:
            check_all_matches(config)
    elif config.action == PROCESS:
        if config.match_id is not None:
            process_match_by_id(config)
        else:
            process_matches(config)
    elif config.action == ALL:
        extract_matches(config)
        check_all_matches(config)
        process_matches(config)


if __name__ == "__main__":
    main()
