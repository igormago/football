from pymongo import UpdateOne, InsertOne

from core.databases import SportMonksDatabase as DataBase
import pandas as pd
import json
import numpy as np
from pandas.io.json import json_normalize
from datetime import datetime

session = DataBase.get_session()


def submit(idx):
    batch_size = 5000
    return idx >= batch_size and idx % batch_size == 0


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


class StreamIsNotEqualsToScoreException(PreProcessingException):

    def __init__(self):
        self.msg = 'STREAM_IS_NOT_EQUALS_TO_SCORE'
        self.code = 'SES'


class TimeEventGreaterTimeMatchException(PreProcessingException):

    def __init__(self):
        self.msg = 'TIME_EVENT_GREATER_TIME_MATCH'
        self.code = 'EGT'


class AmountNullEventException(PreProcessingException):

    def __init__(self):
        self.msg = 'AMOUNT_NULL_EVENT_EXCEPTION'
        self.code = 'ANE'


class AmountNegativeEventException(PreProcessingException):

    def __init__(self):
        self.msg = 'AMOUNT_NEGATIVE_EVENT_EXCEPTION'
        self.code = 'ANG'


class MinuteNegativeException(PreProcessingException):

    def __init__(self):
        self.msg = 'MINUTE_NEGATIVE_EXCEPTION'
        self.code = 'MNE'


# PART 1
def extract_matches():
    """
    Initial insertion in the collection matches
    """
    fixtures = session.get_collection('fixtures').find({'trends.data.type': 'possession'})

    requests = list()

    for idx, f in enumerate(fixtures):
        #print(idx, f['id'])

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
            except KeyError:
                raise RequiredKeyException

            match['observed'] = dict()

            if match['score_home'] > match['score_away']:
                match['result'] = 'H'
                match['observed']['home'] = 1
                match['observed']['draw'] = 0
                match['observed']['away'] = 0
            elif match['score_home'] == match['score_away']:
                match['result'] = 'D'
                match['observed']['home'] = 0
                match['observed']['draw'] = 1
                match['observed']['away'] = 0
            else:
                match['result'] = 'A'
                match['observed']['home'] = 0
                match['observed']['draw'] = 0
                match['observed']['away'] = 1

            if f['time']['minute'] is None:
                raise NoneMinutesException
            else:
                if f['time']['injury_time'] is None:
                    match['minutes'] = f['time']['minute']
                else:
                    match['minutes'] = f['time']['minute'] + f['time']['injury_time']

                match['minute_max'] = match['minutes']

        except (RequiredKeyException, NoneMinutesException) as ex:
            match['status_error'] = ex.code

        requests.append(InsertOne(match))

        if submit(idx):
            print(datetime.now(), idx, f['id'])
            session.get_collection('matches').bulk_write(requests)
            requests = list()

    session.get_collection('matches').bulk_write(requests)


# PART 2
def matches_status_error_2():
    """
    status_error = 2, if the team_id of trends is different of team_home_id and team_away_id
    """

    matches = session.get_collection('matches').find()
    requests = list()

    for idx, m in enumerate(matches):

        stat = {}
        print(idx, m['fixture_id'])
        fixture = session.get_collection('fixtures').find_one({'id': m['fixture_id']})
        trends = fixture['trends']['data']
        c = 0
        home_ids = [m['team_home_id'], m['team_home_leg_id'] ]
        away_ids = [m['team_home_id'], m['team_away_leg_id']]
        for trend in trends:
            if (trend['team_id'] not in home_ids) and (trend['team_id'] not in away_ids):
                c = c + 1

        if c > 0:
            print("update")
            stat['status_error'] = 2
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': stat}))

    session.get_collection('matches').bulk_write(requests)

# PART 3
def extract_events_by_minute_old():

    def fill_dataframe(trend, local):

        regs = trend['analyses']
        col = trend['type'] + "_" + local

        last_value = 0

        if trend['type'] != 'possession':

            for r in regs:
                minute = int(r['minute'])
                value = int(r['amount'])
                events.loc[minute, col] = value - last_value
                last_value = value

    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = []
    batch_size = 1000

    for idx, m in enumerate(matches):

        print(idx, m['fixture_id'])
        stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']
        columns = ['minute']

        for s in stats_list:
            columns.append(s + '_home')
            columns.append(s + '_away')

        minutes = m['minutes']
        events = pd.DataFrame(columns=columns)
        events['minute'] = range(0, minutes+1)
        events.set_index('minute', inplace=True)
        events.fillna(0, inplace=True)

        # keys = events.keys()
        # for k in keys:
        #     events[k] = 0

        fixture = session.get_collection('fixtures').find_one({'id': m['fixture_id']})
        trends = fixture['trends']['data']

        for trend in trends:

            home_ids = [m['home_team_id'], m['home_team_leg_id'] ]
            away_ids = [m['away_team_id'], m['away_team_leg_id']]

            if trend['team_id'] in home_ids:
                fill_dataframe(trend, 'home')
            elif trend['team_id'] in away_ids:
                fill_dataframe(trend, 'away')
            else:
                raise Exception

        u = dict()

        events_stats = events.to_json()
        dict_stats = json.loads(events_stats)
        u['events'] = dict_stats

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': u}))

        if idx >= batch_size and idx % batch_size == 0:
            session.get_collection('matches').bulk_write(requests)
            requests = []

    session.get_collection('matches').bulk_write(requests)


# PART 4
def extract_counters():

    def fill_dataframe(trend, local):

        regs = trend['analyses']
        col = trend['type'] + "_" + local

        if (trend['type'] != 'possession'):

            initial_minute = int(regs[0]['minute'])
            value = int(regs[0]['amount'])

            if initial_minute > 0:
                events.loc[0:initial_minute - 1, col] = 0

            for r in regs[1:]:
                final_minute = int(r['minute'])
                events.loc[initial_minute:final_minute - 1, col] = value
                value = int(r['amount'])
                initial_minute = final_minute

            events.loc[initial_minute:len(events) + 1, col] = value

    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = []
    batch_size = 1000

    for idx, m in enumerate(matches):

        print(idx, m['fixture_id'])
        stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']
        columns = ['minute']

        for s in stats_list:
            columns.append(s + '_home')
            columns.append(s + '_away')

        minute_max = m['minute_max']
        events = pd.DataFrame(columns=columns)
        events['minute'] = range(0, minute_max + 1)
        events.set_index('minute', inplace=True)
        keys = events.keys()
        for k in keys:
            events[k] = 0

        fixture = session.get_collection('fixtures').find_one({'id': m['fixture_id']})
        trends = fixture['trends']['data']

        for trend in trends:

            if trend['team_id'] == m['team_home_id']:
                fill_dataframe(trend, 'home')
            else:
                fill_dataframe(trend, 'away')

        u = dict()
        events_stats = events.to_json()
        dict_stats = json.loads(events_stats)
        u['counters'] = dict_stats

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': u}))

        if idx >= batch_size and idx % batch_size == 0:
            session.get_collection('matches').bulk_write(requests)
            requests = []

    session.get_collection('matches').bulk_write(requests)


# PART 5
def extract_propotions():

    def get_ratio(row, stat):

        if row[stat + "_home"] == row[stat + "_away"]:
            return 0.5
        else:
            return row[stat + "_home"] / (row[stat + "_away"] + row[stat + "_home"])

    def fill_dataframe(trend, local):

        regs = trend['analyses']
        col = trend['type']

        initial_minute = int(regs[0]['minute'])
        value = int(regs[0]['amount'])

        if initial_minute > 0:
            df.loc[0:initial_minute - 1, col] = 0

        for r in regs[1:]:
            final_minute = int(r['minute'])
            df.loc[initial_minute:final_minute - 1, col] = round(value / 100, 4)
            value = int(r['amount'])
            initial_minute = final_minute

        df.loc[initial_minute:len(df) + 1, col] = round(value / 100, 4)

    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = []
    batch_size = 1000

    for idx, m in enumerate(matches):

        print(idx, m['fixture_id'])
        stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']

        trends = m['counters']

        df = pd.DataFrame()
        for s in stats_list:

            if s != 'possession':
                df[s + '_home'] = trends[s + '_home'].values()
                df[s + '_away'] = trends[s + '_away'].values()

                df[s] = round(df.apply(lambda row: get_ratio(row, s), axis=1), 3)
            else:
                fixture = session.get_collection('fixtures').find_one({'id': m['fixture_id']})
                trends = fixture['trends']['data']

                for trend in trends:
                    if trend['type'] == 'possession':
                        if trend['team_id'] == m['team_home_id']:
                            fill_dataframe(trend, 'home')

        u = dict()
        try:
            events_stats = df[stats_list].to_json()
            dict_stats = json.loads(events_stats)
            u['ratios'] = dict_stats

            requests.append(UpdateOne({'_id': m['_id']}, {'$set': u}))
        except:
            u['status_error'] = 2

            requests.append(UpdateOne({'_id': m['_id']}, {'$set': u}))

        if idx >= batch_size and idx % batch_size == 0:
            session.get_collection('matches').bulk_write(requests)
            requests = []

    session.get_collection('matches').bulk_write(requests)


# PART 6
def extract_differences():
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = []
    batch_size = 1000

    for idx, m in enumerate(matches):

        print(idx, m['fixture_id'])
        stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']

        trends = m['counters']

        df = pd.DataFrame()
        for s in stats_list:

            if s != 'possession':
                df[s + '_home'] = trends[s + '_home'].values()
                df[s + '_away'] = trends[s + '_away'].values()

                df[s] = df[s + '_home'] - df[s + '_away']

        u = dict()
        try:
            events_stats = df[stats_list].to_json()
            dict_stats = json.loads(events_stats)
            u['difs'] = dict_stats

            requests.append(UpdateOne({'_id': m['_id']}, {'$set': u}))
        except:
            u['status_error'] = 3

            requests.append(UpdateOne({'_id': m['_id']}, {'$set': u}))

        if idx >= batch_size and idx % batch_size == 0:
            session.get_collection('matches').bulk_write(requests)
            requests = []

    session.get_collection('matches').bulk_write(requests)


# PART 7
def extract_cards():

    def get_minute(t):
        minute = t['minute']
        if minute == 90:
            extra_minute = t['extra_minute']
            if extra_minute is not None:
                minute = minute + extra_minute

        return minute-1

    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = []

    for idx, m in enumerate(matches):

        print(idx, m['fixture_id'])
        stats_list = ['cards_yellow', 'cards_red']
        columns = ['minute']

        for s in stats_list:
            columns.append(s + '_home')
            columns.append(s + '_away')

        minute_max = m['minute_max']
        events = pd.DataFrame(columns=columns)
        events['minute'] = range(0, minute_max + 1)
        events.set_index('minute', inplace=True)

        counters = pd.DataFrame(columns=columns)
        counters['minute'] = range(0, minute_max + 1)
        counters.set_index('minute', inplace=True)

        keys = events.keys()
        for k in keys:
            events[k] = 0
            counters[k] = 0

        fixture = session.get_collection('fixtures').find_one({'id': m['fixture_id']})
        trends = fixture['cards']['data']

        cards = dict()
        for k in keys:
            cards[k] = list()

        for t in trends:

            if m['team_home_id'] == int(t['team_id']):

                if t['type'] == 'yellowcard':
                    minute = get_minute(t)
                    cards['cards_yellow_home'].append(minute)
                else:
                    minute = get_minute(t)
                    cards['cards_red_home'].append(minute)

            elif m['team_away_id'] == int(t['team_id']):

                if t['type'] == 'yellowcard':
                    minute = get_minute(t)
                    cards['cards_yellow_away'].append(minute)
                else:
                    minute = get_minute(t)
                    cards['cards_red_away'].append(minute)

        try:
            for k in keys:

                for t in cards[k]:
                    events.loc[t, k] +=1

            for k in keys:
                for t in range(0, minute_max+1):
                    counters.loc[t, k] = sum(events[k][0:t+1])

        except Exception as ex:
            print(m)



    #             fill_dataframe(trend, 'home')
    #         else:
    #             fill_dataframe(trend, 'away')
    #
    #     u = dict()
    #
    #     events_stats = events.to_json()
    #     dict_stats = json.loads(events_stats)
    #     u['events'] = dict_stats
    #
    #     requests.append(UpdateOne({'_id': m['_id']}, {'$set': u}))
    #
    #     if idx >= batch_size and idx % batch_size == 0:
    #         session.get_collection('matches').bulk_write(requests)
    #         requests = []
    #
    # session.get_collection('matches').bulk_write(requests)




def matches_status_error_4():
    # Status = 4, if the trends goals is different of score

    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = []

    for idx, m in enumerate(matches):

        u = dict()
        dif = False
        minute_max = m['minute_max']

        if (m['counters']['goals_home'][str(minute_max)] != m['score_home']) \
                or (m['counters']['goals_away'][str(minute_max)] != m['score_away']):
            dif = True

        if dif:
            print(idx)
            u['status_error'] = 4
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': u}))

    session.get_collection('matches').bulk_write(requests)


def matches_2():

    def fill_dataframe(trend, local):

        amounts = trend['analyses']
        col = trend['type'] + "_" + local

        initial_minute = 0
        value = 0
        last_value = 0

        if len(amounts) > 0:

            reg = amounts[0]
            final_minute = int(reg['minute'])

            if final_minute > 0:
                stats.loc[initial_minute:final_minute - 1, col] = 0
                events.loc[initial_minute:final_minute - 1, col] = 0
                initial_minute = final_minute
                value = int(reg['amount'])

            for reg in amounts[1:]:
                final_minute = int(reg['minute'])
                stats.loc[initial_minute:final_minute - 1, col] = value
                events.loc[initial_minute, col] = value - last_value
                events.loc[initial_minute + 1:final_minute - 1, col] = 0
                initial_minute = final_minute
                last_value = value
                value = int(reg['amount'])

        stats.loc[initial_minute:96, col] = value
        events.loc[initial_minute, col] = value - last_value
        if initial_minute <= 95:
            events.loc[initial_minute + 1:96, col] = 0

    def stat_proportion(row, stat):

        if row[stat + "_home"] == row[stat + "_away"]:
            return 0.5
        else:
            return row[stat + "_home"] / (row[stat + "_away"] + row[stat + "_home"])

    # matches = session.get_collection('matches').find({'status':1, 'ts' : { '$exists': False}})
    matches = session.get_collection('matches').find({'status': 1})

    requests = []
    for idx, m in enumerate(matches):

        print(idx)
        stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']
        columns = ['minute']
        for s in stats_list:
            columns.append(s + '_home')
            columns.append(s + '_away')

        events = pd.DataFrame(columns=columns)
        events.set_index('minute', inplace=True)
        events['minute'] = range(0, 96)

        stats = pd.DataFrame(columns=columns)
        stats.set_index('minute', inplace=True)
        stats['minute'] = range(0, 96)

        fixture = session.get_collection('fixtures').find_one({'id': m['fixture_id']})
        trends = fixture['trends']['data']

        for trend in trends:

            if trend['team_id'] == m['team_home_id']:
                fill_dataframe(trend, 'home')
            else:
                fill_dataframe(trend, 'away')

        keys = stats.keys()
        for k in keys:

            if stats[k].isnull().values.any():
                stats[k] = 0

            if events[k].isnull().values.any():
                events[k] = 0

        stats_cum_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']

        for stat in stats_list:
            col = stat + '_dif'
            stats[col] = stats[stat + "_home"] - stats[stat + "_away"]

        for stat in stats_cum_list:
            col = stat + '_div'
            stats[col] = round(stats.apply(lambda row: stat_proportion(row, stat), axis=1), 3)

        for stat in stats_cum_list:
            col = stat + '_dif'
            events[col] = events[stat + "_home"] - events[stat + "_away"]

        for stat in stats_cum_list:
            col = stat + '_div'
            events[col] = round(events.apply(lambda row: stat_proportion(row, stat), axis=1), 3)

        json_stats = stats.to_json()
        json_events = events.to_json()
        dict_stats = json.loads(json_stats)
        dict_events = json.loads(json_events)

        ts = {}
        ts['cum'] = dict_stats
        ts['stream'] = dict_events

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': ts}))
        if idx >= 500 and idx % 500 == 0:
            session.get_collection('matches').bulk_write(requests)
            requests = []

    session.get_collection('matches').bulk_write(requests)


def matches_set_odds():
    def get_summary_by_column(df, col):

        max_odd = round(df[col].max(), 5)
        min_odd = round(df[col].min(), 5)
        avg_odd = round(df[col].mean(), 5)
        std_odd = round(df[col].std(), 5)

        return max_odd, min_odd, avg_odd, std_odd

    batch_size = 1000
    matches = session.get_collection('matches').find({})
    requests = []

    for idx, m in enumerate(matches):

        print(idx, m)
        fixture = session.get_collection('fixtures').find_one({'id': m['fixture_id'], 'odds.data.id': 1},
                                                              {'odds.data.$': 1})
        try:
            data1 = fixture['odds']['data']

            frame = list()
            for d1 in data1:
                data2 = d1['bookmaker']['data']
                for d2 in data2:
                    odds = d2['odds']['data']
                    for o in odds:
                        frame.append(o)

            df = pd.DataFrame.from_dict(json_normalize(frame), orient='columns')
            df['value'] = pd.to_numeric(df['value'])

            df['prob'] = 1 / df['value']

            home = df[df['label'] == '1']
            draw = df[df['label'] == 'X']
            away = df[df['label'] == '2']

            up = dict()
            up['n_books'] = len(home)

            t = 'o'
            r = 'home'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                home,
                'value')
            r = 'draw'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                draw,
                'value')
            r = 'away'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                away,
                'value')

            t = 'p'
            r = 'home'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                home,
                'prob')
            r = 'draw'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                draw,
                'prob')
            r = 'away'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                away,
                'prob')
            raw = [up['p_avg_home'], up['p_avg_draw'], up['p_avg_away']]
            pred_final = [float(i) / sum(raw) for i in raw]

            up['p_norm_home'] = round(pred_final[0], 5)
            up['p_norm_draw'] = round(pred_final[1], 5)
            up['p_norm_away'] = round(pred_final[2], 5)

            fidx = np.argmax(pred_final)

            if fidx == 0:
                fav = 'home'
            elif fidx == 1:
                fav = 'draw'
            else:
                fav = 'away'

            up['o_max_fav'] = up['o_max_' + fav]
            up['o_avg_fav'] = up['o_avg_' + fav]
            up['p_max_fav'] = up['p_max_' + fav]
            up['p_avg_fav'] = up['p_avg_' + fav]

            odds = dict()
            odds['odds'] = up
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': odds}))

            if idx >= batch_size and idx % batch_size == 0:
                session.get_collection('matches').bulk_write(requests)
                requests = []

        except:
            print('Não tem odds')

    print(requests)
    session.get_collection('matches').bulk_write(requests)


def matches_insert_per_minute():

    def fill_dataframe(trend, local):

        amounts = trend['analyses']
        col = trend['type'] + "_" + local

        initial_minute = 0
        value = 0
        last_value = 0

        if len(amounts) > 0:
            reg = amounts[0]
            final_minute = int(reg['minute'])

            if final_minute > 0:
                stats.loc[initial_minute:final_minute - 1, col] = 0
                initial_minute = final_minute
                value = int(reg['amount']) - last_value
                last_value = value

            for reg in amounts[1:]:
                final_minute = int(reg['minute'])
                stats.loc[initial_minute:final_minute, col] = 0
                initial_minute = final_minute
                value = int(reg['amount']) - last_value
                last_value = value

        stats.loc[initial_minute:95, col] = value

    def stat_proportion(row, stat):

        if row[stat + "_home"] == row[stat + "_away"]:
            return 0.5
        else:
            return row[stat + "_home"] / (row[stat + "_away"] + row[stat + "_home"])

    matches = session.get_collection('matches').find({'status': 1, 'ts': {'$exists': False}})

    requests = []
    for idx, m in enumerate(matches):

        print(idx)
        stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks', 'possession']
        columns = ['minute']
        for s in stats_list:
            columns.append(s + '_home')
            columns.append(s + '_away')

        stats = pd.DataFrame(columns=columns)
        stats.set_index('minute', inplace=True)
        stats['minute'] = range(0, 96)

        fixture = session.get_collection('fixtures').find_one({'id': m['fixture_id']})
        trends = fixture['trends']['data']

        for trend in trends:

            if trend['team_id'] == m['team_home_id']:
                fill_dataframe(trend, 'home')
            else:
                fill_dataframe(trend, 'away')

        keys = stats.keys()
        for k in keys:

            if stats[k].isnull().values.any():
                stats[k] = 0

        for stat in stats_list:
            col = stat + '_dif'
            stats[col] = stats[stat + "_home"] - stats[stat + "_away"]

            col = stat + '_div'
            stats[col] = round(stats.apply(lambda row: stat_proportion(row, stat), axis=1), 3)

        json_stats = stats.to_json()
        dict_stats = json.loads(json_stats)

        ts = {}
        ts['stream'] = dict_stats

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': ts}))
        if idx >= 500 and idx % 500 == 0:
            session.get_collection('matches').bulk_write(requests)
            requests = []

    session.get_collection('matches').bulk_write(requests)





def matches_status_2():
    # PART 2: Status = 2, if the match has minutes out of the range 90 and 95

    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = []

    for m in matches:
        minute = m['minute_max']

        stat = {}
        if minute < 90 or minute > 95:
            stat['status'] = 2
            # session.get_collection('matches').update_one({'_id':m['_id']}, {'$set': stat})
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': stat}))

    session.get_collection('matches').bulk_write(requests)

# Status = 4, if the trends goals is different of score
def matches_status4():
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = []

    for idx, m in enumerate(matches):

        stat = {}
        dif = False
        if (m['ts']['goals_home']['95'] != m['score_home']) \
                or (m['ts']['goals_away']['95'] != m['score_away']):
            dif = True

        if dif:
            print(idx)
            stat['status'] = 4
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': stat}))

    session.get_collection('matches').bulk_write(requests)


def extract_goals_old():

    def get_minute(t):
        minute = t['minute']
        if minute == 90:
            extra_minute = t['extra_minute']
            if extra_minute is not None:
                minute = minute + extra_minute

        if (minute == 0):
            print("TEMOS")
        return minute-1

    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = []

    for idx, m in enumerate(matches):

        update = dict()
        print(idx, m['fixture_id'])
        stats_list = ['goals']
        columns = ['minute']

        for s in stats_list:
            columns.append(s + '_home')
            columns.append(s + '_away')

        minutes = m['minutes']

        events = pd.DataFrame(columns=columns)
        events['minute'] = range(0, minutes + 1)
        events.set_index('minute', inplace=True)

        counters = pd.DataFrame(columns=columns)
        counters['minute'] = range(0, minutes + 1)
        counters.set_index('minute', inplace=True)

        keys = events.keys()
        for k in keys:
            events[k] = 0
            counters[k] = 0

        fixture = session.get_collection('fixtures').find_one({'id': m['fixture_id']})
        trends = fixture['goals']['data']

        goals = dict()
        for k in keys:
            goals[k] = list()

        try:
            for t in trends:

                home_ids = [m['team_home_id'], m['team_home_leg_id']]
                away_ids = [m['team_away_id'], m['team_away_leg_id']]

                minute = get_minute(t)
                if int(t['team_id']) in home_ids:
                    events.loc[minute, 'goals_home'] += 1
                elif int(t['team_id']) in away_ids:
                    events.loc[minute, 'goals_away'] += 1
                else:
                    raise Exception

            # counters.loc[0] = events.loc[0]
            # for j in range(1, len(events)):
            #     counters.loc[j] = counters.loc[j-1] + events.loc[j]
            #
            # if counters.loc[minutes, 'goals_home'] != m['score_home']:
            #     raise StreamIsNotEqualsToScore("Stream of goals is not equals of scoreboard")
            # elif counters.loc[minutes, 'goals_away'] != m['score_away']:
            #     raise StreamIsNotEqualsToScore("Stream of goals is not equals of scoreboard")

        except Exception as ex:
            update['status_error'] = 2
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))
        except StreamIsNotEqualsToScoreException as ex:
            update['status_error'] = "Stream of goals is not equals of scoreboard"
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))
        except KeyError as ek:
            update['status_error'] = 3
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))
            raise ek


def extract_event_goals():

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):
        # print(idx, m['id'])

        stats_list = ['goals']
        columns = get_columns(stats_list)

        minutes = max(m['minutes'], m['minute_max'])

        events = create_events_dataframe(columns, minutes)

        update = dict()

        fixture = session.get_collection('fixtures').find_one({'id': m['id']})
        stream = fixture['goals']['data']

        home_ids = [m['team_home_id'], m['team_home_leg_id']]
        away_ids = [m['team_away_id'], m['team_away_leg_id']]

        try:
            for s in stream:

                try:
                    minute = get_minute_of_event(s)
                except TypeError:
                    # if a event doesn't have value in field minute, then it did not happened.
                    continue

                if minute > minutes:
                    events = expand_dataframe(events, minute, minutes)
                    update['minute_max'] = minute
                    minutes = minute

                if int(s['team_id']) in home_ids:
                    events.loc[minute, 'goals_home'] += 1
                elif int(s['team_id']) in away_ids:
                    events.loc[minute, 'goals_away'] += 1
                else:
                    raise TeamIdentifierException

        except (TeamIdentifierException, TimeEventGreaterTimeMatchException) as ex:
            update['status_error'] = ex.code
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        events_json = events.to_json()
        events_dict = json.loads(events_json)
        update['goals'] = events_dict

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if submit(idx):
            print(idx, m['id'])
            session.get_collection('matches').bulk_write(requests)
            requests = []

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end-start)

def extract_event_cards():

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):
        # print(idx, m['id'])

        stats_list = ['yellow_cards', 'red_cards']
        columns = get_columns(stats_list)

        minutes = max(m['minutes'], m['minute_max'])

        events = create_events_dataframe(columns, minutes)

        update = dict()

        fixture = session.get_collection('fixtures').find_one({'id': m['id']})
        stream = fixture['cards']['data']

        home_ids = [m['team_home_id'], m['team_home_leg_id']]
        away_ids = [m['team_away_id'], m['team_away_leg_id']]

        try:
            for s in stream:

                try:
                    minute = get_minute_of_event(s)
                except TypeError:
                    # if a event doesn't have value in field minute, then it did not happened.
                    continue

                if minute > minutes:
                    events = expand_dataframe(events, minute, minutes)
                    update['minute_max'] = minute
                    minutes = minute

                if int(s['team_id']) in home_ids:
                    add_card('home', s, minute, events)
                elif int(s['team_id']) in away_ids:
                    add_card('away', s, minute, events)
                else:
                    raise TeamIdentifierException

        except (TeamIdentifierException, TimeEventGreaterTimeMatchException) as ex:
            update['status_error'] = ex.code
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        events_json = events.to_json()
        events_dict = json.loads(events_json)
        update['cards'] = events_dict

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if submit(idx):
            print(idx, m['id'])
            session.get_collection('matches').bulk_write(requests)
            requests = list()

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end-start)


def extract_event_trends():

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):
        print(idx, m['id'])

        stats_list = ['goals', 'corners', 'on_target', 'off_target', 'attacks', 'dangerous_attacks']
        columns = get_columns(stats_list)

        minutes = max(m['minutes'], m['minute_max'])

        events = create_events_dataframe(columns, minutes)

        update = dict()

        fixture = session.get_collection('fixtures').find_one({'id': m['id']})
        stream = fixture['trends']['data']

        home_ids = [m['team_home_id'], m['team_home_leg_id']]
        away_ids = [m['team_away_id'], m['team_away_leg_id']]

        try:
            for s in stream:

                try:
                    team_id = int(s['team_id'])
                except Exception:
                    raise TeamIdentifierException

                if team_id in home_ids:
                    minutes = add_trend('home', s, events, update, minutes)
                elif team_id in away_ids:
                    minutes = add_trend('away', s, events, update, minutes)
                else:
                    raise TeamIdentifierException

        except (TeamIdentifierException, TimeEventGreaterTimeMatchException) as ex:
            update['status_error'] = ex.code
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        events_json = events.to_json()
        events_dict = json.loads(events_json)
        update['trends'] = events_dict

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if submit(idx):
            print(idx, m['id'])
            session.get_collection('matches').bulk_write(requests)
            requests = list()

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end-start)


def extract_ball_posession():

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):
        print(idx, m['id'])

        stats_list = ['possession']
        columns = get_columns(stats_list)

        minutes = max(m['minutes'], m['minute_max'])

        events = create_events_dataframe(columns, minutes)

        update = dict()

        fixture = session.get_collection('fixtures').find_one({'id': m['id']})
        stream = fixture['trends']['data']

        home_ids = [m['team_home_id'], m['team_home_leg_id']]
        away_ids = [m['team_away_id'], m['team_away_leg_id']]

        try:
            for s in stream:

                try:
                    team_id = int(s['team_id'])
                except Exception:
                    raise TeamIdentifierException

                if team_id in home_ids:
                    minutes = add_possession('home', s, events, update, minutes)
                elif team_id in away_ids:
                    minutes = add_possession('away', s, events, update, minutes)
                else:
                    raise TeamIdentifierException

        except (TeamIdentifierException, TimeEventGreaterTimeMatchException) as ex:
            update['status_error'] = ex.code
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        events_json = events.to_json()
        events_dict = json.loads(events_json)
        update['possession'] = events_dict

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if submit(idx):
            print(idx, m['id'])
            session.get_collection('matches').bulk_write(requests)
            requests = list()

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end-start)


def add_trend(local, trend, events, update, minutes):

    stream = trend['analyses']
    col = trend['type'] + "_" + local
    last_value = 0

    if col in events.columns:

        for s in stream:

            minute = int(s['minute'])

            if minute > minutes:
                events = expand_dataframe(events, minute, minutes, tp=2)
                update['minute_max'] = minute
                minutes = minute

            amount = int(s['amount'])
            value = amount - last_value
            events.loc[minute, col] = value
            last_value = amount

    return minutes


def add_possession(local, trend, events, update, minutes):

    stream = trend['analyses']
    col = trend['type'] + "_" + local

    inital_minute = 0

    if col in events.columns:

        for s in stream:

            minute = int(s['minute'])

            if minute > minutes:
                events = expand_dataframe(events, minute, minutes, tp=2)
                update['minute_max'] = minute
                minutes = minute

            amount = int(s['amount'])
            events.loc[inital_minute:minute+1, col] = amount
            inital_minute = minute+1

        events.loc[inital_minute:minutes+1, col] = amount
    return minutes


def counter_events(event):

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):
        print(idx, m['id'])

        update = dict()

        stream = m[event]
        df = pd.DataFrame.from_dict(stream)
        df.index = df.index.map(int)
        df = df.sort_index()
        for j in range(1, df.index[-1]+1):
            df.loc[j] = df.loc[j-1] + df.loc[j]

        counter_json = df.to_json()
        counter_dict = json.loads(counter_json)
        col = 'counter_' + event
        update[col] = counter_dict

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if submit(idx):
            print(idx, m['id'])
            session.get_collection('matches').bulk_write(requests)
            requests = list()

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end - start)


def counter_trends():

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):
        print(idx, m['id'])

        update = dict()

        stream = m['trends']

        df = pd.DataFrame.from_dict(stream)
        df.index = df.index.map(int)
        df = df.sort_index()
        for j in range(1, df.index[-1]+1):
            df.loc[j] = df.loc[j-1] + df.loc[j]

        counter_json = df.to_json()
        counter_dict = json.loads(counter_json)
        col = 'counter_trends'
        update[col] = counter_dict

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if submit(idx):
            # print(idx, m['id'])
            session.get_collection('matches').bulk_write(requests)
            requests = list()

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end - start)


def add_card(local, event, minute, events):

    if event['type'] == 'yellowcard':
        col = 'yellow_cards_' + local
    else:
        col = 'red_cards_' + local

    events.loc[minute, col] += 1


def create_events_dataframe(columns, minutes):

    df = pd.DataFrame(columns=columns)
    df['minute'] = range(0, minutes + 1)
    df.set_index('minute', inplace=True)
    df.fillna(0, inplace=True)
    return df


def get_columns(stats_list):
    columns = ['minute']
    for s in stats_list:
        columns.append(s + '_home')
        columns.append(s + '_away')
    return columns


def expand_dataframe(df, minute, minutes, tp=1):

    complement = pd.DataFrame(columns=df.columns)
    if tp == 1:
        complement['minute'] = range(minutes + 1, minute + 1)
    else:
        print("dif", df.index[-1], minutes)
        complement['minute'] = range(minutes, minute + 1)
    complement.set_index('minute', inplace=True)
    complement.fillna(0, inplace=True)
    df = df.append(complement)

    return df


def get_minute_of_event(event):

    minute = event['minute']
    if minute == 90:
        extra_minute = event['extra_minute']
        if extra_minute is not None:
            minute = minute + extra_minute

    return minute-1

# def a():
#     a = pd.DataFrame()
#     a["casa"] = 96 * [0]
#     a.loc[0:0,"casa"] = 20
#
#     print(a)


def assert_score_and_trends():

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):

        print(idx, m['id'])

        try:

            goals_home = m['score_home']
            goals_away = m['score_away']

            df_trends = pd.DataFrame.from_dict(m['counter_trends'])

            goals_home_trends = df_trends['goals_home'][-1]
            goals_away_trends = df_trends['goals_away'][-1]

            if goals_home_trends != goals_home or goals_away_trends != goals_away:
                raise StreamIsNotEqualsToScoreException

        except StreamIsNotEqualsToScoreException as ex:
            update = dict()
            update['status_error'] = ex.code
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end - start)


def assert_events():

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):

        print(idx, m['id'])

        try:

            df_trends = pd.DataFrame.from_dict(m['trends'])
            df_trends.index = df_trends.index.map(int)

            if (df_trends.index < 0).any():
                raise MinuteNegativeException

            if df_trends.isnull().values.any():
                raise AmountNullEventException

            # if (df_trends.values < 0).any():
            #     raise AmountNegativeEventException

        except (MinuteNegativeException, AmountNullEventException, AmountNegativeEventException) as ex:
            print("BUM")
            update = dict()
            update['status_error'] = ex.code
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end - start)


def compare_home_away(label, stats_list, tp='dif'):

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):
        print(idx, m['id'])

        update = dict()

        stream = m[label]

        df = pd.DataFrame.from_dict(stream)
        df.index = df.index.map(int)
        df = df.sort_index()

        if tp == 'dif':
            for stat in stats_list:
                df[stat] = df[stat + "_home"] - df[stat + "_away"]
        else:
            for stat in stats_list:
                df[stat] = round(df.apply(lambda row: stat_proportion(row, stat), axis=1), 3)

        df_json = df[stats_list].to_json()
        df_dict = json.loads(df_json)
        col = tp + '_' + label
        update[col] = df_dict

        requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

        if submit(idx):
            # print(idx, m['id'])
            session.get_collection('matches').bulk_write(requests)
            requests = list()

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end - start)


def selecting_matches():

    start = datetime.now()
    matches = session.get_collection('matches').find({'status_error': {'$exists': False}})
    requests = list()

    for idx, m in enumerate(matches):

        print(idx, m['id'])

        minute_max = m['minute_max']

        if 90 <= minute_max <= 95:

            update = dict()
            update['selected'] = 1

            requests.append(UpdateOne({'_id': m['_id']}, {'$set': update}))

    session.get_collection('matches').bulk_write(requests)
    end = datetime.now()

    print('FINISH: ', end - start)

def stat_proportion(row, stat):

    if row[stat + "_home"] == row[stat + "_away"]:
        return 0.5
    else:
        return row[stat + "_home"] / (row[stat + "_away"] + row[stat + "_home"])



def extract_odds():

    def get_summary_by_column(df, col):

        max_odd = round(df[col].max(), 5)
        min_odd = round(df[col].min(), 5)
        avg_odd = round(df[col].mean(), 5)
        std_odd = round(df[col].std(), 5)

        return max_odd, min_odd, avg_odd, std_odd

    matches = session.get_collection('matches').find({'selected':1}).skip(50000)
    requests = []

    for idx, m in enumerate(matches):

        print(idx, m)
        fixture = session.get_collection('fixtures').find_one({'id': m['id'], 'odds.data.id': 1},
                                                              {'odds.data.$': 1})
        try:
            data1 = fixture['odds']['data']

            frame = list()
            for d1 in data1:
                data2 = d1['bookmaker']['data']
                for d2 in data2:
                    odds = d2['odds']['data']
                    for o in odds:
                        frame.append(o)

            df = pd.DataFrame.from_dict(json_normalize(frame), orient='columns')
            df['value'] = pd.to_numeric(df['value'])

            df['prob'] = 1 / df['value']

            home = df[df['label'] == '1']
            draw = df[df['label'] == 'X']
            away = df[df['label'] == '2']

            up = dict()
            up['n_books'] = len(home)

            t = 'o'
            r = 'home'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                home,
                'value')
            r = 'draw'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                draw,
                'value')
            r = 'away'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                away,
                'value')

            t = 'p'
            r = 'home'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                home,
                'prob')
            r = 'draw'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                draw,
                'prob')
            r = 'away'
            up[t + "_max_" + r], up[t + "_min_" + r], up[t + "_avg_" + r], up[t + "_std_" + r] = get_summary_by_column(
                away,
                'prob')
            raw = [up['p_avg_home'], up['p_avg_draw'], up['p_avg_away']]
            pred_final = [float(i) / sum(raw) for i in raw]

            up['p_norm_home'] = round(pred_final[0], 5)
            up['p_norm_draw'] = round(pred_final[1], 5)
            up['p_norm_away'] = round(pred_final[2], 5)

            fidx = np.argmax(pred_final)

            if fidx == 0:
                fav = 'home'
            elif fidx == 1:
                fav = 'draw'
            else:
                fav = 'away'

            up['o_max_fav'] = up['o_max_' + fav]
            up['o_avg_fav'] = up['o_avg_' + fav]
            up['p_max_fav'] = up['p_max_' + fav]
            up['p_avg_fav'] = up['p_avg_' + fav]

            odds = dict()
            odds['odds'] = up
            requests.append(UpdateOne({'_id': m['_id']}, {'$set': odds}))

            if submit(idx):
                session.get_collection('matches').bulk_write(requests)
                requests = []

        except:
            print('Não tem odds')

    session.get_collection('matches').bulk_write(requests)

extract_odds()

stats_list = ['yellow_cards', 'red_cards']
label = 'counter_cards'

#compare_home_away(label, stats_list, tp='div')

#assert_events()
#selecting_matches()
#assert_score_and_trends()
# extract_ball_posession()
# counter_events('goals')
# counter_events('cards')
# matches_status_error_1()
#extract_event_trends()
#extract_matches()
#extract_event_goals()
#extract_event_cards()