from apps.sportmonks import api
from core.databases import SportMonksDatabase as DataBase
from dateutil import rrule
from datetime import datetime, timedelta
import time

session = DataBase.get_session()

token = "putyourtokenhere"
api.init(token)

CONTINENTS_URL = 'continents'
COUNTRIES_URL = 'countries'
LEAGUES_URL = 'leagues'
SEASONS_URL = 'seasons'
ROUNDS_URL = 'rounds'
FIXTURES_URL = 'fixtures'
INPLAY_ODDS_URL = 'odds/inplay/fixture'


def continents():
    r = api.get(CONTINENTS_URL, paginated=False)
    session.get_collection('continents').insert_many(r)


def countries():
    r = api.get(COUNTRIES_URL, paginated=False)
    session.get_collection('countries').insert_many(r)


def leagues():
    r = api.get(LEAGUES_URL, paginated=False)
    session.get_collection('leagues').insert_many(r)


def seasons():
    r = api.get(SEASONS_URL, paginated=False)
    session.get_collection('seasons').insert_many(r)


def rounds():
    include = 'results'
    seasons = session.get_collection('seasons').find({}, {'id': True})

    for s in seasons:

        id_season = s['id']
        url = ROUNDS_URL + '/season/' + str(id_season)

        print(url)
        r = api.get(url, include, paginated=False)

        try:
            session.get_collection('rounds').insert_many(r)
        except Exception:
            pass


def fixtures_by_date(start_date, end_date):
    include = 'localTeam, visitorTeam, substitutions, goals, cards, other, corners, lineup, bench, sidelined, stats,' \
              ' comments, tvstations, highlights, league, season, round, stage, referee, events, venue, odds, ' \
              'flatOdds, inplay, localCoach, visitorCoach, group, trends'

    url = FIXTURES_URL + "/between/" + start_date + '/' + end_date
    # url = FIXTURES_URL +  "/date/" + start_date

    print(url)
    r = api.get(url, include, paginated=False)

    print(len(r))
    session.get_collection('fixtures').insert_many(r)


def fixtures():
    dt_start = datetime(2009, 3, 16)
    dt_end = datetime(2009, 5, 14)

    for dt in rrule.rrule(rrule.WEEKLY, dtstart=dt_start, until=dt_end):
        start = dt
        end = dt_start + timedelta(days=6)

        time.sleep(15)
        fixtures_by_date(str(start.date()), str(end.date()))
