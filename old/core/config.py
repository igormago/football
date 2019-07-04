PATH_USER = '/home/igorcosta/'
PATH_PROJECTS = PATH_USER + 'PycharmProjects/'
PATH_APP = PATH_PROJECTS + "soccer/"
PATH_FILES = PATH_APP + "files/"
PATH_FILES_APPS = PATH_FILES + "apps/"
PATH_LOGS = PATH_FILES + "logs/"
PATH_DATAFRAMES = PATH_FILES + "dataframes/"

PATH_WEB = PATH_FILES + "web/"
PATH_PREMIERLEAGUE = PATH_WEB + 'premierleague/'

PATH_BETEXPLORER_DATAFRAMES = PATH_DATAFRAMES + "betexplorer/"
PATH_BETEXPLORER = PATH_WEB + "betexplorer/"
PATH_BETEXPLORER_CHAMPIONSHIPS = PATH_BETEXPLORER + "championships/"
PATH_BETEXPLORER_MATCHES = PATH_BETEXPLORER + "matches/"

PATH_SPORTMONKS_DATAFRAMES = PATH_DATAFRAMES + "sportmonks/"

PATH_CARTOLA = PATH_WEB + 'cartola/'
PATH_JOBS = PATH_APP + "jobs/"
PATH_JOBS_RESULTS = PATH_JOBS + "results/"
PATH_JOBS_MODELS = PATH_JOBS + "models/"

# PATH_FILES = PATH_APP + "files/"# PATH_LOGS = PATH_FILES + "logs/"
# PATH_DATAFRAMES = PATH_FILES + "dataframes/"
# PATH_PREDICTIONS = PATH_DATAFRAMES + "predictions/"
# PATH_RESULTS = PATH_DATAFRAMES + "results/"
# PATH_PLOTS = PATH_FILES + "plots/"
# PATH_WEB = PATH_FILES + "web/"
#
# PATH_BETEXPLORER = PATH_WEB + "betexplorer/"
# PATH_BETEXPLORER_CHAMPIONSHIPS = PATH_BETEXPLORER + "championships/"
# PATH_BETEXPLORER_MATCHES = PATH_BETEXPLORER + "matches/"
#
# PATH_FILE_STATS_TOTAL = PATH_FILES + 'dataframes/bts_stats.csv'
# PATH_FILE_BTS_TOTAL = PATH_FILES + 'dataframes/bts_total.csv'
# PATH_FILE_BRAZIL = PATH_FILES + 'dataframes/bts_brazil.csv'
#
# PATH_CARTOLA = PATH_WEB + "cartola/"
# PATH_CARTOLA_MARKET = PATH_CARTOLA + 'market/'
# PATH_CARTOLA_CHAMPIONSHIP = PATH_CARTOLA + 'championship/'

DataBase_KAGGLE_EUROPEAN_LOCAL = 'mysql+mysqlconnector://root:root@localhost/betexplorer_local'
DataBase_BETEXPLORER_LOCAL = 'mysql+mysqlconnector://root:root@localhost/betexplorer_local'
DataBase_SOCCER_LOCAL = 'mysql+mysqlconnector://root:root@localhost/soccer_local'
DataBase_OLD_BETFAIR_LOCAL = 'mysql+mysqlconnector://root:root@localhost/old_betfair_local'
DataBase_CENTRAL_LOCAL = 'mysql+mysqlconnector://root:root@localhost/central_local'

DataBase_BETEXPLORER_DEV = 'mysql+mysqlconnector://dev:s0cc3r@150.165.85.3/betexplorer_dev'
DataBase_SOCCER_DEV = 'mysql+mysqlconnector://dev:dev@150.165.85.3/soccer_dev'

DataBase_BETEXPLORER_PROD = 'mysql+mysqlconnector://dev:dev@localhost/betexplorer'
DataBase_SOCCER_PROD = 'mysql+mysqlconnector://root:root@localhost/total_soccer'
DataBase_CENTRAL_PROD = 'mysql+mysqlconnector://root:root@localhost/central_prod'


DataBase_KAGGLE_EUROPEAN = DataBase_KAGGLE_EUROPEAN_LOCAL
DataBase_BETEXPLORER = DataBase_BETEXPLORER_DEV
DataBase_SOCCER = DataBase_SOCCER_DEV
DataBase_OLD_BETFAIR = DataBase_OLD_BETFAIR_LOCAL
DataBase_CENTRAL = DataBase_CENTRAL_LOCAL

CRAWLER_INITIAL_YEAR_CHAMPIONSHIP = 2006
CRAWLER_LAST_YEAR_CHAMPIONSHIP = 2016

PATH_FOOTBALLDATA = PATH_WEB + 'footballdata/'
PATH_FOOTBALLDATA_CHAMPIONSHIPS = PATH_FOOTBALLDATA + 'championships/'
PATH_FOOTBALLDATA_CSV = PATH_FOOTBALLDATA + 'csv/'

PATH_OLD_BETFAIR = PATH_WEB + 'old_betfair/'

APP_BETFAIR_HD='betfairhd'
APP_FOOTBALLDATA='footballdata'

