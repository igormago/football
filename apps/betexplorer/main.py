from modules.betexplorer2 import services as api

#api.reset_database()
#api.create_championships()
##api.create_seasons(2006, 2018)
#api.request_seasons(replace=True)
#api.create_matches(replace=True)
#api.request_matches()
api.create_odds(replace=True)
api.process_hits(replace=True)
api.create_performances(replace=True)
api.create_markov_chains(replace=True)
api.create_dataframe(replace=True)
