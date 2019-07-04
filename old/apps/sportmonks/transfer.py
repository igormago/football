import requests
from bs4 import BeautifulSoup

import pandas as pd

headers = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

page = "https://www.transfermarkt.pt/tiquinho-soares/leistungsdaten/spieler/103381/plus/0?saison=2017"
pageTree = requests.get(page, headers=headers)
pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

Players = pageSoup.find_all("a", {"class": "spielprofil_tooltip"})

#Let's look at the first name in the Players list.
print(print(pageSoup))