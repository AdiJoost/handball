import pandas as pd
from bs4 import BeautifulSoup

with open("playersHTML.txt", "r") as file:
    soup = BeautifulSoup(file)

with open("playersHTML.txt", "r") as file:
    pds = pd.read_html(file)

print(pds)



