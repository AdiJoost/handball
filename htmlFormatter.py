import pandas as pd

table = pd.read_html("playersHTML.txt")

print(table.head())