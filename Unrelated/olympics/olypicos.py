import pandas as pd

df = pd.read_csv("olympics/archive/athlete_events.csv")

sports = df["Sport"].unique()

handball = df[df["Sport"] == "Handball"]

print(len(handball))
print(handball.head())
