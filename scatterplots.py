import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Data/cardio_train.csv", sep=";")

print(df.head())

plt.scatter(df["age"] / 365 , df["weight"], c=df["cardio"], marker=",", linewidths=0.4, edgecolors=None)
plt.legend()

plt.show()