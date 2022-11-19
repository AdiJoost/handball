import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    df = importData()
    getInfo(df)


def importData():
    df = pd.read_csv("Data/cardio_train.csv", delimiter=";")
    return df.head(2000)

def getInfo(df):
    df.describe()
    df.info()
    pd.plotting.scatter_matrix(df)
    #pd.plotting.hist_frame(df)
    plt.show()
    print(len(df.columns))


if __name__ == "__main__":
    main()