import pandas as pd




def main():
    load_data()


def load_data():
    return pd.read_csv("Data/cardio_train.csv", sep=";", usecols=["age", "weight", "cardio"])


if __name__ == "__main__":
    main()