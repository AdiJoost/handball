import pandas as pd
import os

def main():
    df = load_data("cardio_train")
    Y, X = prepare_data(df)
    model = get_model()


def load_data(file):
    my_path = os.getcwd().split("handball", 1)[0]
    my_path = os.path.join(my_path, "handball", "Data", f"{file}.csv")
    df =  pd.read_csv(my_path, sep=";")
    df =  df.drop("id", axis=1)
    return df


def prepare_data(df):

    #extract label
    Y = df["cardio"]
    df = df.drop("cardio", axis=1)

    #normalize columns
    min_max_columns = ("age", "height", "weight", "ap_lo", "ap_hi", "cholesterol", "gluc")
    for col in min_max_columns:
        df[col] = min_max(df, col)

    #gender-stuff
    df["gender"] = df["gender"] -1
    print(df.head())

    return Y.to_numpy(), df.to_numpy()
    
def min_max(df, column):
    return_df = pd.DataFrame()
    return_df[column] = (df[column]-df[column].min()) / (df[column].max() - df[column].min())
    return return_df

def get_model():
    pass


if __name__ == "__main__":
    main()