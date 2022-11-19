import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D




def main():
    df = load_data()
    print(f"Any null: {df.isnull().values.any()}")
    Y, X = prepare_data(df)
    print_array_info(Y)
    print_array_info(X)


def load_data():
    return pd.read_csv("Data/cardio_train.csv", sep=";")

def prepare_data(df):
    #drop id
    df = df.drop("id", axis=1)

    #extract label
    Y = df["cardio"]
    df = df.drop("cardio", axis=1)

    #normalize columns
    min_max_columns = ("age", "height", "weight", "ap_lo", "ap_hi")
    for col in min_max_columns:
        df[col] = min_max(df, col)

    #One_hot encode
    df = get_one_hot(df, "cholesterol", ("col_normal", "col_above", "col_extreme"))
    df = get_one_hot(df, "gluc", ("gluc_normal", "gluc_above", "gluc_extreme"))

    #gender-stuff
    df["gender"] = df["gender"] -1

    return Y.to_numpy(), df.to_numpy()

def min_max(df, column):
    return_df = pd.DataFrame()
    return_df[column] = (df[column]-df[column].min()) / (df[column].max() - df[column].min())
    return return_df

def get_one_hot(df, col, axis_names):
    one_h = pd.get_dummies(df[col])
    one_h = one_h.rename(columns={
        1: axis_names[0],
        2: axis_names[0],
        3: axis_names[0]
    })
    df = df.drop(col, axis=1)
    df = df.join(one_h)
    return df

if __name__ == "__main__":
    main()