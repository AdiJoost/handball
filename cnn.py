import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split




def main():
    df = load_data()
    #print(f"Any null: {df.isnull().values.any()}")
    Y, X = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    model = get_model()
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=10
    )
    print(model.evaluate(X_test, y_test))



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

def print_array_info(arry, name="name"):
    print(f"Array {name}:")
    print(f"ndim: {arry.ndim}")
    print(f"shape: {arry.shape}")
    print(f"size: {arry.shape}")
    print("___________________________________")

def get_model(
        layers=3,
        neurons=(20,20,1), 
        activations=("sigmoid", "sigmoid", "sigmoid"),
        optimizer="sgd",
        loss="binary_crossentropy",
        metrics=["accuracy"]):
    model = Sequential()
    model.add(Dense(neurons[0], activation=activations[0], input_shape=(15,)))

    for i in range(1, layers):
        model.add(Dense(neurons[i], activation=activations[i]))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

if __name__ == "__main__":
    main()