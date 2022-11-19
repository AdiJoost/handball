import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from random import randint
import json
from datetime import datetime

#Hyper-parameters
optimizers_hyper = ["sgd", "adam", "RMSprop"]
loss_hyper=["binary_crossentropy"]
activations_hyper=["relu", "sigmoid", "softmax", "tanh"]




def main():
    df = load_data()
    #print(f"Any null: {df.isnull().values.any()}")
    Y, X = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    model = get_model(layers=5, neurons=30, activations="tanh", loss="binary_crossentropy")
    
    for optimi in optimizers_hyper:
        for activat in activations_hyper:
            for i in range (2, 5):
                for j in range(8, 10):
                    model = get_model(layers= i, activations=activat, neurons=j, optimizer=optimi)
                    mod, predic = train_model(model, X_train, X_test, y_train, y_test, epochs=2)
                    save(mod, predic, layers=i, activation=activat, neurons=j, optimizer=optimi)

def train_model(model, X_train, X_test, y_train, y_test, epochs=10, batch_size=10):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return (model, model.evaluate(X_test, y_test))

def save(model, prediction_values, layers, activation, neurons, optimizer):
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    model_name = f"{activation}-{ts}"
    model.save(f"trained_models/{model_name}")
    data = json.dumps({
        "model": model_name,
        "predictions": prediction_values,
        "layers": layers,
        "activation": activation,
        "neurons": neurons,
        "optimizer": optimizer
    })
    with open("trained_models/meta_data.json", "a") as file:
        file.write(data)

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
        neurons=20, 
        activations="tanh",
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]):
    model = Sequential()
    model.add(Dense(neurons, activation=activations, input_shape=(15,)))

    for i in range(1, layers -1):
        model.add(Dense(neurons, activation=activations))
    model.add(Dense(1, activation=activations))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def hyper_param_setter(layers, min_neuro_range=5, max_neuro_range=20, possible_activations=activations_hyper):
    neurons = []
    activations = []
    for i in range(layers):
        number = randint(min_neuro_range, max_neuro_range)
        neurons.append(number)
        rand_activation = randint(0, len(possible_activations) - 1)
        activations.append(possible_activations[rand_activation])
    neurons[-1] = 1
    return tuple(neurons), tuple(activations)

if __name__ == "__main__":
    main()