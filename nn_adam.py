import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Dropout
from sklearn.model_selection import train_test_split
from random import randint
import json
import os
from datetime import datetime
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


#Hyper-parameters
optimizers_hyper = ["sgd", "adam", "RMSprop"]
loss_hyper=["binary_crossentropy"]
activations_hyper=["relu", "sigmoid", "softmax", "tanh"]




def main():
    df = load_data()
    #print(f"Any null: {df.isnull().values.any()}")
    Y, X = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    #X_train = reshape(X_train)
    #print(X_train.shape)
    #X_test = reshape(X_test)
    model = get_static_model()
    _, preds = train_model(model, X_train, X_test, y_train, y_test)
    y_test_predi = model.predict(X_test)
    precision, recall, f1, acc, confM = eval(model, X_train, X_test, y_train, y_test, y_test_predi)
    log_csv((precision, recall, f1, acc, f"({confM[0][0], confM[0][1], confM[1][0], confM[1][1]})", "all"))

    
def reshape(X_train):
    sample_size = X_train.shape[0]
    time_steps = X_train.shape[1]
    input_dimension = 1
    return_value = X_train.reshape(sample_size, time_steps, input_dimension)
    return return_value

def train_model(model, X_train, X_test, y_train, y_test, epochs=10, batch_size=5):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return (model, model.evaluate(X_test, y_test))

def eval(model, X_train, X_test, y_train, y_test, y_test_predi):
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=3)
    y_test_pred = cross_val_predict(model, X_test, y_test, cv=3)
    #precision_score(y_train, y_train_pred)
    precision = precision_score(y_test, y_test_pred)
    #recall_score(y_train, y_train_pred)
    recall = recall_score(y_test, y_test_pred)
    #f1_score(y_train, y_train_pred)
    f1 = f1_score(y_test, y_test_pred)

    acc = accuracy_score(y_test, y_test_predi)
    # Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    print("Null accuracy", max(y_test.mean(), 1 - y_test.mean()))
    confM = confusion_matrix(y_test, y_test_predi)
    return precision, recall, f1, acc, confM

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
    df = removeUnrealistic(df)
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

def removeUnrealistic(df):
    cols = [
        ("height", 100, 210),
        ("weight", 40, 250),
        ("ap_hi", 80,200),
        ("ap_lo", 50, 120),
        ("gender", 1,1)
    ]
    for col, min, max in cols:
        filtermag =  df[col] <= max
        df = df[filtermag]
        filtermag =  df[col] >= min
        df = df[filtermag]
    return df

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

def get_static_model():
    model = Sequential()
    model.add(Dense(15, activation="relu", input_shape=(15,)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


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

def log_csv(message: tuple, file="SVM_EVAL"):
    entry = get_entry(message)
    #create correct path to file
    my_path = os.getcwd().split("handball", 1)[0]
    my_path = os.path.join(my_path, "handball", "SVM", f"{file}.csv")
    
    #write to file
    if not os.path.exists(my_path):
        csv_header = \
        "Precision;recall;f1;accuracy;confMatrix:perm\n"
        with open(my_path, "w", encoding=("UTF-8")) as f:
            f.write(csv_header)
    with open(my_path, "a", encoding=("UTF-8")) as f:
        f.write(entry)

def get_entry(message: tuple):
    return_value = ""
    for item in message:
        return_value += f"{item};"
    return f"{return_value[:-1]}\n"

if __name__ == "__main__":
    main()