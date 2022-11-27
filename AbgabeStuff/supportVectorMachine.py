import pandas as pd
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import itertools

def main():
    df = load_data("cardio_train")
    print(df["cardio"].value_counts())
    Y, X = prepare_data(df)
    model = get_model()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    print(np.unique(y_test, return_counts=True))
    model.fit(X_train, y_train)
    y_test_predi = model.predict(X_test)
    precision, recall, f1, acc, confM = eval(model, X_train, X_test, y_train, y_test, y_test_predi)
    log_csv((precision, recall, f1, acc, f"({confM[0][0], confM[0][1], confM[1][0], confM[1][1]})", "all"))
    #hyper_search(X_train, X_test, y_train, y_test)
    #perms = get_permutations()
    #cols = [False for _ in range(X_train.shape[1])]
    #for i in perms[11]:
    #    cols[i-1] = True
    #print(cols)
    #model = train(X_train, y_train, cols)
    #y_test_predi = model.predict(X_test[:,cols])
    #eval(model, X_train, X_test, y_train, y_test, y_test_predi)


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

def load_data(file):
    my_path = os.getcwd().split("handball", 1)[0]
    my_path = os.path.join(my_path, "handball", "Data", f"{file}.csv")
    df =  pd.read_csv(my_path, sep=";")
    df =  df.drop("id", axis=1)
    return df


def prepare_data(df):
    df = removeUnrealistic(df)
    print(df.head())
    #extract label
    Y = df["cardio"]
    df = df.drop("cardio", axis=1)

    #normalize columns
    min_max_columns = ("age", "height", "weight", "ap_lo", "ap_hi", "cholesterol", "gluc")
    for col in min_max_columns:
        df[col] = min_max(df, col)

    #gender-stuff
    df["gender"] = df["gender"] -1

    return Y.to_numpy(), df.to_numpy()

def removeUnrealistic(df):
    cols = [
        ("height", 150, 210),
        ("weight", 40, 250),
        ("ap_hi", 80,200),
        ("ap_lo", 50, 120),
    ]
    for col, min, max in cols:
        filtermag =  df[col] < max
        df = df[filtermag]
        filtermag =  df[col] > min
        df = df[filtermag]
    return df

    
def min_max(df, column):
    return_df = pd.DataFrame()
    return_df[column] = (df[column]-df[column].min()) / (df[column].max() - df[column].min())
    return return_df

def get_model():
    sgd = SGDClassifier(loss="hinge", max_iter=1000, tol=None, random_state=42)
    return sgd

def train(X_train, y_train, cols):
    filtered_test_set = X_train[:, cols]
    model = get_model()
    model.fit(filtered_test_set, y_train)
    return model
    

def get_permutations():
    stuff = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    perms = []
    for L in range(len(stuff) + 1):
        for subset in itertools.combinations(stuff, L):
            perms.append(subset)
    return perms

def hyper_search(X_train, X_test, y_train, y_test):
    perms = get_permutations()
    for perm in perms:
        if len(perm) < 1:
            continue
        cols = [False for _ in range(X_train.shape[1])]
        for i in perm:
            cols[i-1] = True
        model = train(X_train, y_train, cols)
        y_test_predi = model.predict(X_test[:,cols])
        precision, recall, f1, acc, confM = eval(model, X_train, X_test, y_train, y_test, y_test_predi)
        log_csv((precision, recall, f1, acc, f"({confM[0][0], confM[0][1], confM[1][0], confM[1][1]})", perm))


        

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