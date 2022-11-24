import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier


def main():
    df = loadCardioData()
    # getInfo(df)

    scaler = preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = trainTestSplit(df)

    X_test = scaler.fit_transform(X_test)
    X_train = scaler.fit_transform(X_train)

    clf = SGDClassifier(loss="log", penalty="l2", max_iter=900, tol=1e-3)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    print("Coef: " , clf.coef_)

    print("act ", y_test.head(20))
    print("pred", pred[:20])
    print("Params" , clf.get_params())
    print(accuracy_score(y_test,pred))
    evaluateClassifier(pred,y_test)

def evaluateClassifier(pred, act):
    print("Accuracy", accuracy_score(act, pred))
    # Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    print("Null accuracy", max(act.mean(), 1 - act.mean()))
    print("ConfusionMatrix: ", confusion_matrix(act, pred))
    print()


def loadCardioData():
    return pd.read_csv("Data/cardio_train.csv", delimiter=";")

def trainTestSplit(df):
    from sklearn.model_selection import train_test_split
    X_data = df.drop("cardio", axis=1)
    y_data = df["cardio"]
    return train_test_split(X_data, y_data, test_size=0.2, random_state=42)


if __name__ == "__main__":
    main()