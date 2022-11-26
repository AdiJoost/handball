import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


def main():
    df = loadCardioData()
    # getInfo(df)
    scaler = preprocessing.StandardScaler()
    X_train, X_test, y_train, y_test = trainTestSplit(df)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)



    # params = {'max_depth': []}
    # grid_search_cv = GridSearchCV(LogisticRegression(random_state=42), params, verbose=1, cv=3)

    # grid_search_cv.fit(X_train, y_train)
    # print("BestEstimator: ", grid_search_cv.best_estimator_)
    # print("BestParams: ", grid_search_cv.best_params_)
    # print("BestScore: ", grid_search_cv.best_score_)
    #
    # y_pred = grid_search_cv.predict(X_test)
    # evaluateClassifier(y_pred, y_test)
    # print("Act:  ",y_test)
    # print("Pred: ", y_pred)


    logReg =  LogisticRegression()
    logReg.fit(X_train, y_train)
    pred = logReg.predict(X_test)
    print("act ", y_test.head(20))
    print("pred", pred[:20])
    print(logReg.get_params())
    print(accuracy_score(y_test,pred))
    evaluateClassifier(pred,y_test)
def evaluateClassifier(pred, act):
    print("Accuracy", accuracy_score(act, pred))
    # Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    print("Null accuracy", max(act.mean(), 1 - act.mean()))
    print("ConfusionMatrix: ", confusion_matrix(act, pred))


def loadCardioData():


    return pd.read_csv("Data/cardio_train.csv", delimiter=";")


def trainTestSplit(df):
    from sklearn.model_selection import train_test_split
    X_data = df.drop("cardio", axis=1)
    y_data = df["cardio"]
    return train_test_split(X_data, y_data, test_size=0.2, random_state=42)


if __name__ == "__main__":
    main()