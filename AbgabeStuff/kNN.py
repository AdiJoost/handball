import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing



def main():
    df = loadCardioData()
    # getInfo(df)



    scaler = preprocessing.MinMaxScaler()
    X_train, X_test, y_train, y_test = trainTestSplit(df)

    X_test = scaler.fit_transform(X_test)
    X_train = scaler.fit_transform(X_train)

    kNN = KNeighborsClassifier(n_neighbors=7)
    kNN.fit(X_train, y_train)
    pred = kNN.predict(X_test)
    print("act ", y_test.head(20))
    print("pred", pred[:20])
    print(kNN.get_params())
    print(accuracy_score(y_test,pred))
    evaluateClassifier(pred, y_test)


def evaluateClassifier(pred, act):
    print("Accuracy", accuracy_score(act, pred))
    # Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    print("Null accuracy", max(act.mean(), 1 - act.mean()))
    print("ConfusionMatrix: ", confusion_matrix(act, pred))


def loadCardioData():
    df = pd.read_csv("Data/cardio_train.csv", delimiter=";")
    df = removeUnrealistic(df)
    minMaxScaler = preprocessing.MinMaxScaler()
    df[["age"]] = minMaxScaler.fit_transform(df[["age"]])
    df[["height"]] = minMaxScaler.fit_transform(df[["height"]])
    df[["weight"]] = minMaxScaler.fit_transform(df[["weight"]])

    print(df.head())
    return df

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


def trainTestSplit(df):
    from sklearn.model_selection import train_test_split
    X_data = df.drop("cardio", axis=1)
    y_data = df["cardio"]
    return train_test_split(X_data, y_data, test_size=0.2, random_state=42)


if __name__ == "__main__":
    main()