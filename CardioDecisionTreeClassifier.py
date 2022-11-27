import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


def main():
    df = loadCardioData()
    df = removeUnrealistic(df)
    #getInfo(df)
    X_train, X_test, y_train, y_test  = trainTestSplit(df)


    params = {'max_depth': [8,9,10,11,12,13,14,15,18,20,25,30,35,40,42,45,50,55,60], 'max_leaf_nodes': list(range(2, 100))}
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
    grid_search_cv.fit(X_train,y_train)
    print("BestEstimator: ", grid_search_cv.best_estimator_)
    print("BestParams: ", grid_search_cv.best_params_)
    print("BestScore: ",grid_search_cv.best_score_)

    y_pred = grid_search_cv.predict(X_test)
    evaluateClassifier(y_pred, y_test)




def evaluateClassifier(pred, act):

    print("Accuracy", accuracy_score(act, pred))
    #Null accuracy: accuracy that could be achieved by always predicting the most frequent class
    print("Null accuracy",max(act.mean(), 1 - act.mean()))
    print("ConfusionMatrix: ", confusion_matrix(act, pred))


def loadCardioData():
    return pd.read_csv("Data/cardio_train.csv", delimiter=";")


def trainTestSplit(df):
    from sklearn.model_selection import train_test_split
    X_data = df.drop("cardio", axis=1)
    y_data = df["cardio"]
    return train_test_split(X_data, y_data, test_size=0.2, random_state=42)

def getInfo(df):
    print("RowCount: ", len(df.columns))
    df.describe()
    df.info()
    #sns.heatmap(df.corr(), cmap="Blues", annot=True, figsize=(15,15))
    pd.plotting.scatter_matrix(df, figsize=(20,15))
    pd.plotting.hist_frame(df, figsize=(15,15))
    plt.show()

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


if __name__ == "__main__":
    main()
