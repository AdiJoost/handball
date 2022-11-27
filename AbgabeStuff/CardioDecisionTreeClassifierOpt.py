import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import export_text


def main():

    df = loadCardioData()
    df = removeUnrealistic(df)
    X_train, X_test, y_train, y_test = trainTestSplit(df)

    dtc = DecisionTreeClassifier(random_state=42, max_depth=8, max_leaf_nodes=26)
    dtc.fit(X_train, y_train)
    #73.56

    y_pred = dtc.predict(X_test)
    print("act ", y_test.head(20))
    print("pred", y_pred[:20])
    print(dtc.get_params())
    print(accuracy_score(y_test, y_pred))
    evaluateClassifier(y_pred, y_test)

    # plt.figure(figsize=(20,20), facecolor ='k')
    #
    # a = tree.plot_tree(dtc,
    #
    #                    feature_names = list(X_test.columns),
    #
    #                    class_names = "10",
    #
    #                    rounded = True,
    #
    #                    filled = True,
    #
    #                    fontsize=8)
    #
    # plt.show()


    tree_rules = export_text(dtc,
                            feature_names = list(X_test.columns))
    print(tree_rules)

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
