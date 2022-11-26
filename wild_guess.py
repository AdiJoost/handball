"""
This model predicts cardiological problems statistically from
weight and heigth. A reference-function is set. If the Point(age | weight)
is above the function, the patient is predicted to have a cardiac-diseas

The best found parameters for the referance-function are 
m = -0.08
q = 90
"""

import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = load_data()
    show_plot(df)
    correct, total = evaluate(df, -(100/9125), 100)
    #print(f"From {total}-predictions are {correct} cases correct predicted, resulting in"\
    #    f" accuracy of {correct / total}")
    best = train_me(df, m_power_10=1000)
    print(f"Best accuracy: {best[2]}")
    print(f"m: {best[0]}")
    print(f"q: {best[1]}")

def load_data():
    df = pd.read_csv("Data/cardio_train.csv", sep=";", usecols=["age", "weight", "cardio"])
    df["age"] = df["age"] - 14600
    df["weight"] = df["weight"] - 25
    return df

def show_plot(df):
    plt.scatter(df["age"] , df["weight"], c=df["cardio"], marker=",", linewidths=0.4, edgecolors=None)
    plt.axline((0, 90), (11250,0))
    plt.show()

def predict (age, weight, m, q):
    #calculates, whether point (age|weight) is above the predicted line
    # returns 1 if above, 0 if on or below line
    mwfa = m * age + q
    if mwfa < weight:
        return 1
    return 0

def evaluate(df, m, q):
    correct_predricted = 0
    total_predicted = 0
    for i in range(len(df)):
        if (df["cardio"][i] == predict(df["age"][i], df["weight"][i], m, q)):
            correct_predricted += 1
        total_predicted += 1
    if total_predicted != len(df):
        raise ValueError
    return (correct_predricted, total_predicted)

def train_me(df, m_range=(-20, 0), m_steps=1, m_power_10 = 100, q_range=(90,110), q_steps=2, chatty=True):
    best_prediction = (0,0,-1)
    for m in range(m_range[0], m_range[1], m_steps):
        for q in range(q_range[0], q_range[1], q_steps): 
            correct, total = evaluate(df, m/m_power_10, q)
            quote = correct / total
            if best_prediction[2] < quote:
                best_prediction=(m, q, quote)
            if chatty:
                print(f"m:{m}, q:{q}, quote:{quote}")
    return best_prediction
            


if __name__ == "__main__":
    main()