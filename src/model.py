import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
def train_model():
        df = pd.read_csv("data/featured_data.csv")

        x = df.drop(["Sleep Disorder"], axis=1)
        y = df["Sleep Disorder"]

        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        regression = LinearRegression()
        ml_model = regression.fit(x_train, y_train)

        with open("model/model.pkl", "wb") as f:
            pickle.dump(ml_model, f)

        # print("Train score:", ml_model.score(x_train, y_train))
        # print("Test score:", ml_model.score(x_test, y_test))