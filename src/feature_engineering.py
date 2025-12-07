import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def feature_engineering():
    df = pd.read_csv("data/preprocessed_data.csv")
    
    
    
    # Label encoding categorical columns
    labeler = LabelEncoder()
    df["Gender"] = labeler.fit_transform(df["Gender"])
    df["Occupation"] = labeler.fit_transform(df["Occupation"])
    df["BMI Category"] = labeler.fit_transform(df["BMI Category"])
    df["Sleep Disorder"] = labeler.fit_transform(df["Sleep Disorder"])
    

    scaler = StandardScaler()
    df["Mean_BP"] = scaler.fit_transform(df[["Mean_BP"]])

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    df.to_csv("data/featured_data.csv", index=False)


feature_engineering()