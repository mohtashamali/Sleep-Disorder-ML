import numpy as np
import pandas as pd
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def pre_processing():
        df=pd.read_csv("data/raw_data.csv")
        def taking_mean_of_non_nanvalue(series):
            list1 = list(series)
            f_list = [i for i in list1 if pd.notna(i) and i != "" and i is not None]
            return f_list
        labeler=LabelEncoder()
        values=taking_mean_of_non_nanvalue(df["Sleep Disorder"])
        filled_value=mode(values)
        # numerical_labeled_data=labeler.fit_transform(filled_value)
        df["Sleep Disorder"]=df["Sleep Disorder"].fillna(filled_value)
        #------------->
        df["Blood Pressure"] = df["Blood Pressure"].str.strip('/')  
        df[["Systolic", "Diastolic"]] = df["Blood Pressure"].str.split('/', expand=True)

        df["Systolic"] = pd.to_numeric(df["Systolic"], errors='coerce')
        df["Diastolic"] = pd.to_numeric(df["Diastolic"], errors='coerce')
        df["Mean_BP"] = (df["Systolic"] + 2*df["Diastolic"]) / 3 
        # Removing unnesseary columns
        df.drop(["Systolic", "Diastolic","Blood Pressure","Person ID"], axis=1, inplace=True)
        df.to_csv("data/preprocessed_data.csv",index=False)
pre_processing()