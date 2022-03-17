import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

from model import ModelBase, ModelRegulizedL2, ModelRegulizedL1, ThresholdBasedModel, Model2

cal_housing = fetch_california_housing() 
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names) 
y = cal_housing.target 
df = pd.DataFrame( 
        dict(MedInc=X['MedInc'], Price=cal_housing.target))
X, y = df["MedInc"].to_numpy(), df["Price"].to_numpy()   


myModel3 = Model2()
myModel3.fit(X,y)