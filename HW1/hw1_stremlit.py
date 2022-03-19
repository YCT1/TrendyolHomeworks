from distutils.log import error
from unittest import result
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.datasets import fetch_california_housing

from model import ModelBase, ModelRegulizedL2, ModelRegulizedL1, ThresholdBasedModel, TorchBasedModel


def CalculateError(x, y,threshold, b=0.2):
    error1 = (x-y)**2

    error2 = threshold * (1/-np.exp(b*error1) + 1)
    return error1.mean(), error2.mean()



def main():
    st.header("Homework 1")
    st.text("By Yekta Can Tursun")


    cal_housing = fetch_california_housing() 
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names) 
    y = cal_housing.target 
    df = pd.DataFrame( 
            dict(MedInc=X['MedInc'], Price=cal_housing.target))
    X, y = df["MedInc"].to_numpy(), df["Price"].to_numpy()   

    myModel = ModelBase()
    myModel2 = ModelRegulizedL2()

    threshold = st.slider("Threshold",1.,25.,3.5)
    myModel3 = TorchBasedModel(threshold)
    
    myModel4 = ThresholdBasedModel(threshold)

    epoch = st.slider("Epoch",10,1000,100)
    
    myModel.fit(X,y,epoch=epoch)
    myModel2.fit(X,y,epoch=epoch)
    myModel3.fit(X,y,epoch=epoch)
    myModel4.fit(X,y,epoch=epoch)

    y_pred = myModel.pred(X)
    y_pred2 = myModel2.pred(X)
    y_pred3 = myModel3.pred(X)
    y_pred4 = myModel4.pred(X)

    error = CalculateError(y_pred,y,threshold)
    error2 = CalculateError(y_pred2,y,threshold)
    error3 = CalculateError(y_pred3,y,threshold)
    error4 = CalculateError(y_pred4,y,threshold)

    st.write(f"(MAE) Base Model Error (MAE):{error[0]:.3f} ;  (MAE-Capped): {error[1]:.3f}")
    st.write(f"(MAE) L2 Model Error (MAE): {error2[0]:.3f} ;  (MAE-Capped): {error2[1]:.3f}")
    st.write(f"(MAE) Torch Based New Loss (MAE): {error3[0]:.3f} ;  (MAE-Capped): {error3[1]:.3f}")
    st.write(f"(MAE) Custom Based New Loss (MAE): {error4[0]:.3f} ;  (MAE-Capped): {error4[1]:.3f}")

    results = pd.DataFrame()
    results["X"]  = X 
    results["Data"] = y
    results["Base Model"] = y_pred
    results["Regulized Base Model"] = y_pred2
    results["Torch Based Model"] = y_pred3
    results["Custom Based Model"] = y_pred4

   
    fig = px.scatter(results, x="X", y=["Data","Base Model","Regulized Base Model","Torch Based Model","Custom Based Model"])

    st.plotly_chart(fig, use_container_width=True)
    pass

if __name__ == '__main__':
    main()