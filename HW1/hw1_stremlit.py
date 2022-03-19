from unittest import result
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.datasets import fetch_california_housing

from model import ModelBase, ModelRegulizedL2, ModelRegulizedL1, ThresholdBasedModel, TorchBasedModel




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

    threshold = st.slider("Threshold",5.,25.,7.99)
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

    error = np.mean((y_pred-y)**2)
    error2 = np.mean((y_pred2-y)**2)
    error3 = np.mean((y_pred3-y)**2)
    error4 = np.mean((y_pred4-y)**2)

    st.write(f"(MAE) Base Model Error:{error}")
    st.write(f"(MAE) L2 Model Error: {error2}")
    st.write(f"(MAE) Torch Based New Loss: {error3}")
    st.write(f"(MAE) Custom Based New Loss: {error4}")

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