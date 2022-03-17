from unittest import result
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.datasets import fetch_california_housing

from model import ModelBase, ModelRegulizedL2, ModelRegulizedL1, ThresholdBasedModel, Model2




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

    threshold = st.slider("Threshold",0.5,10.,7.99)
    myModel3 = Model2(threshold)
    myModel.fit(X,y)
    myModel2.fit(X,y)
    myModel3.fit(X,y)

    y_pred = myModel.pred(X)
    y_pred2 = myModel2.pred(X)
    y_pred3 = myModel3.pred(X)

    error = np.mean((y_pred-y)**2)
    error2 = np.mean((y_pred2-y)**2)
    error3 = np.mean((y_pred3-y)**2)

    st.write(f"(MAE) Base Model Error:{error}")
    st.write(f"(MAE) L2 Model Error: {error2}")
    st.write(f"(MAE) New Model Error: {error3}")

    results = pd.DataFrame()
    results["X"]  = X 
    results["Base"] = y
    results["Model 1"] = y_pred
    results["Model 2"] = y_pred2
    results["Model 3"] = y_pred3

   
    fig = px.scatter(results, x="X", y=["Base","Model 1","Model 2","Model 3"])

    st.plotly_chart(fig, use_container_width=True)
    pass

if __name__ == '__main__':
    main()