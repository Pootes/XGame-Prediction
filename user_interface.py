""" import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/randomforest_model.pkl")

st.title("Marketing Campaign Conversion Predictor")

uploaded = st.file_uploader("Upload CSV with Campaign Details")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Raw Data", df.head())

    from cleaning import clean_data
    from preprocessing import preprocess_data

    df_clean = clean_data(uploaded)
    df_ready = preprocess_data(df_clean)

    preds = model.predict(df_ready.drop(columns=['Conversion_Rate'], errors='ignore'))
    st.write("Predicted Conversion Rates:", preds)
 """