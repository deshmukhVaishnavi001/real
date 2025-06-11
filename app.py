import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("🏠 Real Estate Price Prediction App")

st.markdown("Enter property details below to estimate the price per unit area.")

# Input fields
house_age = st.slider("House Age (in years)", 0.0, 50.0, 10.0)
distance_to_mrt = st.slider("Distance to Nearest MRT Station (meters)", 0.0, 10000.0, 300.0)
num_convenience = st.slider("Number of Convenience Stores Nearby", 0, 10, 2)
latitude = st.number_input("Latitude", value=24.9670)
longitude = st.number_input("Longitude", value=121.5400)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("real_estate_model.joblib")
    return model

model = load_model()

# Predict
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "X1 transaction date": [transaction_date],
        "X2 house age": [house_age],
        "X3 distance to the nearest MRT station": [distance_to_mrt],
        "X4 number of convenience stores": [num_convenience],
        "X5 latitude": [latitude],
        "X6 longitude": [longitude]
    })

    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price per Unit Area: **{prediction:.2f}**")

# About
st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    "This Streamlit app predicts real estate prices using a machine learning model trained on Taipei housing data."
)
