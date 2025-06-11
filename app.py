import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

# Load model
@st.cache_resource
def load_model():
    return joblib.load("real_estate_model.joblib")

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("Real estate valuation data set.xlsx")

data = load_data()

# Sidebar navigation
st.sidebar.title("🔍 Choose Page")
page = st.sidebar.radio("Navigation", ["Home", "Raw Data", "Summary", "Graphs & Charts"])

# --- Page: Home ---
if page == "Home":
    st.title("🏠 Real Estate Price Prediction App")
    st.markdown("Enter property details below to estimate the price per unit area.")

    # Input fields
    transaction_date = st.slider("Transaction Date (e.g., 2013.250 = March 2013)", 2012.0, 2015.0, 2013.25, step=0.01)
    house_age = st.slider("House Age (in years)", 0.0, 50.0, 10.0)
    distance_to_mrt = st.slider("Distance to Nearest MRT Station (meters)", 0.0, 10000.0, 300.0)
    num_convenience = st.slider("Number of Convenience Stores Nearby", 0, 10, 2)
    latitude = st.number_input("Latitude", value=24.9670)
    longitude = st.number_input("Longitude", value=121.5400)

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

# --- Page: Raw Data ---
elif page == "Raw Data":
    st.title("📄 Raw Dataset")
    st.dataframe(data)

# --- Page: Summary ---
elif page == "Summary":
    st.title("📊 Data Summary")
    st.write(data.describe())

# --- Page: Graphs & Charts ---
elif page == "Graphs & Charts":
    st.title("📈 Graphs & Charts")

    st.subheader("Correlation Heatmap")
    corr = data.corr(numeric_only=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt.gcf())

    st.subheader("House Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["X2 house age"], kde=True, ax=ax)
    st.pyplot(fig)

# Sidebar info
st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    "This Streamlit app predicts real estate prices using a machine learning model "
    "trained on Taipei housing data."
)
