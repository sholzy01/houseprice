import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("House Price Prediction App")

# Load model
model = joblib.load("house_price_model.pkl")

st.write("Enter the house features below:")

longitude = st.number_input("Longitude", step=0.01)
latitude = st.number_input("Latitude", step=0.01)
housing_median_age = st.number_input("Housing Median Age", min_value=1, max_value=100)
total_rooms = st.number_input("Total Rooms", min_value=1)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1)
population = st.number_input("Population", min_value=1)
households = st.number_input("Households", min_value=1)
median_income = st.number_input("Median Income", step=0.1)

# Prepare input
input_data = np.array([[longitude, latitude, housing_median_age,
                        total_rooms, total_bedrooms, population,
                        households, median_income]])

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")
