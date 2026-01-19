import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -----------------------------
# Fix import for src folder
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.feature_engineering import engineer_features

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "../outputs/models/random_forest_model.pkl"
model = joblib.load(MODEL_PATH)

# -----------------------------
# App Title
# -----------------------------
st.title("Supermarket Sales Prediction")
st.write("Enter transaction details to predict total sales.")

# -----------------------------
# User Inputs
# -----------------------------
st.sidebar.header("Transaction Inputs")

# Sample categorical options (dynamic options can also be loaded from dataset)
branch_options = ['A', 'B', 'C']
city_options = ['Yangon', 'Naypyitaw', 'Mandalay']
customer_type_options = ['Member', 'Normal']
gender_options = ['Male', 'Female']
product_line_options = [
    'Health and beauty', 'Electronic accessories', 
    'Home and lifestyle', 'Sports and travel', 
    'Food and beverages', 'Fashion accessories'
]
payment_options = ['Cash', 'Ewallet', 'Credit card']

branch = st.sidebar.selectbox("Branch", branch_options)
city = st.sidebar.selectbox("City", city_options)
customer_type = st.sidebar.selectbox("Customer Type", customer_type_options)
gender = st.sidebar.selectbox("Gender", gender_options)
product_line = st.sidebar.selectbox("Product Line", product_line_options)
unit_price = st.sidebar.number_input("Unit Price", min_value=0.0, value=50.0)
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
payment = st.sidebar.selectbox("Payment Method", payment_options)

date_input = st.sidebar.date_input("Transaction Date", datetime.today())
time_input = st.sidebar.time_input("Transaction Time", datetime.now().time())

# -----------------------------
# Feature Engineering
# -----------------------------
day_of_week = date_input.weekday()
month = date_input.month
hour = time_input.hour

# Create initial dataframe
input_df = pd.DataFrame({
    'Unit price': [unit_price],
    'Quantity': [quantity],
    'DayOfWeek': [day_of_week],
    'Month': [month],
    'Hour': [hour],
    'Branch': [branch],
    'City': [city],
    'Customer type': [customer_type],
    'Gender': [gender],
    'Product line': [product_line],
    'Payment': [payment]
})

# Use your feature engineering function to apply dummy encoding
input_df = engineer_features(input_df)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Sales: {prediction:.2f}")
