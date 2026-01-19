import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
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

# Sample categorical options (you can load dynamically from your dataset)
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
# Compute DayOfWeek, Month, Hour
day_of_week = date_input.weekday()
month = date_input.month
hour = time_input.hour

# -----------------------------
# Create DataFrame for prediction
# -----------------------------
input_dict = {
    'Unit price': [unit_price],
    'Quantity': [quantity],
    'DayOfWeek': [day_of_week],
    'Month': [month],
    'Hour': [hour],
    'Branch_B': [0],
    'Branch_C': [0],
    'City_Mandalay': [0],
    'City_Naypyitaw': [0],
    'Customer type_Normal': [0],
    'Gender_Male': [0],
    'Product line_Electronic accessories': [0],
    'Product line_Food and beverages': [0],
    'Product line_Fashion accessories': [0],
    'Product line_Health and beauty': [0],
    'Product line_Home and lifestyle': [0],
    'Product line_Sports and travel': [0],
    'Payment_Ewallet': [0],
    'Payment_Credit card': [0]
}

# Map user input to dummy variables
if branch != 'A':
    input_dict[f'Branch_{branch}'] = [1]
if city != 'Yangon':
    input_dict[f'City_{city}'] = [1]
if customer_type != 'Member':
    input_dict[f'Customer type_{customer_type}'] = [1]
if gender != 'Female':
    input_dict[f'Gender_{gender}'] = [1]
if product_line != 'Electronic accessories':  # base is any first category
    input_dict[f'Product line_{product_line}'] = [1]
if payment != 'Cash':
    input_dict[f'Payment_{payment}'] = [1]

input_df = pd.DataFrame(input_dict)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Sales: {prediction:.2f}")
