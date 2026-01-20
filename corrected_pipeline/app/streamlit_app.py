import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -----------------------------
# BASE DIRECTORY (CRITICAL FIX)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Fix import for src folder
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from src.feature_engineering import engineer_features

# -----------------------------
# Load trained model (SAFE PATH)
# -----------------------------
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "outputs",
    "models",
    "random_forest_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please check deployment paths.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# App Title
# -----------------------------
st.title("🛒 Supermarket Sales Prediction")
st.write("Enter transaction details to predict total sales.")

# -----------------------------
# User Inputs
# -----------------------------
st.sidebar.header("Transaction Inputs")

branch = st.sidebar.selectbox("Branch", ['A', 'B', 'C'])
city = st.sidebar.selectbox("City", ['Yangon', 'Naypyitaw', 'Mandalay'])
customer_type = st.sidebar.selectbox("Customer Type", ['Member', 'Normal'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
product_line = st.sidebar.selectbox(
    "Product Line",
    [
        'Health and beauty',
        'Electronic accessories',
        'Home and lifestyle',
        'Sports and travel',
        'Food and beverages',
        'Fashion accessories'
    ]
)
payment = st.sidebar.selectbox("Payment Method", ['Cash', 'Ewallet', 'Credit card'])

unit_price = st.sidebar.number_input("Unit Price", min_value=0.0, value=50.0)
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)

date_input = st.sidebar.date_input("Transaction Date", datetime.today())
time_input = st.sidebar.time_input("Transaction Time", datetime.now().time())

# -----------------------------
# Feature Engineering
# -----------------------------
input_df = pd.DataFrame({
    "Unit price": [unit_price],
    "Quantity": [quantity],
    "DayOfWeek": [date_input.weekday()],
    "Month": [date_input.month],
    "Hour": [time_input.hour],
    "Branch": [branch],
    "City": [city],
    "Customer type": [customer_type],
    "Gender": [gender],
    "Product line": [product_line],
    "Payment": [payment],
})

input_df = engineer_features(input_df)

# -----------------------------
# Align features with training
# -----------------------------
model_features = model.feature_names_in_
input_df = input_df.reindex(columns=model_features, fill_value=0)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Sales: {prediction:.2f}")
