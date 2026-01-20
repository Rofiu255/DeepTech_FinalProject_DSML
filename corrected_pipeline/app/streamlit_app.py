import sys
import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -----------------------------
# BASE DIRECTORY
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# IMPORT FEATURE ENGINEERING
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from src.feature_engineering import engineer_features

# -----------------------------
# LOAD MODEL & FEATURE LIST
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "..", "outputs", "models", "random_forest_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "outputs", "models", "training_features.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("❌ Model or feature file not found. Please retrain and redeploy.")
    st.stop()

model = joblib.load(MODEL_PATH)
training_features = joblib.load(FEATURES_PATH)

# -----------------------------
# APP UI
# -----------------------------
st.set_page_config(page_title="Supermarket Sales Predictor", layout="centered")

st.title("🛒 Supermarket Sales Prediction")
st.write("Predict total sales for a supermarket transaction.")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Transaction Details")

branch = st.sidebar.selectbox("Branch", ["A", "B", "C"])
city = st.sidebar.selectbox("City", ["Yangon", "Naypyitaw", "Mandalay"])
customer_type = st.sidebar.selectbox("Customer Type", ["Member", "Normal"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
product_line = st.sidebar.selectbox(
    "Product Line",
    [
        "Health and beauty",
        "Electronic accessories",
        "Home and lifestyle",
        "Sports and travel",
        "Food and beverages",
        "Fashion accessories"
    ]
)
payment = st.sidebar.selectbox("Payment Method", ["Cash", "Ewallet", "Credit card"])

unit_price = st.sidebar.number_input("Unit Price", min_value=0.0, value=50.0)
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)

date_input = st.sidebar.date_input("Transaction Date", datetime.today())
time_input = st.sidebar.time_input("Transaction Time", datetime.now().time())

# -----------------------------
# CREATE INPUT DATAFRAME
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
    "Payment": [payment]
})

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
input_df = engineer_features(input_df)

# -----------------------------
# ALIGN FEATURES (CRITICAL)
# -----------------------------
for col in training_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[training_features]

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Sales: {prediction:.2f}")
