import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/best_model.pkl")

st.set_page_config(page_title="Supermarket Sales Predictor", layout="wide")

st.title("🛒 Supermarket Sales Prediction Dashboard")

st.sidebar.header("Transaction Details")

unit_price = st.sidebar.slider("Unit Price", 1.0, 100.0, 50.0)
quantity = st.sidebar.slider("Quantity", 1, 10, 3)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
month = st.sidebar.slider("Month", 1, 12, 6)

input_data = pd.DataFrame({
    'Unit price':[unit_price],
    'Quantity':[quantity],
    'Time':[hour],
    'Month':[month]
})

prediction = model.predict(input_data)

st.metric("Predicted Sales ($)", round(prediction[0],2))

st.subheader("📌 Business Insights")
st.markdown("""
- Quantity and Unit Price drive most sales
- Afternoon hours show higher transaction value
- Members & e-wallet users spend more
""")
