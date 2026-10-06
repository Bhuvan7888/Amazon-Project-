import streamlit as st
import requests
import json
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Amazon Supply Chain Intelligence",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Amazon Supply Chain Intelligence")
st.subheader("Delivery Delay Risk Prediction")

# Flask API Endpoint
FLASK_API_URL = "http://localhost:5000/predict"

# Attempt to load features from model_features.pkl to know what inputs to ask for
features = [
    'price', 'freight_value', 'payment_value', 'review_score', 
    'processing_days', 'purchase_month', 'purchase_weekday', 
    'num_sellers', 'total_items', 'payment_installments'
]
FEATURES_PATH = 'model_features.pkl'
if os.path.exists(FEATURES_PATH):
    try:
        features = joblib.load(FEATURES_PATH)
    except:
        pass

st.markdown("""
This application predicts the risk of delivery delay for an Amazon order based on historical data.
Please input the order details below and click **Predict** to see the estimated delivery risk.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🛒 Order Value & Shipping")
    price = st.number_input("Item Price (BRL)", min_value=0.0, value=50.0, step=1.0)
    freight_value = st.number_input("Freight Value (BRL)", min_value=0.0, value=15.0, step=1.0)
    total_items = st.number_input("Total Items in Order", min_value=1, value=1, step=1)
    
with col2:
    st.markdown("### 💳 Payment Details")
    payment_value = st.number_input("Total Payment Value (BRL)", min_value=0.0, value=65.0, step=1.0)
    payment_installments = st.number_input("Payment Installments", min_value=1, max_value=24, value=1, step=1)
    num_sellers = st.number_input("Number of Sellers", min_value=1, value=1, step=1)

with col3:
    st.markdown("### 📅 Timing & Feedback")
    processing_days = st.number_input("Processing Days (Approval time)", min_value=0, value=1, step=1)
    purchase_month = st.selectbox("Purchase Month", options=list(range(1, 13)), index=5) # Default June
    
    # Map weekday number to string
    weekdays = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    purchase_weekday_str = st.selectbox("Purchase Weekday", options=list(weekdays.values()))
    purchase_weekday = [k for k, v in weekdays.items() if v == purchase_weekday_str][0]
    
    review_score = st.slider("Past/Expected Review Score", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

st.divider()

if st.button("🚀 Predict Delivery Risk", use_container_width=True):
    # Construct payload
    payload = {
        "price": price,
        "freight_value": freight_value,
        "payment_value": payment_value,
        "review_score": review_score,
        "processing_days": processing_days,
        "purchase_month": purchase_month,
        "purchase_weekday": purchase_weekday,
        "num_sellers": num_sellers,
        "total_items": total_items,
        "payment_installments": payment_installments
    }
    
    with st.spinner("Analyzing order details..."):
        try:
            response = requests.post(FLASK_API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                risk_level = result.get('prediction', 0)
                risk_label = result.get('risk_label', 'Unknown')
                
                st.subheader("Prediction Result")
                
                if risk_level == 0:
                    st.success(f"✅ **{risk_label}** - The order is expected to be delivered on time or early.")
                elif risk_level == 1:
                    st.warning(f"⚠️ **{risk_label}** - The order might experience a slight delay.")
                else:
                    st.error(f"🚨 **{risk_label}** - High chance of significant delay. Consider proactive customer communication.")
                    
            else:
                st.error(f"Error from API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Failed to connect to the backend. Please ensure the Flask server is running on localhost:5000.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

