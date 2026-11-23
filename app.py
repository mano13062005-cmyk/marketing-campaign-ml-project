import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Marketing Campaign Predictor", page_icon="📊", layout="wide")

st.title("📊 Marketing Campaign Performance Prediction")
st.write("Enter campaign details to estimate revenue and predict profit or loss.")

required_files = [
    "models/revenue_model.pkl",
    "models/profit_model.pkl",
    "models/metadata.pkl"
]
if not all(os.path.exists(path) for path in required_files):
    st.error("Models are not ready. First run: python train_model.py")
    st.stop()

revenue_model = joblib.load("models/revenue_model.pkl")
profit_model = joblib.load("models/profit_model.pkl")
metadata = joblib.load("models/metadata.pkl")

with st.form("campaign_form"):
    left, right = st.columns(2)

    with left:
        brand = st.selectbox("Brand", ["Nykaa", "Purplle", "Tira"])
        campaign_type = st.selectbox(
            "Campaign Type", ["SEO", "Paid Ads", "Social Media", "Email", "Awareness", "Promotion", "Conversion"]
        )
        target_audience = st.selectbox(
            "Target Audience", ["Working Women", "College Students", "Premium Shoppers", "Tier 2 City Customers", "All"]
        )
        duration = st.number_input("Duration (days)", min_value=1, value=10)
        impressions = st.number_input("Impressions", min_value=0, value=50000)
        clicks = st.number_input("Clicks", min_value=0, value=2000)
        leads = st.number_input("Leads", min_value=0, value=500)

    with right:
        conversions = st.number_input("Conversions", min_value=0, value=200)
        acquisition_cost = st.number_input("Acquisition Cost", min_value=0.0, value=500.0)
        engagement_score = st.number_input("Engagement Score", min_value=0.0, value=10.0)
        language = st.selectbox("Language", ["English", "Tamil", "Hindi", "Bengali", "Telugu", "Unknown"])
        customer_segment = st.selectbox(
            "Customer Segment", ["New Customers", "Returning Customers", "Premium Shoppers", "College Students", "Unknown"]
        )
        channel_text = st.multiselect(
            "Marketing Channels",
            ["Facebook", "Google", "Instagram", "WhatsApp", "YouTube", "Email", "Website"]
        )
        month = st.selectbox("Campaign Month", list(range(1, 13)), index=0)
        year = st.selectbox("Campaign Year", [2024, 2025, 2026], index=1)

    submitted = st.form_submit_button("Predict Campaign Performance")

if submitted:
    row = {
        "Brand": brand,
        "Campaign_Type": campaign_type,
        "Target_Audience": target_audience,
        "Duration": duration,
        "Impressions": impressions,
        "Clicks": clicks,
        "Leads": leads,
        "Conversions": conversions,
        "Acquisition_Cost": acquisition_cost,
        "Engagement_Score": engagement_score,
        "Language": language,
        "Customer_Segment": customer_segment,
        "Campaign_Month": month,
        "Campaign_Year": year
    }

    # Create the same multi-label channel columns used during training.
    for channel_column in metadata["channel_columns"]:
        channel_name = channel_column.replace("Channel_", "").replace("_", " ")
        row[channel_column] = 1 if channel_name in channel_text else 0

    input_data = pd.DataFrame([row])
    revenue = revenue_model.predict(input_data[metadata["all_features"]])[0]
    profit = profit_model.predict(input_data[metadata["classification_features"]])[0]

    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Revenue", f"₹ {revenue:,.2f}")
    if profit == 1:
        col2.success("Predicted result: PROFIT")
    else:
        col2.error("Predicted result: LOSS")

    st.caption("Predictions are estimates based on patterns in the uploaded historical campaign data.")
