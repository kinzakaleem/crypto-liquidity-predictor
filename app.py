import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("liquidity_model.pkl")

# Streamlit page settings
st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")
st.title(" Cryptocurrency Liquidity Predictor")
st.markdown("### Predict the 24-hour trading volume (liquidity) based on market data")

# User inputs
price = st.number_input(" Current Price (in USD)", min_value=0.0, value=1000.0, step=1.0)
h1 = st.number_input(" % Change in Last 1 Hour", value=0.0, step=0.1)
h24 = st.number_input(" % Change in Last 24 Hours", value=0.0, step=0.1)
d7 = st.number_input(" % Change in Last 7 Days", value=0.0, step=0.1)
market_cap = st.number_input(" Market Capitalization (in USD)", min_value=0.0, value=1_000_000.0, step=1.0)

# Predict button
if st.button("Predict Liquidity"):
    input_df = pd.DataFrame([[price, h1, h24, d7, market_cap]],
                            columns=['price', '1h', '24h', '7d', 'mkt_cap'])
    prediction = model.predict(input_df)[0]
    st.success(f" Predicted 24H Trading Volume: *${prediction:,.2f}*")