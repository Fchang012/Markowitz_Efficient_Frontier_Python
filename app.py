import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import plotting
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Local Portfolio Optimizer", layout="wide")
st.title("Self-Hosted Portfolio Analyzer & Frontier Comparison")
st.write("Compare your actual portfolio against the efficient frontier locally and privately.")

# Cache data to prevent Yahoo Finance rate-limiting/blocking inside your container
@st.cache_data(ttl="1d")
def load_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)['Close']
        return data
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Asset Selection")
tickers_input = st.sidebar.text_input("Enter Tickers (comma-separated)", "VTI, VXUS")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2016-01-01"))

# 1. Initialize the session state for the end date if it doesn't exist
if "end_date_key" not in st.session_state:
    st.session_state["end_date_key"] = pd.to_datetime("today").date()

# 2. Define the callback function to reset the state to today
def set_today():
    st.session_state["end_date_key"] = pd.to_datetime("today").date()

# 3. Create a side-by-side layout in the sidebar
col1, col2 = st.sidebar.columns([3, 2])
with col1:
    # Tie the date input to the session state key
    end_date = st.date_input("End Date", key="end_date_key")
with col2:
    st.write("") # Adds a little vertical spacing to align the button
    st.write("")
    st.button("Today", on_click=set_today)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.sidebar.header("2. Your Current Allocation (%)")
current_weights = {}

if tickers:
    default_weight = float(100.0 / len(tickers))
    for ticker in tickers:
        current_weights[ticker] = st.sidebar.number_input(
            f"{ticker} Weight (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=default_weight,
            step=0.1,
            format="%.2f"
        )

total_weight = sum(current_weights.values())
st.sidebar.write(f"**Total Specified:** {total_weight:.2f}%")

if abs(total_weight - 100.0) > 0.01:
    st.sidebar.warning("⚠️ Weights do not equal 100%. They will be mathematically normalized for accurate comparison.")

# --- OPTIMIZATION CONSTRAINTS ---
st.sidebar.header("3. Optimization Constraints")
min_weight_pct = st.sidebar.number_input(
    "Min Allocation per Asset (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=1.0,  # Default to forcing at least 1% in every asset
    step=1.0,
    help="Force the optimizer to hold at least this percentage of every asset to ensure diversification."
)

# Convert percentage to a decimal for the math engine
min_weight_bound = min_weight_pct / 100.0

# Mathematical Validation: Ensure the constraint is physically possible
if tickers and (min_weight_bound * len(tickers) > 1.0):
    st.sidebar.error(f"❌ Impossible Constraint! A minimum of {min_weight_pct}% across {len(tickers)} assets equals {min_weight_pct * len(tickers)}%. It must be 100% or less.")
    st.stop()

# --- NEW: TARGET RISK INPUT ---
st.sidebar.header("4. Target Risk Profile")
target_vol_pct = st.sidebar.number_input(
    "Max Tolerable Volatility (%)", 
    min_value=1.0, 
    max_value=100.0, 
    value=15.0,  # Example: 15% annual volatility
    step=1.0,
    help="Finds the absolute maximum return possible without exceeding this volatility level."
)
target_vol_bound = target_vol_pct / 100.0

# --- MAIN ANALYSIS EXECUTION ---
if st.sidebar.button("Run Comparison Analysis") and tickers:
    with st.spinner("Processing local financial data..."):
        data = load_data(tickers, start_date, end_date)
        
        if data.empty:
            st.error("No historical data found. Please check ticker spellings or network configuration.")
            st.stop()
            
        actual_tickers = data.columns.tolist()
        
        try:
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)

            user_w_vector = np.array([current_weights[t] for t in actual_tickers])
            if np.sum(user_w_vector) > 0:
                user_w_vector = user_w_vector / np.sum(user_w_vector)
            else:
                st.error("Total allocation cannot be 0%.")
                st.stop()

            risk_free_rate = 0.0455  # Updated to ~4.55% (10-Year Treasury proxy)
            current_return = np.dot(user_w_vector, mu)
            current_volatility = np.sqrt(np.dot(user_w_vector.T, np.dot(S, user_w_vector)))
            current_sharpe = (current_return - risk_free_rate) / current_volatility

            # Apply the weight bounds to the Max Sharpe Optimizer
            ef_max = EfficientFrontier(mu, S, weight_bounds=(min_weight_bound, 1.0))
            ef_max.max_sharpe(risk_free_rate=risk_free_rate)
            max_s_weights = ef_max.clean_weights()
            max_s_ret, max_s_vol, max_s_sharpe = ef_max.portfolio_performance(risk_free_rate=risk_free_rate)

            # Apply the weight bounds to the Min Volatility Optimizer
            ef_min = EfficientFrontier(mu, S, weight_bounds=(min_weight_bound, 1.0))
            ef_min.min_volatility()
            min_v_weights = ef_min.clean_weights()
            min_v_ret, min_v_vol, min_v_sharpe = ef_min.portfolio_performance(risk_free_rate=risk_free_rate)

            # --- NEW: Apply the weight bounds to the Target Risk Optimizer ---
            ef_target = EfficientFrontier(mu, S, weight_bounds=(min_weight_bound, 1.0))
            try:
                # Maximize return for a target volatility
                ef_target