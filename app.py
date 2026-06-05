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
        # Fetch adjusted close prices
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        return data
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Asset Selection")
tickers_input = st.sidebar.text_input("Enter Tickers (comma-separated)", "SPY, QQQ, BND, GLD")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Clean and parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.sidebar.header("2. Your Current Allocation (%)")
current_weights = {}

# Dynamically generate sliders for input tickers
if tickers:
    default_weight = 100.0 / len(tickers)
    for ticker in tickers:
        current_weights[ticker] = st.sidebar.slider(
            f"{ticker} Weight", 
            min_value=0.0, 
            max_value=100.0, 
            value=default_weight,
            step=0.5
        )

# Check allocation totals
total_weight = sum(current_weights.values())
st.sidebar.write(f"**Total Specified:** {total_weight:.1f}%")

if abs(total_weight - 100.0) > 0.1:
    st.sidebar.warning("⚠️ Weights do not equal 100%. They will be mathematically normalized for accurate comparison.")

# --- MAIN ANALYSIS EXECUTION ---
if st.sidebar.button("Run Comparison Analysis") and tickers:
    with st.spinner("Processing local financial data..."):
        data = load_data(tickers, start_date, end_date)
        
        if data.empty:
            st.error("No historical data found. Please check ticker spellings or network configuration.")
            st.stop()
            
        # Ensure data columns match our ticker tracking order
        actual_tickers = data.columns.tolist()
        
        try:
            # Calculate mean historical returns and sample covariance matrix
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)

            # Map and normalize user weights to a vector summing to 1.0
            user_w_vector = np.array([current_weights[t] for t in actual_tickers])
            if np.sum(user_w_vector) > 0:
                user_w_vector = user_w_vector / np.sum(user_w_vector)
            else:
                st.error("Total allocation cannot be 0%.")
                st.stop()

            # Calculate metrics for the user's current portfolio
            risk_free_rate = 0.02 
            current_return = np.dot(user_w_vector, mu)
            current_volatility = np.sqrt(np.dot(user_w_vector.T, np.dot(S, user_w_vector)))
            current_sharpe = (current_return - risk_free_rate) / current_volatility

            # Calculate Max Sharpe Ratio Benchmark
            ef_max = EfficientFrontier(mu, S)
            ef_max.max_sharpe(risk_free_rate=risk_free_rate)
            max_s_weights = ef_max.clean_weights()
            max_s_ret, max_s_vol, max_s_sharpe = ef_max.portfolio_performance(risk_free_rate=risk_free_rate)

            # Calculate Minimum Volatility Benchmark
            ef_min = EfficientFrontier(mu, S)
            ef_min.min_volatility()
            min_v_weights = ef_min.clean_weights()
            min_v_ret, min_v_vol, min_v_sharpe = ef_min.portfolio_performance(risk_free_rate=risk_free_rate)

            # --- DISPLAY DASHBOARD METRICS ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("📋 Your Portfolio Sharpe", f"{current_sharpe:.2f}")
                st.caption(f"Return: {current_return:.2%} | Volatility: {current_volatility:.2%}")
            with m2:
                st.metric("🏆 Max Sharpe Benchmark", f"{max_s_sharpe:.2f}", f"+{max_s_sharpe - current_sharpe:.2f} vs Yours")
                st.caption(f"Return: {max_s_ret:.2%} | Volatility: {max_s_vol:.2%}")
            with m3:
                st.metric("🛡️ Min Volatility Benchmark", f"{min_v_vol:.2%}", f"{current_volatility - min_v_vol:+.2%} Risk Diff", delta_color="inverse")
                st.caption(f"Return: {min_v_ret:.2%} | Sharpe: {min_v_sharpe:.2f}")

            # --- VISUALIZATION: EFFICIENT FRONTIER ---
            st.subheader("📈 Efficient Frontier Mapping")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate the baseline frontier curve
            ef_plot = EfficientFrontier(mu, S)
            plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
            
            # Plot reference markers
            ax.scatter(max_s_vol, max_s_ret, marker="*", s=250, c="gold", label="Max Sharpe Portfolio", zorder=5)
            ax.scatter(min_v_vol, min_v_ret, marker="D", s=150, c="green", label="Minimum Volatility", zorder=5)
            ax.scatter(current_volatility, current_return, marker="X", s=300, c="red", label="YOUR PORTFOLIO", zorder=6)
            
            ax.set_title("Your Allocation vs Optimal Frontiers")
            ax.xlabel("Annual Volatility (Risk)")
            ax.ylabel("Expected Annual Return")
            ax.legend(loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.5)
            
            st.pyplot(fig)

            # --- ALLOCATION BREAKDOWN COMPARISON ---
            st.subheader("📊 Target Allocation Adjustments")
            
            comparison_data = {
                "Your Normalized Allocation": [f"{user_w_vector[i]:.2%}" for i, t in enumerate(actual_tickers)],
                "Max Sharpe Target": [f"{max_s_weights[t]:.2%}" for t in actual_tickers],
                "Min Volatility Target": [f"{min_v_weights[t]:.2%}" for t in actual_tickers]
            }
            
            df_compare = pd.DataFrame(comparison_data, index=actual_tickers)
            st.table(df_compare)

        except Exception as e:
            st.error(f"Mathematical Optimization Error: {e}")
            st.info("This usually happens if assets are perfectly correlated or if date ranges are too narrow.")