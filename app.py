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
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

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

            risk_free_rate = 0.0455  # Updated to ~4.55% (10-Year Treasury as of June 2026)
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
                ef_target.efficient_risk(target_volatility=target_vol_bound)
                target_weights = ef_target.clean_weights()
                target_ret, target_vol_actual, target_sharpe = ef_target.portfolio_performance(risk_free_rate=risk_free_rate)
            except ValueError:
                # Catches error if user sets a target volatility lower than the mathematical minimum possible
                target_ret, target_vol_actual, target_sharpe = 0, 0, 0
                st.warning(f"⚠️ Target volatility of {target_vol_pct}% is lower than the absolute minimum volatility possible for these assets. Adjusting skipped.")

            # --- METRICS ---
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

            # --- PLOTTING ---
            st.subheader("📈 Efficient Frontier Mapping")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Apply the weight bounds to the plotting engine so the curve reflects your constraints
            ef_plot = EfficientFrontier(mu, S, weight_bounds=(min_weight_bound, 1.0))
            plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
            
            ax.scatter(max_s_vol, max_s_ret, marker="*", s=250, c="gold", label="Max Sharpe Portfolio", zorder=5)
            ax.scatter(min_v_vol, min_v_ret, marker="D", s=150, c="green", label="Minimum Volatility", zorder=5)
            ax.scatter(current_volatility, current_return, marker="X", s=300, c="red", label="YOUR PORTFOLIO", zorder=6)
            
            # --- NEW: Plot the Target Risk point if it exists ---
            if target_ret > 0:
                ax.scatter(target_vol_actual, target_ret, marker="P", s=200, c="blue", label=f"Max Return for ≤{target_vol_pct}% Risk", zorder=5)
            
            ax.set_title(f"Your Allocation vs Optimal Frontiers (Min Allocation: {min_weight_pct}%)")
            ax.set_xlabel("Annual Volatility (Risk)")
            ax.set_ylabel("Expected Annual Return")
            ax.legend(loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.5)
            
            st.pyplot(fig)

            # --- DATA TABLE ---
            st.subheader("📊 Target Allocation Adjustments")
            
            # --- NEW: Include the dynamically named Target Risk column ---
            comparison_data = {
                "Your Normalized Allocation": [f"{user_w_vector[i]:.2%}" for i, t in enumerate(actual_tickers)],
                "Max Sharpe Target": [f"{max_s_weights[t]:.2%}" for t in actual_tickers],
                "Min Volatility Target": [f"{min_v_weights[t]:.2%}" for t in actual_tickers],
                f"Max Return at {target_vol_pct}% Risk": [f"{target_weights[t]:.2%}" if target_ret > 0 else "N/A" for t in actual_tickers]
            }
            
            df_compare = pd.DataFrame(comparison_data, index=actual_tickers)
            st.table(df_compare)

        except Exception as e:
            st.error(f"Mathematical Optimization Error: {e}")
            st.info("This usually happens if constraints are too strict (e.g., forcing a high minimum weight on perfectly correlated assets). Try lowering your minimum allocation.")