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

@st.cache_data(ttl="15m")
def fetch_current_price(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price:
            return float(price)
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return None
    except:
        return None

def analyze_pairings(holdings_with_prices, federal_marginal, ltcg_rate, state_rate, niit_rate):
    """
    holdings_with_prices: list of dicts with keys: ticker, basis, shares, holding, current_price
    Returns: list of pairing results sorted by savings descending
    """
    loss_assets = [h for h in holdings_with_prices if h["current_price"] < h["basis"]]
    gain_assets = [h for h in holdings_with_prices if h["current_price"] > h["basis"]]

    pairings = []
    for loss in loss_assets:
        for gain in gain_assets:
            realized_loss = (loss["current_price"] - loss["basis"]) * loss["shares"]  # negative
            realized_gain = (gain["current_price"] - gain["basis"]) * gain["shares"]  # positive

            # Determine federal rate based on gain asset holding period
            is_long_term = "≥ 1 Year" in gain["holding"]
            federal_rate = ltcg_rate if is_long_term else federal_marginal
            combined_rate = (federal_rate + state_rate + niit_rate) / 100.0

            tax_without = max(0.0, realized_gain * combined_rate)
            net_gain = realized_gain + realized_loss
            tax_with = max(0.0, net_gain * combined_rate)
            savings = tax_without - tax_with

            pairings.append({
                "loss_ticker": loss["ticker"],
                "gain_ticker": gain["ticker"],
                "realized_loss": realized_loss,
                "realized_gain": realized_gain,
                "net_taxable": max(0, net_gain),
                "combined_rate": combined_rate,
                "tax_without": tax_without,
                "tax_with": tax_with,
                "savings": savings,
            })

    pairings.sort(key=lambda x: x["savings"], reverse=True)
    return pairings, loss_assets, gain_assets

# Setup Sidebar top level selector
active_tool = st.sidebar.radio("Active Tool", ["📈 Portfolio Optimizer", "🧾 Tax-Loss Harvesting Simulator"])
st.sidebar.markdown("---")

tab1, tab2 = st.tabs(["📈 Portfolio Optimizer", "🧾 Tax-Loss Harvesting Simulator"])

# --- TAB 1: PORTFOLIO OPTIMIZER ---
with tab1:
    if active_tool == "📈 Portfolio Optimizer":
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
        
        run_analysis = st.sidebar.button("Run Comparison Analysis")

    # The action output happens in main area if button clicked
    if active_tool == "📈 Portfolio Optimizer" and 'run_analysis' in locals() and run_analysis and tickers:
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

# --- TAB 2: TAX-LOSS HARVESTING SIMULATOR ---
with tab2:
    if "tlh_holdings_count" not in st.session_state:
        st.session_state["tlh_holdings_count"] = 2

    state_rates = {
        "Alaska": 0.0, "Florida": 0.0, "Nevada": 0.0, "New Hampshire": 0.0, 
        "South Dakota": 0.0, "Tennessee": 0.0, "Texas": 0.0, "Washington": 0.0, "Wyoming": 0.0,
        "California": 13.3, "New York": 10.9, "New Jersey": 10.75, "Oregon": 9.9, "Minnesota": 9.85, 
        "Vermont": 8.75, "Iowa": 8.53, "District of Columbia": 10.75, "Hawaii": 11.0, 
        "Connecticut": 6.99, "Massachusetts": 9.0, "Wisconsin": 7.65, "Maine": 7.15, 
        "South Carolina": 6.5, "Idaho": 5.8, "Montana": 6.75, "Kansas": 5.7, "Nebraska": 6.84, 
        "West Virginia": 6.5, "Arkansas": 4.4, "Georgia": 5.49, "Kentucky": 4.0, "Colorado": 4.4, 
        "Virginia": 5.75, "North Carolina": 4.5, "Michigan": 4.25, "Indiana": 3.05, "Utah": 4.65, 
        "Ohio": 3.5, "Pennsylvania": 3.07, "Illinois": 4.95, "Arizona": 2.5, "New Mexico": 5.9, 
        "Alabama": 5.0, "Missouri": 4.8, "Oklahoma": 4.75, "Louisiana": 4.25, "Mississippi": 5.0, 
        "North Dakota": 1.95, "Rhode Island": 5.99, "Delaware": 6.6, "Maryland": 5.75
    }

    if active_tool == "🧾 Tax-Loss Harvesting Simulator":
        st.sidebar.header("1. Holdings (Batch Mode)")
        
        holdings = []
        for i in range(st.session_state["tlh_holdings_count"]):
            st.sidebar.markdown(f"**Holding {i+1}**")
            ticker = st.sidebar.text_input("Ticker", key=f"tlh_ticker_{i}", value="PFE" if i == 0 else ("NVDA" if i==1 else ""))
            basis = st.sidebar.number_input("Purchase Price/Share", min_value=0.01, value=50.0 if i==0 else 200.0, key=f"tlh_basis_{i}")
            shares = st.sidebar.number_input("Shares", min_value=1.0, step=1.0, value=100.0 if i==0 else 50.0, key=f"tlh_shares_{i}")
            holding = st.sidebar.selectbox("Holding Period", ["< 1 Year (Short-Term)", "≥ 1 Year (Long-Term)"], key=f"tlh_holding_{i}")
            st.sidebar.divider()
            
            if ticker:
                holdings.append({
                    "ticker": ticker.strip().upper(),
                    "basis": basis,
                    "shares": shares,
                    "holding": holding
                })

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("➕ Add Holding", key="tlh_add"):
                st.session_state["tlh_holdings_count"] += 1
                st.rerun()
        with col2:
            if st.button("🗑️ Remove Last", key="tlh_rem") and st.session_state["tlh_holdings_count"] > 1:
                st.session_state["tlh_holdings_count"] -= 1
                st.rerun()

        st.sidebar.header("2. Tax Configuration")
        federal_marginal = st.sidebar.number_input("Marginal Federal Tax Bracket (%)", value=24.0, key="tlh_marginal")
        ltcg_rate = st.sidebar.number_input("Long-Term Capital Gains Rate (%)", value=15.0, key="tlh_ltcg")
        
        state_list = sorted(list(state_rates.keys()))
        default_state_idx = state_list.index("Texas")
        selected_state = st.sidebar.selectbox("State", state_list, index=default_state_idx, key="tlh_state")
        
        default_state_rate = state_rates.get(selected_state, 0.0)
        state_rate = st.sidebar.number_input("State Capital Gains Rate (%)", value=float(default_state_rate), key="tlh_state_rate")
        
        niit_enabled = st.sidebar.checkbox("Include NIIT (3.8%)", value=False, key="tlh_niit", help="Net Investment Income Tax applies to high earners (>$200k single / >$250k married).")
        niit_rate = 3.8 if niit_enabled else 0.0

        run_tlh = st.sidebar.button("🔍 Analyze All Pairings", key="tlh_run")

    st.header("🧾 Tax-Loss Harvesting Simulator")
    st.write("Simulate offsetting capital gains by selling assets currently at a loss.")

    if active_tool == "🧾 Tax-Loss Harvesting Simulator" and 'run_tlh' in locals() and run_tlh and len(holdings) > 0:
        with st.spinner("Fetching current prices..."):
            holdings_with_prices = []
            for h in holdings:
                cp = fetch_current_price(h["ticker"])
                if cp is None:
                    st.warning(f"Could not fetch price for {h['ticker']}. Skipping this holding.")
                    continue
                
                unrealized_pl = (cp - h["basis"]) * h["shares"]
                status = "📉 LOSS" if unrealized_pl < 0 else ("📈 GAIN" if unrealized_pl > 0 else "➡️ FLAT")
                
                h_new = h.copy()
                h_new["current_price"] = cp
                h_new["unrealized_pl"] = unrealized_pl
                h_new["status"] = status
                holdings_with_prices.append(h_new)

        if len(holdings_with_prices) > 0:
            st.subheader("Your Holdings Summary")
            summary_df = pd.DataFrame(holdings_with_prices)
            # Reorder and format columns
            summary_df = summary_df[["ticker", "basis", "current_price", "shares", "unrealized_pl", "status"]]
            summary_df.columns = ["Ticker", "Purchase Price", "Current Price", "Shares", "Unrealized P&L", "Status"]
            
            st.dataframe(summary_df.style.format({
                "Purchase Price": "${:,.2f}", 
                "Current Price": "${:,.2f}", 
                "Unrealized P&L": "${:,.2f}"
            }))

            pairings, loss_assets, gain_assets = analyze_pairings(holdings_with_prices, federal_marginal, ltcg_rate, state_rate, niit_rate)

            if not loss_assets:
                st.warning("None of your holdings are currently at a loss. There are no tax-loss harvesting opportunities.")
            elif not gain_assets:
                st.info("None of your holdings are currently at a gain. No capital gains tax to offset.")
            elif not pairings:
                st.info("No valid pairings found.")
            else:
                st.subheader("🏆 Optimal Pairing Rankings")
                rankings_data = []
                for idx, p in enumerate(pairings):
                    rankings_data.append({
                        "Rank": idx + 1,
                        "Sell at Loss": p["loss_ticker"],
                        "Sell at Gain": p["gain_ticker"],
                        "Realized Loss": p["realized_loss"],
                        "Realized Gain": p["realized_gain"],
                        "Net Taxable": p["net_taxable"],
                        "Tax Without Harvesting": p["tax_without"],
                        "Tax With Harvesting": p["tax_with"],
                        "💰 Savings": p["savings"]
                    })
                
                rankings_df = pd.DataFrame(rankings_data)
                st.dataframe(rankings_df.style.format({
                    "Realized Loss": "${:,.2f}",
                    "Realized Gain": "${:,.2f}",
                    "Net Taxable": "${:,.2f}",
                    "Tax Without Harvesting": "${:,.2f}",
                    "Tax With Harvesting": "${:,.2f}",
                    "💰 Savings": "${:,.2f}"
                }))

                best_pairing = pairings[0]
                
                st.subheader("📋 Detailed Breakdown — Best Pairing")
                breakdown_data = {
                    "Line Item": [
                        "Realized Gain", "Realized Loss", "Net Taxable Gain", 
                        "Federal Rate Applied", "State Rate Applied", "NIIT Rate Applied", "Combined Rate", 
                        "Estimated Tax (Without Harvesting)", "Estimated Tax (With Harvesting)", "💰 Tax Savings"
                    ],
                    "Without Harvesting": [
                        f"${best_pairing['realized_gain']:,.2f}", "—", f"${best_pairing['realized_gain']:,.2f}",
                        f"{(best_pairing['combined_rate']*100 - state_rate - niit_rate):.1f}%", f"{state_rate:.1f}%", f"{niit_rate:.1f}%", f"{best_pairing['combined_rate']*100:.1f}%",
                        f"${best_pairing['tax_without']:,.2f}", "—", "—"
                    ],
                    "With Harvesting": [
                        f"${best_pairing['realized_gain']:,.2f}", f"${best_pairing['realized_loss']:,.2f}", f"${best_pairing['net_taxable']:,.2f}",
                        f"{(best_pairing['combined_rate']*100 - state_rate - niit_rate):.1f}%", f"{state_rate:.1f}%", f"{niit_rate:.1f}%", f"{best_pairing['combined_rate']*100:.1f}%",
                        "—", f"${best_pairing['tax_with']:,.2f}", f"${best_pairing['savings']:,.2f}"
                    ]
                }
                st.table(pd.DataFrame(breakdown_data).set_index("Line Item"))

                m1, m2, m3 = st.columns(3)
                eff_rate_reduction = 0.0
                if best_pairing['realized_gain'] > 0:
                    eff_rate_reduction = (best_pairing['tax_without'] / best_pairing['realized_gain']) - (best_pairing['tax_with'] / best_pairing['realized_gain'])
                
                best_loss_h = next(h for h in holdings_with_prices if h["ticker"] == best_pairing["loss_ticker"])
                best_gain_h = next(h for h in holdings_with_prices if h["ticker"] == best_pairing["gain_ticker"])
                
                total_proceeds = (best_loss_h["current_price"] * best_loss_h["shares"]) + (best_gain_h["current_price"] * best_gain_h["shares"]) - best_pairing['tax_with']

                m1.metric("💰 Maximum Tax Savings", f"${best_pairing['savings']:,.2f}")
                m2.metric("📉 Effective Rate Reduction", f"{eff_rate_reduction*100:.1f} pts")
                m3.metric("📊 Total Proceeds After Tax", f"${total_proceeds:,.2f}")

                st.subheader("Tax Impact Comparison — Top Pairings")
                
                top_3 = pairings[:3]
                labels = [f"{p['loss_ticker']} → {p['gain_ticker']}" for p in top_3]
                tax_without = [p["tax_without"] for p in top_3]
                tax_with = [p["tax_with"] for p in top_3]

                x = np.arange(len(labels))
                width = 0.35

                fig, ax = plt.subplots(figsize=(8, 5))
                rects1 = ax.bar(x - width/2, tax_without, width, label='Without Harvesting', color='indianred')
                rects2 = ax.bar(x + width/2, tax_with, width, label='With Harvesting', color='mediumseagreen')

                ax.set_ylabel('Tax ($)')
                ax.set_title('Tax Owed: Without vs. With Harvesting')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()

                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'${height:,.0f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),  
                                    textcoords="offset points",
                                    ha='center', va='bottom')

                autolabel(rects1)
                autolabel(rects2)

                st.pyplot(fig)

    with st.expander("⚠️ Important Disclaimer"):
        st.write("""
        This tool is for educational and illustrative purposes only and does not constitute tax, legal, or investment advice. 
        Consult a qualified tax professional before making any investment decisions. 
        This calculator uses simplified assumptions and does not account for wash-sale rules (IRS 30-day rule), 
        alternative minimum tax (AMT), complex multi-lot cost basis calculations, qualified dividends, 
        or capital loss carryforward rules beyond the current tax year.
        """)