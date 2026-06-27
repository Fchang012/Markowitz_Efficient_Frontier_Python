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

def compute_portfolio_tax(holdings, federal_marginal, ltcg_rate, state_rate, niit_rate):
    """
    Implements IRS-compliant portfolio-wide netting (Schedule D logic).
    
    Args:
        holdings: list of dicts, each with keys: ticker, basis, shares, holding, current_price
        federal_marginal: float, marginal tax rate % (e.g. 24.0)
        ltcg_rate: float, long-term capital gains rate % (e.g. 15.0)
        state_rate: float, state capital gains rate % (e.g. 0.0)
        niit_rate: float, NIIT rate % (e.g. 3.8 or 0.0)
    
    Returns: dict with all netting waterfall values
    """
    st_gains = 0.0
    st_losses = 0.0
    lt_gains = 0.0
    lt_losses = 0.0
    
    for h in holdings:
        pl = (h["current_price"] - h["basis"]) * h["shares"]
        is_short = "< 1 Year" in h["holding"]
        if pl >= 0:
            if is_short:
                st_gains += pl
            else:
                lt_gains += pl
        else:
            if is_short:
                st_losses += pl
            else:
                lt_losses += pl
    
    net_st = st_gains + st_losses
    net_lt = lt_gains + lt_losses
    
    st_combined_rate = (federal_marginal + state_rate + niit_rate) / 100.0
    lt_combined_rate = (ltcg_rate + state_rate + niit_rate) / 100.0
    
    # Cross-category netting per IRS Schedule D rules
    if net_st >= 0 and net_lt >= 0:
        tax_on_st = net_st * st_combined_rate
        tax_on_lt = net_lt * lt_combined_rate
        overall_net = net_st + net_lt
        ordinary_income_deduction = 0.0
        carryforward = 0.0
    elif net_st < 0 and net_lt < 0:
        total_loss = net_st + net_lt
        tax_on_st = 0.0
        tax_on_lt = 0.0
        overall_net = total_loss
        ordinary_income_deduction = min(abs(total_loss), 3000.0)
        carryforward = max(0, abs(total_loss) - 3000.0)
    elif net_st >= 0 and net_lt < 0:
        cross_netted = net_st + net_lt
        if cross_netted >= 0:
            tax_on_st = cross_netted * st_combined_rate
            tax_on_lt = 0.0
            ordinary_income_deduction = 0.0
            carryforward = 0.0
        else:
            tax_on_st = 0.0
            tax_on_lt = 0.0
            ordinary_income_deduction = min(abs(cross_netted), 3000.0)
            carryforward = max(0, abs(cross_netted) - 3000.0)
        overall_net = cross_netted
    else:  # net_st < 0 and net_lt >= 0
        cross_netted = net_lt + net_st
        if cross_netted >= 0:
            tax_on_st = 0.0
            tax_on_lt = cross_netted * lt_combined_rate
            ordinary_income_deduction = 0.0
            carryforward = 0.0
        else:
            tax_on_st = 0.0
            tax_on_lt = 0.0
            ordinary_income_deduction = min(abs(cross_netted), 3000.0)
            carryforward = max(0, abs(cross_netted) - 3000.0)
        overall_net = cross_netted
    
    total_tax = tax_on_st + tax_on_lt
    ordinary_income_tax_benefit = ordinary_income_deduction * (federal_marginal / 100.0)
    
    return {
        "st_gains": st_gains, "st_losses": st_losses,
        "lt_gains": lt_gains, "lt_losses": lt_losses,
        "net_st": net_st, "net_lt": net_lt,
        "overall_net": overall_net,
        "tax_on_st": tax_on_st, "tax_on_lt": tax_on_lt,
        "total_tax": total_tax,
        "st_combined_rate": st_combined_rate,
        "lt_combined_rate": lt_combined_rate,
        "ordinary_income_deduction": ordinary_income_deduction,
        "ordinary_income_tax_benefit": ordinary_income_tax_benefit,
        "carryforward": carryforward,
    }

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
        
        input_method = st.sidebar.radio("Input Method", ["Manual Entry", "CSV Upload"])
        
        holdings = []
        if input_method == "CSV Upload":
            csv_file = st.sidebar.file_uploader("Upload Holdings CSV", type=["csv"])
            
            st.sidebar.caption("Required format:")
            st.sidebar.code("ticker,purchase_price,shares,holding_period\nAAPL,150.00,100,long\nPFE,50.00,200,short")
            
            csv_template = "ticker,purchase_price,shares,holding_period\nAAPL,150.00,100,long\nPFE,50.00,200,short"
            st.sidebar.download_button(
                label="📋 Download Template",
                data=csv_template,
                file_name="tlh_template.csv",
                mime="text/csv"
            )
            
            if csv_file is not None:
                try:
                    df = pd.read_csv(csv_file)
                    df.columns = df.columns.str.strip().str.lower()
                    required_columns = ["ticker", "purchase_price", "shares", "holding_period"]
                    if not all(col in df.columns for col in required_columns):
                        st.sidebar.error(f"Missing required columns. Expected: {', '.join(required_columns)}")
                    else:
                        invalid_rows = []
                        for index, row in df.iterrows():
                            ticker = str(row['ticker']).strip().upper()
                            purchase_price = pd.to_numeric(row['purchase_price'], errors='coerce')
                            shares = pd.to_numeric(row['shares'], errors='coerce')
                            holding_period = str(row['holding_period']).strip().lower()
                            
                            if not ticker or ticker == 'NAN' or pd.isna(purchase_price) or purchase_price <= 0 or pd.isna(shares) or shares <= 0 or holding_period not in ['short', 'long']:
                                invalid_rows.append(index + 2) # +2 for 1-based indexing and header
                                continue
                                
                            holding = "< 1 Year (Short-Term)" if holding_period == 'short' else "≥ 1 Year (Long-Term)"
                            
                            holdings.append({
                                "ticker": ticker,
                                "basis": float(purchase_price),
                                "shares": float(shares),
                                "holding": holding
                            })
                            
                        if invalid_rows:
                            st.sidebar.warning(f"Skipped {len(invalid_rows)} invalid rows (rows: {', '.join(map(str, invalid_rows[:5]))}{'...' if len(invalid_rows) > 5 else ''})")
                            
                        if holdings:
                            st.sidebar.success(f"Loaded {len(holdings)} valid holdings from CSV.")
                            st.sidebar.dataframe(pd.DataFrame(holdings))
                except Exception as e:
                    st.sidebar.error(f"Error parsing CSV: {e}")

        elif input_method == "Manual Entry":
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

        run_tlh = st.sidebar.button("🔍 Analyze Tax Impact", key="tlh_run")

    st.header("🧾 Tax-Loss Harvesting Simulator")
    st.write("Simulate offsetting capital gains by selling assets currently at a loss.")

    if active_tool == "🧾 Tax-Loss Harvesting Simulator" and 'run_tlh' in locals() and run_tlh and len(holdings) > 0:
        with st.spinner("Fetching current prices..."):
            all_holdings_with_prices = []
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
                all_holdings_with_prices.append(h_new)

        if len(all_holdings_with_prices) > 0:
            st.subheader("Your Holdings Summary")
            summary_df = pd.DataFrame(all_holdings_with_prices)
            summary_df = summary_df[["ticker", "basis", "current_price", "shares", "unrealized_pl", "status"]]
            summary_df.columns = ["Ticker", "Purchase Price", "Current Price", "Shares", "Unrealized P&L", "Status"]
            
            st.dataframe(summary_df.style.format({
                "Purchase Price": "${:,.2f}", 
                "Current Price": "${:,.2f}", 
                "Unrealized P&L": "${:,.2f}"
            }))

            gain_only_holdings = [h for h in all_holdings_with_prices if h["unrealized_pl"] > 0]
            loss_holdings = [h for h in all_holdings_with_prices if h["unrealized_pl"] < 0]

            if not gain_only_holdings:
                st.info("None of your holdings are currently at a gain. No capital gains tax to analyze.")
            elif not loss_holdings:
                st.info("None of your holdings are at a loss. No harvesting opportunity — both scenarios produce identical results.")

            # Compute Scenarios
            scenario_a = compute_portfolio_tax(gain_only_holdings, federal_marginal, ltcg_rate, state_rate, niit_rate)
            scenario_b = compute_portfolio_tax(all_holdings_with_prices, federal_marginal, ltcg_rate, state_rate, niit_rate)

            st.subheader("📊 IRS Netting Waterfall")
            
            waterfall_data = {
                "Row Label": [
                    "Short-Term Gains",
                    "Short-Term Losses",
                    "Net Short-Term",
                    "Long-Term Gains",
                    "Long-Term Losses",
                    "Net Long-Term",
                    "Overall Net Gain/(Loss)",
                    "Tax on Short-Term",
                    "Tax on Long-Term",
                    "Total Capital Gains Tax",
                    "$3,000 Ordinary Income Deduction",
                    "Tax Benefit of Deduction",
                    "Loss Carryforward to Future Years"
                ],
                "Sell Gains Only": [
                    f"${scenario_a['st_gains']:,.2f}",
                    f"${scenario_a['st_losses']:,.2f}",
                    f"${scenario_a['net_st']:,.2f}",
                    f"${scenario_a['lt_gains']:,.2f}",
                    f"${scenario_a['lt_losses']:,.2f}",
                    f"${scenario_a['net_lt']:,.2f}",
                    f"${scenario_a['overall_net']:,.2f}",
                    f"${scenario_a['tax_on_st']:,.2f}",
                    f"${scenario_a['tax_on_lt']:,.2f}",
                    f"${scenario_a['total_tax']:,.2f}",
                    f"${scenario_a['ordinary_income_deduction']:,.2f}",
                    f"${scenario_a['ordinary_income_tax_benefit']:,.2f}",
                    f"${scenario_a['carryforward']:,.2f}"
                ],
                "Sell All (Harvest Losses)": [
                    f"${scenario_b['st_gains']:,.2f}",
                    f"${scenario_b['st_losses']:,.2f}",
                    f"${scenario_b['net_st']:,.2f}",
                    f"${scenario_b['lt_gains']:,.2f}",
                    f"${scenario_b['lt_losses']:,.2f}",
                    f"${scenario_b['net_lt']:,.2f}",
                    f"${scenario_b['overall_net']:,.2f}",
                    f"${scenario_b['tax_on_st']:,.2f}",
                    f"${scenario_b['tax_on_lt']:,.2f}",
                    f"${scenario_b['total_tax']:,.2f}",
                    f"${scenario_b['ordinary_income_deduction']:,.2f}",
                    f"${scenario_b['ordinary_income_tax_benefit']:,.2f}",
                    f"${scenario_b['carryforward']:,.2f}"
                ]
            }
            st.table(pd.DataFrame(waterfall_data).set_index("Row Label"))

            m1, m2, m3 = st.columns(3)
            
            total_savings = (scenario_a["total_tax"] - scenario_b["total_tax"]) + scenario_b["ordinary_income_tax_benefit"]

            total_gains_a = scenario_a["st_gains"] + scenario_a["lt_gains"]
            eff_rate_a = (scenario_a["total_tax"] / total_gains_a * 100) if total_gains_a > 0 else 0.0
            
            total_gains_b = scenario_b["st_gains"] + scenario_b["lt_gains"]
            eff_rate_b = (scenario_b["total_tax"] / total_gains_b * 100) if total_gains_b > 0 else 0.0

            total_sale_proceeds = sum(h["current_price"] * h["shares"] for h in all_holdings_with_prices)
            net_cash = total_sale_proceeds - scenario_b["total_tax"]

            m1.metric("💰 Total Tax Savings", f"${total_savings:,.2f}")
            m2.metric("📉 Effective Tax Rate", f"{eff_rate_a:.1f}%", f"{eff_rate_b - eff_rate_a:.1f}% vs Gains Only", delta_color="inverse")
            m3.metric("📊 Net Cash After Tax", f"${net_cash:,.2f}")

            if scenario_b["carryforward"] > 0:
                st.info(f"📋 You have **${scenario_b['carryforward']:,.2f}** in capital losses that exceed your gains plus the $3,000 deduction. This amount carries forward to offset capital gains in future tax years.")

            st.subheader("Tax Impact Comparison")
            
            labels = ["Total Tax"]
            tax_a = [scenario_a["total_tax"]]
            tax_b = [scenario_b["total_tax"]]

            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(8, 5))
            rects1 = ax.bar(x - width/2, tax_a, width, label='Sell Gains Only', color='indianred')
            rects2 = ax.bar(x + width/2, tax_b, width, label='Sell All + Harvest', color='mediumseagreen')

            ax.set_ylabel('Total Capital Gains Tax ($)')
            ax.set_title('Total Tax: Gains Only vs. With Harvesting')
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