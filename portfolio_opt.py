import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from datetime import datetime, timedelta
import warnings
import os
from pykrx import stock

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# ================== 1. Configuration Section ==================
# --- Define your current portfolio and new investment ---
# IMPORTANT: Update this dictionary with your actual holdings.
CURRENT_HOLDINGS = {
    "200250.KS": 2,
    "245710.KS": 6,
    "314250.KS": 9,
    "360200.KS": 21,
    "379800.KS": 156,
    "390390.KS": 7
}
MONTHLY_SAVING = 450457  # New cash to invest this month (in KRW)

# --- Define your rebalancing rules ---
REBALANCE_THRESHOLD = 0.05  # Trigger a full rebalance if any asset drifts by more than 5% (0.05)
OPTIMIZATION_FREQUENCY_DAYS = 180  # Re-run optimization semi-annually (180) or annually (365)
WEIGHTS_FILE = "target_weights.csv" # File to store the optimal weights
TRANSACTION_FEE_PERCENTAGE = 0.003 # Estimate of 0.3% for brokerage fees and taxes

# --- Ticker list and historical data parameters ---
TICKERS = [
    "360750.KS", "133690.KS", "381170.KS", "381180.KS", "379800.KS",
    "371460.KS", "360200.KS", "379810.KS", "367380.KS", "314250.KS",
    "371160.KS", "390390.KS", "402970.KS", "195930.KS", "241180.KS",
    "394660.KS", "200250.KS", "245710.KS", "143850.KS", "372330.KS",
    "419170.KS" # Example of a potentially delisted ticker
]
START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
MIN_DATA_YEARS = 3

# ================== 2. Helper Functions ==================

def fetch_etf_data(tickers, start_date_str, end_date_str):
    """Fetch historical ETF data using the pykrx library."""
    print(f"Fetching data for {len(tickers)} tickers from {start_date_str} to {end_date_str}...")
    price_data = {}
    start_date_fmt = datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%Y%m%d")
    end_date_fmt = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%Y%m%d")

    for ticker in tickers:
        try:
            ticker_code = ticker.replace(".KS", "")
            df = stock.get_etf_ohlcv_by_date(start_date_fmt, end_date_fmt, ticker_code)
            if not df.empty:
                price_data[ticker] = df['종가'].rename(ticker)
                print(f"✓ Successfully fetched {ticker}")
            else:
                print(f"✗ No data for {ticker}")
        except Exception as e:
            print(f"✗ Error fetching {ticker}: {str(e)}")

    if not price_data:
        return pd.DataFrame()
    return pd.DataFrame(price_data).ffill()


def get_or_create_target_weights(tickers_for_opt, weights_file, start_date, end_date):
    """Loads target weights if file is recent, otherwise runs optimization."""
    run_optimization = True
    if os.path.exists(weights_file):
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(weights_file))
        if (datetime.now() - last_modified_time) < timedelta(days=OPTIMIZATION_FREQUENCY_DAYS):
            print(f"\n✅ Loading target weights from '{weights_file}' (less than {OPTIMIZATION_FREQUENCY_DAYS} days old).")
            weights_df = pd.read_csv(weights_file, index_col=0)
            target_weights = weights_df.iloc[:, 0].to_dict()
            run_optimization = False
        else:
            print(f"\n⚠️ Target weights file is outdated. Re-running optimization.")

    if run_optimization:
        print("\n🚀 Running new portfolio optimization...")
        df = fetch_etf_data(tickers_for_opt, start_date, end_date)
        if df.empty:
            print("\n❌ CRITICAL ERROR: Could not fetch any data. Halting optimization.")
            return None
        valid_tickers = [t for t in df.columns if (df[t].last_valid_index() - df[t].first_valid_index()).days / 365 >= MIN_DATA_YEARS]
        df_filtered = df[valid_tickers]
        print(f"   {len(df_filtered.columns)} of {len(df.columns)} ETFs with ≥{MIN_DATA_YEARS} years data retained for optimization.")
        if df_filtered.empty:
            print("\n❌ CRITICAL ERROR: No ETFs met the minimum data requirement. Halting optimization.")
            return None
        mu = expected_returns.mean_historical_return(df_filtered)
        S = risk_models.CovarianceShrinkage(df_filtered).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        target_weights = ef.clean_weights()
        pd.DataFrame.from_dict(target_weights, orient='index', columns=['weight']).to_csv(weights_file)
        print(f"   💾 New target weights saved to '{weights_file}'.")
        print("\nOptimized Portfolio Performance:")
        ef.portfolio_performance(verbose=True)
    return target_weights

def _get_renormalized_weights(weights, available_tickers):
    """Filters weights for available tickers and renormalizes them to sum to 1."""
    filtered_weights = {}
    for ticker, weight in weights.items():
        if ticker in available_tickers:
            filtered_weights[ticker] = weight
        else:
            print(f"⚠️ WARNING: Ticker '{ticker}' has no price data and will be excluded from this transaction.")
    
    weight_sum = sum(filtered_weights.values())
    if weight_sum <= 0:
        print(f"❌ ERROR: No valid tickers with positive weights left after filtering.")
        return {}

    return {ticker: weight / weight_sum for ticker, weight in filtered_weights.items()}

# ================== 3. Main Execution ==================
if __name__ == "__main__":
    all_relevant_tickers = list(set(TICKERS) | set(CURRENT_HOLDINGS.keys()))
    target_weights = get_or_create_target_weights(all_relevant_tickers, WEIGHTS_FILE, START_DATE, END_DATE)

    if target_weights is None:
        print("\nCould not generate target weights. Exiting.")
        exit()

    print("\n" + "="*50); print("ANALYZING CURRENT PORTFOLIO".center(50)); print("="*50)
    owned_tickers = [ticker for ticker, shares in CURRENT_HOLDINGS.items() if shares > 0]
    tickers_to_price = list(set(owned_tickers) | set(target_weights.keys()))
    prices_df = fetch_etf_data(tickers_to_price, START_DATE, END_DATE)
    if prices_df.empty:
        print("Could not fetch any price data. Exiting.")
        exit()
    latest_prices = get_latest_prices(prices_df)

    current_values = {ticker: latest_prices.get(ticker, 0) * CURRENT_HOLDINGS.get(ticker, 0) for ticker in owned_tickers}
    current_portfolio_value = sum(current_values.values())
    print(f"Current Portfolio Value: {current_portfolio_value:,.0f} KRW")

    current_weights = {ticker: 0 for ticker in tickers_to_price}
    if current_portfolio_value > 0:
        for ticker, value in current_values.items():
            if ticker in current_weights:
                current_weights[ticker] = value / current_portfolio_value

    print("\n" + "="*50); print("CHECKING REBALANCE TRIGGER".center(50)); print("="*50)
    drift_data, needs_full_rebalance = [], False
    for ticker in sorted(list(set(current_weights.keys()) | set(target_weights.keys()))):
        target = target_weights.get(ticker, 0)
        current = current_weights.get(ticker, 0)
        drift = current - target
        if drift != 0 or target != 0:
            drift_data.append({"Ticker": ticker, "Target": f"{target:.1%}", "Current": f"{current:.1%}", "Drift": f"{drift:.1%}"})
        if abs(drift) > REBALANCE_THRESHOLD: needs_full_rebalance = True
    print(pd.DataFrame(drift_data).to_string(index=False))
    
    print("\n" + "="*50); print("RECOMMENDED ACTION".center(50)); print("="*50)

    final_holdings = {}
    if needs_full_rebalance:
        print(f"‼️ THRESHOLD EXCEEDED! A full rebalance is recommended.")
        value_before_fees = current_portfolio_value + MONTHLY_SAVING
        total_value_for_rebalance = value_before_fees * (1 - TRANSACTION_FEE_PERCENTAGE)
        print(f"   Rebalancing based on a total value of: {total_value_for_rebalance:,.0f} KRW (after {TRANSACTION_FEE_PERCENTAGE:.1%} fee estimate)")
        
        renormalized_weights = _get_renormalized_weights(target_weights, latest_prices.index)
        
        if renormalized_weights:
            da = DiscreteAllocation(renormalized_weights, latest_prices, total_portfolio_value=total_value_for_rebalance)
            target_allocation, leftover = da.greedy_portfolio()
            final_holdings = target_allocation.copy()
            
            print("\n--- Recommended Trades to Rebalance Portfolio ---")
            all_trade_tickers = set(CURRENT_HOLDINGS.keys()) | set(target_allocation.keys())
            for ticker in sorted(list(all_trade_tickers)):
                current_shares = CURRENT_HOLDINGS.get(ticker, 0)
                target_shares = target_allocation.get(ticker, 0)
                trade = target_shares - current_shares
                if trade < 0: print(f"🔴 {ticker}: SELL {abs(trade):>4} shares. (From {current_shares} down to {target_shares})")
                elif trade > 0: print(f"🟢 {ticker}: BUY  {abs(trade):>4} shares. (From {current_shares} up to {target_shares})")
            print(f"\nFunds remaining after all trades: {leftover:,.0f} KRW")
        else:
            final_holdings = CURRENT_HOLDINGS.copy()

    else:
        print(f"✅ Drift is within the {REBALANCE_THRESHOLD:.0%} threshold. Performing Cash-Flow Rebalancing.")
        cash_to_invest = MONTHLY_SAVING * (1 - TRANSACTION_FEE_PERCENTAGE)
        print(f"   Investing new cash ({cash_to_invest:,.0f} KRW) to correct underweight assets.")

        shortfall = {}
        final_portfolio_value = current_portfolio_value + MONTHLY_SAVING
        for ticker in target_weights:
            target_value = final_portfolio_value * target_weights.get(ticker, 0)
            current_value = current_portfolio_value * current_weights.get(ticker, 0)
            shortfall[ticker] = max(0, target_value - current_value)
        
        total_shortfall = sum(shortfall.values())
        purchase_weights_raw = target_weights if total_shortfall == 0 else {ticker: s / total_shortfall for ticker, s in shortfall.items()}
        
        renormalized_weights = _get_renormalized_weights(purchase_weights_raw, latest_prices.index)
        
        if renormalized_weights:
            da = DiscreteAllocation(renormalized_weights, latest_prices, total_portfolio_value=cash_to_invest)
            allocation, leftover = da.greedy_portfolio()
            final_holdings = CURRENT_HOLDINGS.copy()
            for ticker, shares in allocation.items():
                final_holdings[ticker] = final_holdings.get(ticker, 0) + shares

            print("\n--- Recommended Buys with New Cash ---")
            if not allocation: print("No purchases recommended.")
            else:
                for ticker, shares in sorted(allocation.items()):
                    value = shares * latest_prices.get(ticker, 0)
                    print(f"🟢 {ticker}: BUY {shares:>4} shares ({value:,.0f} KRW)")
            print(f"\nRemaining Funds from Monthly Saving: {leftover:,.0f} KRW")
        else:
            final_holdings = CURRENT_HOLDINGS.copy()

    # ================== 4. Final Summary Section ==================
    print("\n" + "="*50); print("FINAL PORTFOLIO SUMMARY (POST-REBALANCE)".center(50)); print("="*50)
    
    summary_tickers = list(target_weights.keys())
    # Ensure all tickers in summary_tickers exist in prices_df
    summary_tickers = [t for t in summary_tickers if t in prices_df.columns]
    summary_df = prices_df[summary_tickers]

    if not summary_df.empty:
        mu_summary = expected_returns.mean_historical_return(summary_df)
        S_summary = risk_models.CovarianceShrinkage(summary_df).ledoit_wolf()
        
        renormalized_summary_weights = _get_renormalized_weights(target_weights, summary_df.columns)

        if renormalized_summary_weights:
            ef_summary = EfficientFrontier(mu_summary, S_summary)
            ef_summary.set_weights(renormalized_summary_weights)
            expected_return, annual_volatility, _ = ef_summary.portfolio_performance()

            print("Expected Performance of the Rebalanced Portfolio:")
            print(f"  - Expected Annual Return: {expected_return*100:.1f}%")
            print(f"  - Annual Volatility (Risk): {annual_volatility*100:.1f}%")
        
        print("\nFinal Holdings After All Trades:")
        holdings_data = []
        for ticker, shares in sorted(final_holdings.items()):
            if shares > 0:
                value = shares * latest_prices.get(ticker, 0)
                holdings_data.append({"Ticker": ticker, "Shares": shares, "Value (KRW)": f"{value:,.0f}"})
        if holdings_data:
            print(pd.DataFrame(holdings_data).to_string(index=False))
        else:
            print("No holdings in the final portfolio.")
    else:
        print("Could not generate final summary because no assets are in the target portfolio.")
