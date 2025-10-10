import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import get_latest_prices
from datetime import datetime, timedelta
import warnings
import os
import yfinance as yf

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ================== 1. Configuration Section ==================
CURRENT_HOLDINGS = {
    "ABBV": 0.003198,
    "XOM": 0.006197,
    "LLY": 0.00095,
    "GE": 0.002313,
    "PM": 0.004341
}
MONTHLY_SAVING = 12000  # in KRW
TRANSACTION_FEE_PERCENTAGE = 0.001  # 0.1%
EXCHANGE_RATE_BASE = 1396.25
SPREAD = 15  # KRW
DISCOUNT_RATE = 0.95

# Apply 95% discount on spread
actual_spread = SPREAD * (1 - DISCOUNT_RATE)
EXCHANGE_RATE_KRW_TO_USD = EXCHANGE_RATE_BASE + actual_spread

REBALANCE_THRESHOLD = 0.05
OPTIMIZATION_FREQUENCY_DAYS = 180
WEIGHTS_FILE = "US_stocks_target_weights.csv"

TICKERS = [
    'NVDA', 'MSFT', 'AAPL', 'GOOG', 'AMZN', 'META', 'AVGO', 'TSLA',
    'BRK-B', 'ORCL', 'JPM', 'WMT', 'LLY', 'V', 'MA', 'NFLX', 'XOM',
    'PLTR', 'JNJ', 'COST', 'HD', 'ABBV', 'BAC', 'PG', 'CVX', 'GE',
    'UNH', 'KO', 'WFC', 'CSCO', 'TMUS', 'AMD', 'MS', 'IBM', 'PM',
    'GS', 'ABT', 'CRM', 'AXP', 'BX', 'LIN', 'CAT', 'APP', 'MCD',
    'RTX', 'UBER', 'DIS', 'MRK', 'INTU', 'SHOP', 'PEP', 'NOW', 'C',
    'MU', 'BLK', 'QCOM', 'VZ', 'ANET', 'BKNG', 'TMO', 'GEV', 'SCHW',
    'LRCX', 'TXN', 'BA', 'ISRG', 'AMAT', 'TJX', 'AMGN', 'ADBE', 'APH',
    'SPGI', 'NEE', 'ACN', 'SPOT', 'LOW', 'BSX', 'ETN', 'SYK', 'COF',
    'GILD', 'KLAC', 'PGR', 'PEE', 'INTC', 'PANW', 'DHR', 'UNP', 'HON',
    'KKR', 'DE', 'MELI', 'MDT', 'CRWD', 'ADI', 'BN', 'ADP', 'CMCSA', 'COP'
]

START_DATE = "2021-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
MIN_DATA_YEARS = 2

# ================== 2. Helper Functions ==================
def fetch_etf_data(tickers, start_date, end_date):
    print(f"🗕️ Fetching data for {len(tickers)} tickers...")
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    if isinstance(raw_data.columns, pd.MultiIndex):
        price_data = raw_data['Close']
    else:
        price_data = raw_data
    return price_data.dropna(axis=1, how='any')


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
        ef.max_sharpe()
        target_weights = ef.clean_weights()
        pd.DataFrame.from_dict(target_weights, orient='index', columns=['weight']).to_csv(weights_file)
        print(f"   📎 New target weights saved to '{weights_file}'.")
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


def print_recommended_action_table(final_holdings, current_holdings, latest_prices, exchange_rate):
    """Prints buy/sell recommendations in 1,000 KRW units."""
    actions = []
    for ticker in sorted(set(current_holdings.keys()) | set(final_holdings.keys())):
        current_shares = current_holdings.get(ticker, 0)
        target_shares = final_holdings.get(ticker, 0)
        trade = target_shares - current_shares
        price_usd = latest_prices.get(ticker, 0)
        krw_value = trade * price_usd * exchange_rate
        krw_thousand = round(krw_value / 1000, 1)
        if abs(krw_thousand) >= 1:
            actions.append({
                "Ticker": ticker,
                "Action": "BUY" if trade > 0 else "SELL",
                "Shares Δ": f"{trade:+.6f}",
                "Approx KRW (₩1,000)": f"{krw_thousand:+,.1f}"
            })
    if actions:
        print("\n📊 Recommended Action Summary (₩1,000 units):")
        print(pd.DataFrame(actions).to_string(index=False))
    else:
        print("\nNo buy/sell actions exceeding ₩1,000 were recommended.")


# ================== 3. Main Execution ==================
if __name__ == "__main__":
    all_relevant_tickers = list(set(TICKERS) | set(CURRENT_HOLDINGS.keys()))
    target_weights = get_or_create_target_weights(all_relevant_tickers, WEIGHTS_FILE, START_DATE, END_DATE)
    if target_weights is None:
        print("\nCould not generate target weights. Exiting.")
        exit()

    print("\n" + "="*50)
    print("ANALYZING CURRENT PORTFOLIO".center(50))
    print("="*50)
    owned_tickers = [ticker for ticker, shares in CURRENT_HOLDINGS.items() if shares > 0]
    tickers_to_price = list(set(owned_tickers) | set(target_weights.keys()))
    prices_df = fetch_etf_data(tickers_to_price, START_DATE, END_DATE)
    if prices_df.empty:
        print("Could not fetch any price data. Exiting.")
        exit()
    latest_prices = get_latest_prices(prices_df)

    current_values = {ticker: latest_prices.get(ticker, 0) * CURRENT_HOLDINGS.get(ticker, 0) for ticker in owned_tickers}
    current_portfolio_value = round(sum(current_values.values()), 2)
    current_portfolio_value_krw = current_portfolio_value * EXCHANGE_RATE_KRW_TO_USD
    print(f"Current Portfolio Value: {current_portfolio_value:,.2f} USD ({current_portfolio_value_krw:,.0f} KRW)")

    current_weights = {ticker: 0 for ticker in tickers_to_price}
    if current_portfolio_value > 0:
        for ticker, value in current_values.items():
            if ticker in current_weights:
                current_weights[ticker] = value / current_portfolio_value

    print("\n" + "="*50)
    print("CHECKING REBALANCE TRIGGER".center(50))
    print("="*50)
    drift_data, needs_full_rebalance = [], False
    for ticker in sorted(list(set(current_weights.keys()) | set(target_weights.keys()))):
        target = target_weights.get(ticker, 0)
        current = current_weights.get(ticker, 0)
        drift = current - target
        if drift != 0 or target != 0:
            drift_data.append({"Ticker": ticker, "Target": f"{target:.1%}", "Current": f"{current:.1%}", "Drift": f"{drift:.1%}"})
        if abs(drift) > REBALANCE_THRESHOLD:
            needs_full_rebalance = True
    print(pd.DataFrame(drift_data).to_string(index=False))

    print("\n" + "="*50)
    print("RECOMMENDED ACTION".center(50))
    print("="*50)

    usd_monthly_saving = round(MONTHLY_SAVING / EXCHANGE_RATE_KRW_TO_USD, 2)
    print(f"\n💱 KRW {MONTHLY_SAVING:,} → ${usd_monthly_saving} USD (after exchange)")
    final_holdings = CURRENT_HOLDINGS.copy()

    if needs_full_rebalance:
        print(f"‼️ THRESHOLD EXCEEDED! A full rebalance is recommended.")
        value_before_fees = current_portfolio_value + usd_monthly_saving
        total_value_for_rebalance = value_before_fees * (1 - TRANSACTION_FEE_PERCENTAGE)
        print(f"   Rebalancing based on a total value of: {total_value_for_rebalance:,.2f} USD (after {TRANSACTION_FEE_PERCENTAGE:.1%} fee estimate)")

        renormalized_weights = _get_renormalized_weights(target_weights, latest_prices.index)

        if renormalized_weights:
            final_holdings = {}
            for ticker, weight in renormalized_weights.items():
                amount_krw = total_value_for_rebalance * weight * EXCHANGE_RATE_KRW_TO_USD
                rounded_amount_krw = (amount_krw // 1000) * 1000
                amount_usd = rounded_amount_krw / EXCHANGE_RATE_KRW_TO_USD
                shares = amount_usd / latest_prices[ticker]
                krw_value = shares * latest_prices[ticker] * EXCHANGE_RATE_KRW_TO_USD
                if krw_value >= 1000:
                    final_holdings[ticker] = shares
            print("\n--- Recommended Trades to Rebalance Portfolio ---")
            for ticker in sorted(set(CURRENT_HOLDINGS.keys()) | set(final_holdings.keys())):
                current_shares = CURRENT_HOLDINGS.get(ticker, 0)
                target_shares = final_holdings.get(ticker, 0)
                trade = target_shares - current_shares
                if trade < 0:
                    print(f"🔴 {ticker}: SELL {abs(trade):.6f} shares. (From {current_shares:.6f} → {target_shares:.6f})")
                elif trade > 0:
                    print(f"🟢 {ticker}: BUY  {abs(trade):.6f} shares. (From {current_shares:.6f} → {target_shares:.6f})")

            print_recommended_action_table(final_holdings, CURRENT_HOLDINGS, latest_prices, EXCHANGE_RATE_KRW_TO_USD)

    else:
        print(f"✅ Drift is within the {REBALANCE_THRESHOLD:.0%} threshold. Performing Cash-Flow Rebalancing.")
        cash_to_invest = usd_monthly_saving * (1 - TRANSACTION_FEE_PERCENTAGE)
        print(f"   Investing new cash ({cash_to_invest:,.0f} USD) to correct underweight assets.")

        shortfall = {}
        final_portfolio_value = current_portfolio_value + usd_monthly_saving
        for ticker in target_weights:
            target_value = final_portfolio_value * target_weights.get(ticker, 0)
            current_value = current_portfolio_value * current_weights.get(ticker, 0)
            shortfall[ticker] = max(0, target_value - current_value)

        total_shortfall = sum(shortfall.values())
        purchase_weights_raw = target_weights if total_shortfall == 0 else {ticker: s / total_shortfall for ticker, s in shortfall.items()}
        renormalized_weights = _get_renormalized_weights(purchase_weights_raw, latest_prices.index)

        if renormalized_weights:
            final_holdings = CURRENT_HOLDINGS.copy()
            for ticker, weight in renormalized_weights.items():
                amount_krw = cash_to_invest * weight * EXCHANGE_RATE_KRW_TO_USD
                rounded_amount_krw = (amount_krw // 1000) * 1000
                amount_usd = rounded_amount_krw / EXCHANGE_RATE_KRW_TO_USD
                shares = amount_usd / latest_prices[ticker]
                if shares > 0:
                    final_holdings[ticker] = final_holdings.get(ticker, 0) + shares

            print("\n--- Recommended Buys with New Cash ---")
            for ticker, shares in sorted(final_holdings.items()):
                value = shares * latest_prices.get(ticker, 0)
                print(f"🟢 {ticker}: BUY {shares:.6f} shares ({value:,.0f} USD)")

            print_recommended_action_table(final_holdings, CURRENT_HOLDINGS, latest_prices, EXCHANGE_RATE_KRW_TO_USD)

# ================== 4. Final Summary Section ==================
print("\n" + "="*50)
print("FINAL PORTFOLIO SUMMARY (POST-REBALANCE)".center(50))
print("="*50)

summary_tickers = list(target_weights.keys())
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
            price_usd = latest_prices.get(ticker, 0)
            value_usd = shares * price_usd
            value_krw = value_usd * EXCHANGE_RATE_KRW_TO_USD
            holdings_data.append({
                "Ticker": ticker,
                "Shares": round(shares, 6),
                "Value (KRW)": f"{value_krw:,.0f}"
            })
    if holdings_data:
        print(pd.DataFrame(holdings_data).to_string(index=False))
    else:
        print("No holdings in the final portfolio.")
else:
    print("Could not generate final summary because no assets are in the target portfolio.")
