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
    "AVGO": 0.001999,
    "ABBV": 0.003198,
    "XOM": 0.012353,
    "NVDA": 0.003565,
    "LLY": 0.001754,
    "GE": 0.004621,
    "PGR": 0.002836,
    "PM": 0.008731
}


# 💵 Provide new allocation directly in USD
NEW_CASH_USD = 10  # Example: 10 USD newly allocated
TRANSACTION_FEE_PERCENTAGE = 0.001  # 0.1%

EXCHANGE_RATE_BASE = 1396.25
SPREAD = 15
DISCOUNT_RATE = 0.95
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
    price_data = raw_data['Close'] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
    return price_data.dropna(axis=1, how='any')


def get_or_create_target_weights(tickers_for_opt, weights_file, start_date, end_date):
    run_optimization = True
    if os.path.exists(weights_file):
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(weights_file))
        if (datetime.now() - last_modified_time) < timedelta(days=OPTIMIZATION_FREQUENCY_DAYS):
            print(f"\n✅ Loading target weights from '{weights_file}' (less than {OPTIMIZATION_FREQUENCY_DAYS} days old).")
            weights_df = pd.read_csv(weights_file, index_col=0)
            return weights_df.iloc[:, 0].to_dict()
        else:
            print(f"\n⚠️ Target weights file is outdated. Re-running optimization.")

    print("\n🚀 Running new portfolio optimization...")
    df = fetch_etf_data(tickers_for_opt, start_date, end_date)
    if df.empty:
        print("\n❌ CRITICAL ERROR: Could not fetch any data.")
        return None
    valid_tickers = [t for t in df.columns if (df[t].last_valid_index() - df[t].first_valid_index()).days / 365 >= MIN_DATA_YEARS]
    df_filtered = df[valid_tickers]
    if df_filtered.empty:
        print("\n❌ CRITICAL ERROR: No valid tickers met data length criteria.")
        return None

    mu = expected_returns.mean_historical_return(df_filtered)
    S = risk_models.CovarianceShrinkage(df_filtered).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    target_weights = ef.clean_weights()
    pd.DataFrame.from_dict(target_weights, orient='index', columns=['weight']).to_csv(weights_file)
    print(f"   📎 New target weights saved to '{weights_file}'.")
    ef.portfolio_performance(verbose=True)
    return target_weights


def _get_renormalized_weights(weights, available_tickers):
    filtered = {t: w for t, w in weights.items() if t in available_tickers}
    total = sum(filtered.values())
    return {t: w / total for t, w in filtered.items()} if total > 0 else {}


def print_recommended_action_table(final_holdings, current_holdings, latest_prices, exchange_rate):
    """Prints buy/sell recommendations in rounded ₩ units."""
    actions = []
    for ticker in sorted(set(current_holdings.keys()) | set(final_holdings.keys())):
        current_shares = current_holdings.get(ticker, 0)
        target_shares = final_holdings.get(ticker, 0)
        trade = target_shares - current_shares
        price_usd = latest_prices.get(ticker, 0)
        krw_value = trade * price_usd * exchange_rate
        krw_rounded = int(round(krw_value / 1000) * 1000)
        if abs(krw_rounded) >= 1000:
            actions.append({
                "Ticker": ticker,
                "Action": "BUY" if trade > 0 else "SELL",
                "Shares Δ": f"{trade:+.6f}",
                "Approx. Amount": f"{krw_rounded:+,}원"
            })
    if actions:
        print("\n📊 Recommended Action Summary:")
        print(pd.DataFrame(actions).to_string(index=False))
    else:
        print("\nNo significant buy/sell actions.")


# ================== 3. Main Execution ==================
if __name__ == "__main__":
    all_relevant_tickers = list(set(TICKERS) | set(CURRENT_HOLDINGS.keys()))
    target_weights = get_or_create_target_weights(all_relevant_tickers, WEIGHTS_FILE, START_DATE, END_DATE)
    if target_weights is None:
        exit()

    print("\n" + "="*50)
    print("ANALYZING CURRENT PORTFOLIO".center(50))
    print("="*50)
    owned_tickers = [t for t, s in CURRENT_HOLDINGS.items() if s > 0]
    tickers_to_price = list(set(owned_tickers) | set(target_weights.keys()))
    prices_df = fetch_etf_data(tickers_to_price, START_DATE, END_DATE)
    latest_prices = get_latest_prices(prices_df)

    current_values = {t: latest_prices.get(t, 0) * CURRENT_HOLDINGS.get(t, 0) for t in owned_tickers}
    current_value_usd = sum(current_values.values())
    print(f"Current Portfolio Value: ${current_value_usd:,.2f} (≈ ₩{current_value_usd*EXCHANGE_RATE_KRW_TO_USD:,.0f})")

    current_weights = {t: (current_values[t] / current_value_usd if current_value_usd > 0 else 0) for t in owned_tickers}

    # Detect drift
    needs_full_rebalance = False
    for t in target_weights:
        if abs(current_weights.get(t, 0) - target_weights[t]) > REBALANCE_THRESHOLD:
            needs_full_rebalance = True

    print("\n" + "="*50)
    print("RECOMMENDED ACTION".center(50))
    print("="*50)

    final_holdings = CURRENT_HOLDINGS.copy()

    if needs_full_rebalance:
        print("‼️ THRESHOLD EXCEEDED → Full rebalance recommended.")
        total_value = (current_value_usd + NEW_CASH_USD) * (1 - TRANSACTION_FEE_PERCENTAGE)
        weights = _get_renormalized_weights(target_weights, latest_prices.index)
        final_holdings = {t: (total_value * w / latest_prices[t]) for t, w in weights.items()}
        print_recommended_action_table(final_holdings, CURRENT_HOLDINGS, latest_prices, EXCHANGE_RATE_KRW_TO_USD)
    else:
        print("✅ Within threshold → Cash-Flow Rebalancing only.")
        investable = NEW_CASH_USD * (1 - TRANSACTION_FEE_PERCENTAGE)
        weights = _get_renormalized_weights(target_weights, latest_prices.index)
        add_shares = {t: investable * w / latest_prices[t] for t, w in weights.items()}
        final_holdings = {t: CURRENT_HOLDINGS.get(t, 0) + add_shares.get(t, 0) for t in weights}
        print_recommended_action_table(final_holdings, CURRENT_HOLDINGS, latest_prices, EXCHANGE_RATE_KRW_TO_USD)


# ================== 4. Final Summary Section ==================
print("\n" + "="*50)
print("FINAL PORTFOLIO SUMMARY".center(50))
print("="*50)

# Only include holdings > 0
filtered_holdings = {t: s for t, s in final_holdings.items() if s > 0}
if not filtered_holdings:
    print("No holdings in the final portfolio.")
else:
    summary_data = []
    for t, s in sorted(filtered_holdings.items()):
        val_usd = s * latest_prices.get(t, 0)
        val_krw = val_usd * EXCHANGE_RATE_KRW_TO_USD
        summary_data.append({
            "Ticker": t,
            "Shares": round(s, 6),
            "Value (USD)": f"{val_usd:,.2f}",
            "Value (KRW)": f"{val_krw:,.0f}"
        })
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # Pie chart visualization
    values_usd = [float(v.replace(",", "")) for v in df_summary["Value (USD)"]]
    labels = df_summary["Ticker"]
    plt.figure(figsize=(7, 7))
    plt.pie(values_usd, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Final Portfolio Allocation (by USD value)")
    plt.tight_layout()
    plt.show()
