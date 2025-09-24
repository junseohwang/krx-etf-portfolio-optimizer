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
    "245710.KS": 7,
    "314250.KS": 10,
    "360200.KS": 22,
    "379800.KS": 172,
    "390390.KS": 7
}
MONTHLY_SAVING = 6602  # New cash to invest this month (in KRW)

# --- Define your rebalancing rules ---
REBALANCE_THRESHOLD = 0.05  # Trigger a full rebalance if any asset drifts by more than 5% (0.05)
OPTIMIZATION_FREQUENCY_DAYS = 180  # Re-run optimization semi-annually (180) or annually (365)
WEIGHTS_FILE = "target_weights.csv" # File to store the optimal weights
TRANSACTION_FEE_PERCENTAGE = 0.003 # Estimate of 0.3% for brokerage fees and taxes

# --- Ticker list and historical data parameters ---
TICKERS = [
    "360750.KS", # TIGER 미국S&P500
    "133690.KS", # TIGER 미국나스닥100
    "379800.KS", # KODEX 미국S&P500
    "381170.KS", # TIGER 미국테크TOP10 INDXX
    "379810.KS", # KODEX 미국나스닥100
    "381180.KS", # TIGER 미국필라델피아반도체나스닥
    "360200.KS", # ACE 미국S&P500
    "458730.KS", # TIGER 미국배당다우존스
    "367380.KS", # ACE 미국나스닥100
    "371460.KS", # TIGER 차이나전기차SOLACTIVE
    "371160.KS", # TIGER 차이나항셍테크
    "368590.KS", # RISE 미국나스닥100
    "457480.KS", # ACE 테슬라밸류체인액티브
    "487230.KS", # KODEX 미국AI전력핵심인프라
    "379780.KS", # RISE 미국S&P500
    "458760.KS", # TIGER 미국배당다우존스타겟커버드콜2호
    "486290.KS", # TIGER 미국나스닥100타겟데일리커버드콜
    "465580.KS", # ACE 미국빅테크TOP7 Plus
    "456600.KS", # TIMEFOLIO 글로벌AI인공지능액티브
    "446720.KS", # SOL 미국배당다우존스
    "314250.KS", # KODEX 미국빅테크10(H)
    "449180.KS", # KODEX 미국S&P500(H)
    "441640.KS", # KODEX 미국배당커버드콜액티브
    "402970.KS", # ACE 미국배당다우존스
    "426030.KS", # TIMEFOLIO 미국나스닥100액티브
    "497570.KS", # TIGER 미국필라델피아AI반도체나스닥
    "390390.KS", # KODEX 미국반도체
    "474220.KS", # TIGER 미국테크TOP10타겟커버드콜
    "453870.KS", # TIGER 인도니프티50
    "483280.KS", # KODEX 미국AI테크TOP10타겟커버드콜
    "449190.KS", # KODEX 미국나스닥100(H)
    "453810.KS", # KODEX 인도Nifty50
    "448290.KS", # TIGER 미국S&P500(H)
    "251350.KS", # KODEX MSCI선진국
    "446770.KS", # ACE 글로벌반도체TOP4 Plus SOLACTIVE
    "441680.KS", # TIGER 미국나스닥100커버드콜(합성)
    "494300.KS", # KODEX 미국나스닥100데일리커버드콜OTM
    "489250.KS", # KODEX 미국배당다우존스
    "482730.KS", # TIGER 미국S&P500타겟데일리커버드콜
    "466950.KS", # TIGER 글로벌AI액티브
    "442320.KS", # RISE 글로벌원자력
    "394660.KS", # TIGER 글로벌자율주행&전기차SOLACTIVE
    "245710.KS", # ACE 베트남VN30(합성)
    "241180.KS", # TIGER 일본니케이225
    "448300.KS", # TIGER 미국나스닥100(H)
    "481180.KS", # SOL 미국AI소프트웨어
    "472160.KS", # TIGER 미국테크TOP10 INDXX(H)
    "473460.KS", # KODEX 미국서학개미
    "423920.KS", # TIGER 미국필라델피아반도체레버리지(합성)
    "0060H0.KS", # TIGER 토탈월드스탁액티브
    "394670.KS", # TIGER 글로벌리튬&2차전지SOLACTIVE(합성)
    "491010.KS", # TIGER 글로벌AI전력인프라액티브
    "0051G0.KS", # SOL 미국원자력SMR
    "409820.KS", # KODEX 미국나스닥100레버리지(합성 H)
    "0047A0.KS", # TIGER 차이나테크TOP10
    "486450.KS", # SOL 미국AI전력인프라
    "452360.KS", # SOL 미국배당다우존스(H)
    "481190.KS", # SOL 미국테크TOP10
    "200250.KS", # KIWOOM 인도Nifty50(합성)
    "490590.KS", # RISE 미국AI밸류체인데일리고정커버드콜
    "493810.KS", # TIGER 미국AI빅테크10타겟데일리커버드콜
    "478150.KS", # TIMEFOLIO 글로벌우주테크&방산액티브
    "485540.KS", # KODEX 미국AI테크TOP10
    "418660.KS", # TIGER 미국나스닥100레버리지(합성)
    "192090.KS", # TIGER 차이나CSI300
    "480020.KS", # ACE 미국빅테크7+데일리타겟커버드콜(합성)
    "372330.KS", # KODEX 차이나항셍테크
    "414780.KS", # TIGER 차이나과창판STAR50(합성)
    "283580.KS", # KODEX 차이나CSI300
    "498270.KS" # KIWOOM 미국양자컴퓨팅
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
