import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from datetime import datetime
import warnings
import yfinance as yf

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ===== 1. Config =====
MONTHLY_SAVING = 1000  # in USD
TRANSACTION_FEE_PERCENTAGE = 0.003  # 0.3%

TICKERS = [
    'NVDA', 'MSFT', 'AAPL', 'GOOG', 'AMZN', 'META', 'AVGO', 'TSLA',
    'BRK-B', 'ORCL', 'JPM', 'WMT', 'LLY', 'V', 'MA', 'NFLX', 'XOM',
    'PLTR', 'JNJ', 'COST', 'HD', 'ABBV', 'BAC', 'PG', 'CVX', 'GE',
    'UNH', 'KO', 'WFC', 'CSCO', 'TMUS', 'AMD', 'MS', 'IBM', 'PM',
    'GS', 'ABT', 'CRM', 'AXP', 'BX', 'LIN', 'CAT', 'APP', 'MCD',
    'RTX', 'UBER', 'T', 'SHOP', 'DIS', 'MRK'
]
START_DATE = "2021-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# ===== 2. Fetch Data =====
print(f"📥 Downloading data from {START_DATE} to {END_DATE}...")
raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)

# FIX: Use only closing prices
if isinstance(raw_data.columns, pd.MultiIndex):
    price_data = raw_data['Close']
else:
    price_data = raw_data

price_data = price_data.dropna(axis=1, how='any')

# ===== 3. Optimize Portfolio =====
mu = expected_returns.mean_historical_return(price_data)
S = risk_models.CovarianceShrinkage(price_data).ledoit_wolf()
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("\n🎯 Optimal Portfolio Allocation:")
for ticker, weight in cleaned_weights.items():
    if weight > 0:
        print(f" - {ticker}: {weight:.2%}")

ef.portfolio_performance(verbose=True)

# ===== 4. Continuous Investment Allocation =====
print("\n💸 Allocating Monthly Investment:")

investable_amount = MONTHLY_SAVING * (1 - TRANSACTION_FEE_PERCENTAGE)
latest_prices = price_data.iloc[-1]
allocation = {
    ticker: {
        "Weight": weight,
        "Allocated ($)": investable_amount * weight,
        "Price": latest_prices[ticker],
        "Expected Shares": investable_amount * weight / latest_prices[ticker]
    }
    for ticker, weight in cleaned_weights.items() if weight > 0
}

df_allocation = pd.DataFrame(allocation).T
print(df_allocation.round(2))

# ===== 5. Optional: Pie Chart =====
df_allocation["Allocated ($)"].plot.pie(
    autopct="%1.1f%%", figsize=(8, 8), title="Portfolio Allocation (USD)"
)
plt.ylabel("")
plt.tight_layout()
plt.show()
