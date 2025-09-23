# KRX ETF Portfolio Optimizer

A Python-based tool to **optimize and rebalance an ETF portfolio listed on the Korea Exchange (KRX)**.  
It combines [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/) for modern portfolio optimization with [pykrx](https://github.com/sharebook-kr/pykrx) to fetch live ETF data.

## ✨ Features
- **Historical data fetch**: Automatically downloads ETF price history from KRX.
- **Mean-variance optimization**: Computes optimal weights using Sharpe ratio maximization.
- **Semi-automatic rebalancing**:
  - *Full rebalance*: Triggered if asset drift > configurable threshold (e.g., 5%).
  - *Cash-flow rebalance*: Monthly saving is invested into underweight assets.
- **Transaction cost adjustment**: Accounts for brokerage fees/taxes (default 0.3%).
- **Final summary**: Prints expected annual return, volatility, and recommended trades.

## ⚙️ Configuration
- `CURRENT_HOLDINGS`: Your current ETF holdings (ticker → shares).
- `MONTHLY_SAVING`: New monthly cash to invest (KRW).
- `REBALANCE_THRESHOLD`: % drift that triggers full rebalance.
- `TICKERS`: List of candidate ETFs to include.
- `OPTIMIZATION_FREQUENCY_DAYS`: How often to re-run optimization.

## 🚀 Usage
```bash
pip install -r requirements.txt
python portfolio_opt.py
