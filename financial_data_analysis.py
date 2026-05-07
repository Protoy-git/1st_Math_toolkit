"""
Financial Data Analysis — Python & NumPy
==========================================
Original implementation by Protoy Chowdhury
github.com/Protoy-git

Analyzes financial time-series data using core CS + data concepts:
- Returns calculation
- Moving averages (Simple & Exponential)
- Volatility measurement
- Sharpe Ratio (risk-adjusted return)
- Drawdown analysis
- Correlation between assets

No external financial libraries — built with NumPy only.
Demonstrates Python numerical validation for financial applications.
"""

import numpy as np


# ─── RETURNS ──────────────────────────────────────────────────

def daily_returns(prices):
    
    prices = np.array(prices, dtype=float)
    return (prices[1:] - prices[:-1]) / prices[:-1]


def cumulative_return(prices):
    
    prices = np.array(prices, dtype=float)
    return (prices[-1] / prices[0]) - 1


def annualized_return(prices, trading_days=252):
    
    returns = daily_returns(prices)
    mean_daily = np.mean(returns)
    return (1 + mean_daily) ** trading_days - 1


# ─── MOVING AVERAGES ──────────────────────────────────────────

def simple_moving_average(prices, window):
    
    prices = np.array(prices, dtype=float)
    sma = []
    for i in range(window - 1, len(prices)):
        sma.append(np.mean(prices[i - window + 1 : i + 1]))
    return np.array(sma)


def exponential_moving_average(prices, span):
    
    prices = np.array(prices, dtype=float)
    alpha = 2 / (span + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    return np.array(ema)


# ─── RISK METRICS ─────────────────────────────────────────────

def volatility(prices, trading_days=252):
    
    returns = daily_returns(prices)
    daily_vol = np.std(returns, ddof=1)
    return daily_vol * np.sqrt(trading_days)


def sharpe_ratio(prices, risk_free_rate=0.05, trading_days=252):
    
    ann_return = annualized_return(prices, trading_days)
    ann_vol    = volatility(prices, trading_days)
    if ann_vol == 0:
        return 0
    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(prices):
   
    prices = np.array(prices, dtype=float)
    peak   = prices[0]
    max_dd = 0.0
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (price - peak) / peak
        if drawdown < max_dd:
            max_dd = drawdown
    return max_dd


# ─── CORRELATION ──────────────────────────────────────────────

def asset_correlation(prices_a, prices_b):
    
    ret_a = daily_returns(prices_a)
    ret_b = daily_returns(prices_b)
    return np.corrcoef(ret_a, ret_b)[0][1]


# ─── FULL REPORT ──────────────────────────────────────────────

def financial_report(prices, asset_name="Asset", risk_free=0.05):
   
    returns = daily_returns(prices)

    print(f"\n{'=' * 52}")
    print(f"  FINANCIAL ANALYSIS REPORT — {asset_name}")
    print(f"{'=' * 52}")
    print(f"  Data points         : {len(prices)} days")
    print(f"  Starting price      : ${prices[0]:.2f}")
    print(f"  Ending price        : ${prices[-1]:.2f}")
    print(f"  Cumulative return   : {cumulative_return(prices)*100:.2f}%")
    print(f"  Annualized return   : {annualized_return(prices)*100:.2f}%")
    print(f"  Annualized volatility: {volatility(prices)*100:.2f}%")
    print(f"  Sharpe ratio        : {sharpe_ratio(prices, risk_free):.4f}")
    print(f"  Maximum drawdown    : {max_drawdown(prices)*100:.2f}%")
    print(f"  Mean daily return   : {np.mean(returns)*100:.4f}%")
    print(f"  Best day            : +{np.max(returns)*100:.2f}%")
    print(f"  Worst day           : {np.min(returns)*100:.2f}%")

    sma_20 = simple_moving_average(prices, 20)
    ema_20 = exponential_moving_average(prices, 20)
    print(f"\n  20-day SMA (latest) : ${sma_20[-1]:.2f}")
    print(f"  20-day EMA (latest) : ${ema_20[-1]:.2f}")
    current = prices[-1]
    trend = "ABOVE" if current > sma_20[-1] else "BELOW"
    print(f"  Current price is {trend} 20-day SMA → {'Bullish signal' if trend == 'ABOVE' else 'Bearish signal'}")
    print(f"{'=' * 52}")


# ─── DEMO ─────────────────────────────────────────────────────

def demo():
    print("\n" + "=" * 52)
    print("  FINANCIAL DATA ANALYSIS — Python & NumPy")
    print("  Protoy Chowdhury — github.com/Protoy-git")
    print("=" * 52)

    
    np.random.seed(42)
    days = 252  

    
    returns_a = np.random.normal(0.0005, 0.015, days)
    stock_a   = [100.0]
    for r in returns_a:
        stock_a.append(stock_a[-1] * (1 + r))

    
    returns_b = np.random.normal(0.0003, 0.008, days)
    stock_b   = [100.0]
    for r in returns_b:
        stock_b.append(stock_b[-1] * (1 + r))

    financial_report(stock_a, "Stock A (High Growth)", risk_free=0.05)
    financial_report(stock_b, "Stock B (Conservative)", risk_free=0.05)

    
    corr = asset_correlation(stock_a, stock_b)
    print(f"\n{'=' * 52}")
    print(f"  PORTFOLIO CORRELATION ANALYSIS")
    print(f"{'=' * 52}")
    print(f"  Correlation (A vs B) : {corr:.4f}")
    if abs(corr) < 0.3:
        interpretation = "Low correlation — good for diversification"
    elif abs(corr) < 0.7:
        interpretation = "Moderate correlation — partial diversification benefit"
    else:
        interpretation = "High correlation — limited diversification benefit"
    print(f"  Interpretation       : {interpretation}")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    demo()
