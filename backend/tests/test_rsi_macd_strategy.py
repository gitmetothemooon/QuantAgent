"""Temporary manual check for RSIMACDStrategy.generate_signal()."""

import pandas as pd

from backend.app.strategy.rsi_macd import RSIMACDStrategy
from backend.app.strategy.signals import Signal

strategy = RSIMACDStrategy()

# ------------------------------------------------------------
# Scenario 1: BUY
# Uptrend (Close > SMA50), oversold (RSI14 < 30), and a bullish
# MACD crossover (MACD was <= signal yesterday, > signal today).
# ------------------------------------------------------------
buy_previous_row = pd.Series(
    {
        "Close": 108.0,
        "SMA50": 100.0,
        "RSI14": 22.0,
        "MACD": 0.5,
        "MACD_Signal": 0.6,
    }
)
buy_current_row = pd.Series(
    {
        "Close": 110.0,
        "SMA50": 100.0,
        "RSI14": 25.0,
        "MACD": 0.7,
        "MACD_Signal": 0.6,
    }
)

buy_signal = strategy.generate_signal(buy_current_row, buy_previous_row)

print("BUY Scenario:")
print(buy_signal)

if buy_signal is not Signal.BUY:
    raise ValueError(f"Expected Signal.BUY but got {buy_signal}.")

# ------------------------------------------------------------
# Scenario 2: SELL
# Downtrend (Close < SMA50), overbought (RSI14 > 70), and a bearish
# MACD crossover (MACD was >= signal yesterday, < signal today).
# ------------------------------------------------------------
sell_previous_row = pd.Series(
    {
        "Close": 92.0,
        "SMA50": 100.0,
        "RSI14": 78.0,
        "MACD": 0.6,
        "MACD_Signal": 0.5,
    }
)
sell_current_row = pd.Series(
    {
        "Close": 90.0,
        "SMA50": 100.0,
        "RSI14": 75.0,
        "MACD": 0.4,
        "MACD_Signal": 0.5,
    }
)

sell_signal = strategy.generate_signal(sell_current_row, sell_previous_row)

print("SELL Scenario:")
print(sell_signal)

if sell_signal is not Signal.SELL:
    raise ValueError(f"Expected Signal.SELL but got {sell_signal}.")

# ------------------------------------------------------------
# Scenario 3: HOLD
# Price is in an uptrend, but RSI14 is neutral (not oversold) and
# there is no MACD crossover, so none of the BUY or SELL conditions
# are satisfied.
# ------------------------------------------------------------
hold_previous_row = pd.Series(
    {
        "Close": 109.0,
        "SMA50": 100.0,
        "RSI14": 50.0,
        "MACD": 0.5,
        "MACD_Signal": 0.5,
    }
)
hold_current_row = pd.Series(
    {
        "Close": 110.0,
        "SMA50": 100.0,
        "RSI14": 50.0,
        "MACD": 0.5,
        "MACD_Signal": 0.5,
    }
)

hold_signal = strategy.generate_signal(hold_current_row, hold_previous_row)

print("HOLD Scenario:")
print(hold_signal)

if hold_signal is not Signal.HOLD:
    raise ValueError(f"Expected Signal.HOLD but got {hold_signal}.")

print("All RSIMACDStrategy tests passed.")