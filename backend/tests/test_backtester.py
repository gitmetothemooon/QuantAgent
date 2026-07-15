"""Temporary manual check for Backtester.run()."""

import pandas as pd

from backend.app.backtesting.backtester import Backtester
from backend.app.strategy.base import Strategy
from backend.app.strategy.signals import Signal


class DummyStrategy(Strategy):
    """A strategy that returns a fixed, predetermined sequence of signals.

    Used only to exercise the Backtester with a known, controllable
    order of BUY / SELL / HOLD signals, independent of any real
    indicator values.
    """

    def __init__(self) -> None:
        self._signals = [Signal.BUY, Signal.BUY, Signal.HOLD, Signal.SELL]
        self._call_count = 0

    def generate_signal(
        self,
        current_row: pd.Series,
        previous_row: pd.Series,
    ) -> Signal:
        if self._call_count < len(self._signals):
            signal = self._signals[self._call_count]
        else:
            signal = Signal.HOLD

        self._call_count += 1
        return signal


# The Backtester starts simulating at row index 1, since generating a
# signal requires both a current row and a previous row. With five
# rows (index 0-4), that means exactly four signals are generated,
# corresponding to DummyStrategy's sequence in order:
#
#   Row 0: 100  (no signal generated; used only as "previous row")
#   Row 1: 105  <- BUY executes here
#   Row 2: 110  <- second BUY, ignored (already holding a position)
#   Row 3: 120  <- HOLD, no change
#   Row 4: 125  <- SELL executes here
data = pd.DataFrame({"Close": [100, 105, 110, 120, 125]})

backtester = Backtester(strategy=DummyStrategy(), initial_cash=1000.0)
result_df = backtester.run(data)

print(result_df[["Close", "Portfolio_Value"]])

# BUY at Close=105: floor(1000 / 105) = 9 shares, costing 945, leaving
# 55 in cash. The second BUY at Close=110 is ignored because a
# position is already open. HOLD at Close=120 leaves the position
# unchanged. SELL at Close=125 liquidates the 9 shares:
#   final_portfolio_value = 55 + (9 * 125) = 1180.0
#
# The full expected history, row by row:
#   Row 0: 1000.0  (initial cash; no signal generated yet)
#   Row 1: 1000.0  (BUY: 55 cash + 9 shares * 105 = 1000.0)
#   Row 2: 1045.0  (second BUY ignored: 55 cash + 9 shares * 110)
#   Row 3: 1135.0  (HOLD: 55 cash + 9 shares * 120)
#   Row 4: 1180.0  (SELL: 55 cash + 9 shares * 125, then liquidated)
expected_portfolio_values = [
    1000.0,
    1000.0,
    1045.0,
    1135.0,
    1180.0,
]
actual_portfolio_values = result_df["Portfolio_Value"].tolist()

if actual_portfolio_values != expected_portfolio_values:
    raise ValueError(
        f"Expected portfolio value history {expected_portfolio_values}, "
        f"but got {actual_portfolio_values}."
    )

print("Backtester test passed.")