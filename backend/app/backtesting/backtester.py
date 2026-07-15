"""A simple backtesting engine for simulating strategy execution.

This module is responsible for a single task: replaying a
:class:`~backend.app.strategy.base.Strategy`'s trading signals over a
DataFrame of historical technical indicators, and tracking how a
simulated portfolio's value would have evolved as a result.

This module does NOT:
    - calculate technical indicators
    - generate trading signals (that is the strategy's job)
    - compute performance metrics (e.g. returns, drawdown, Sharpe ratio)
    - download data
    - plot results

This first version of the backtester is intentionally simple. It
makes the following simplifying assumptions:
    - Trades execute at the same day's ``Close`` price used to
      generate the signal (no next-day execution or intra-day
      pricing).
    - Only whole shares can be bought; any leftover cash that can't
      buy another full share is simply carried forward.
    - There are no transaction costs, commissions, or slippage.
    - There is no leverage and no short selling: the strategy can only
      be flat or long a single position at a time.
    - A BUY signal is ignored while a position is already open, and a
      SELL signal is ignored while no position is open.
"""

import pandas as pd

from backend.app.strategy.base import Strategy
from backend.app.strategy.signals import Signal


class Backtester:
    """Simulates a Strategy's trading signals over historical data.

    A ``Backtester`` pairs a :class:`~backend.app.strategy.base.Strategy`
    with a starting cash balance and replays that strategy's signals,
    day by day, over a DataFrame of already-prepared technical
    indicators. It tracks a single long-or-flat position (no
    shorting, no leverage) and records the resulting portfolio value
    for every row, producing a simple equity curve that later stages
    of the project (e.g. performance metrics or plotting) can build
    on.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: float = 100000.0,
    ) -> None:
        """Initialize the backtester with a strategy and starting cash.

        No trading or simulation happens here; this only stores the
        configuration that :meth:`run` will later use.

        Args:
            strategy: The trading strategy whose ``generate_signal``
                method will be called for each row during
                simulation.
            initial_cash: The starting cash balance for the simulated
                portfolio. Defaults to 100,000.0.
        """
        self.strategy = strategy
        self.initial_cash = initial_cash

    def run(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Simulate the strategy's signals over a DataFrame of indicators.

        The input DataFrame is assumed to have already passed through
        validation, feature engineering, and feature quality
        preparation, so every row contains at least ``Close``,
        ``SMA20``, ``SMA50``, ``EMA20``, ``EMA50``, ``RSI14``,
        ``MACD``, ``MACD_Signal``, ``MACD_Histogram``, ``ATR14``,
        ``BB_Middle``, ``BB_Upper``, ``BB_Lower``, and ``OBV``.
        Interpreting these indicators is entirely the strategy's
        responsibility; the backtester only tracks cash, shares, and
        portfolio value.

        The simulation starts at row index 1, since the strategy needs
        both a current row and a previous row to generate a signal.
        Row 0 is given a portfolio value equal to ``initial_cash``
        since no trading has happened yet. On each subsequent row, a
        BUY signal spends all available cash on as many whole shares
        as possible (only if no position is currently open), and a
        SELL signal liquidates the entire position at that day's
        ``Close`` (only if a position is currently open); a HOLD
        signal, or a BUY/SELL signal that doesn't apply, leaves the
        position unchanged.

        Args:
            data: A DataFrame of historical OHLCV data with
                already-generated technical indicator columns. This
                DataFrame is never modified.

        Returns:
            A copy of ``data`` with one additional column,
            ``Portfolio_Value``, containing the simulated portfolio
            value (cash plus the market value of any open position)
            for every row.
        """
        result_df = data.copy()

        cash = self.initial_cash
        shares = 0
        position_open = False

        portfolio_values = [self.initial_cash] * len(result_df)

        for i in range(1, len(result_df)):
            current_row = result_df.iloc[i]
            previous_row = result_df.iloc[i - 1]
            current_close = current_row["Close"]

            signal = self.strategy.generate_signal(current_row, previous_row)

            is_buy_signal = signal == Signal.BUY and not position_open
            is_sell_signal = signal == Signal.SELL and position_open

            if is_buy_signal:
                shares = int(cash // current_close)
                cash = cash - shares * current_close
                position_open = True

            if is_sell_signal:
                cash = cash + shares * current_close
                shares = 0
                position_open = False

            if position_open:
                portfolio_values[i] = cash + shares * current_close
            else:
                portfolio_values[i] = cash

        result_df["Portfolio_Value"] = portfolio_values

        return result_df