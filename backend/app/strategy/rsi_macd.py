"""The project's first concrete trading strategy: RSI + MACD.

This strategy combines a trend filter (price relative to SMA50), a
momentum filter (RSI14 overbought/oversold levels), and a MACD
crossover trigger into a single BUY/SELL/HOLD decision. It is
intentionally simple: it uses only a handful of columns produced by
the feature engineering layer and applies a small set of readable
boolean conditions to them, with no position sizing, risk management,
or optimization of any kind.

Its purpose is as much illustrative as practical - it demonstrates the
intended shape of a concrete :class:`~backend.app.strategy.base.Strategy`
implementation, and shows how the technical indicators produced by
``backend.app.features`` (SMA, RSI, MACD, and the rest) flow directly
into a strategy's decision logic via the ``current_row`` and
``previous_row`` passed to ``generate_signal``.
"""

import pandas as pd

from backend.app.strategy.base import Strategy
from backend.app.strategy.signals import Signal


class RSIMACDStrategy(Strategy):
    """A trend, momentum, and crossover strategy combining SMA, RSI, and MACD.

    This strategy issues a BUY signal only when price is in an uptrend
    relative to its 50-period SMA, RSI14 indicates an oversold
    condition, and MACD crosses bullish on the current row. It issues
    a SELL signal only when the mirrored conditions hold: a downtrend,
    an overbought RSI14, and a bearish MACD crossover. In every other
    case it holds. Requiring all three conditions together is a
    deliberate design choice to keep the strategy conservative and its
    logic easy to follow, at the cost of trading less frequently than
    a strategy using any single condition alone.
    """

    def generate_signal(
        self,
        current_row: pd.Series,
        previous_row: pd.Series,
    ) -> Signal:
        """Generate a BUY, SELL, or HOLD signal from RSI, MACD, and SMA50.

        Args:
            current_row: A pandas Series of today's technical
                indicators. Must include ``Close``, ``SMA50``,
                ``RSI14``, ``MACD``, and ``MACD_Signal``.
            previous_row: A pandas Series of yesterday's technical
                indicators, with the same required keys as
                ``current_row``. Used to detect whether a MACD
                crossover occurred between yesterday and today.

        Returns:
            ``Signal.BUY`` if price is above SMA50, RSI14 is below 30,
            and MACD crosses above its signal line today.
            ``Signal.SELL`` if price is below SMA50, RSI14 is above
            70, and MACD crosses below its signal line today.
            ``Signal.HOLD`` in every other case.
        """
        is_uptrend = current_row["Close"] > current_row["SMA50"]
        is_downtrend = current_row["Close"] < current_row["SMA50"]

        is_oversold = current_row["RSI14"] < 30
        is_overbought = current_row["RSI14"] > 70

        bullish_crossover = (
            previous_row["MACD"] <= previous_row["MACD_Signal"]
            and current_row["MACD"] > current_row["MACD_Signal"]
        )
        bearish_crossover = (
            previous_row["MACD"] >= previous_row["MACD_Signal"]
            and current_row["MACD"] < current_row["MACD_Signal"]
        )

        if is_uptrend and is_oversold and bullish_crossover:
            return Signal.BUY

        if is_downtrend and is_overbought and bearish_crossover:
            return Signal.SELL

        return Signal.HOLD