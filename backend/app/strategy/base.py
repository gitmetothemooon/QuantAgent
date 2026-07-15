"""Abstract base class for all trading strategies.

Every trading strategy in the project must inherit from
:class:`Strategy` and implement its ``generate_signal`` method. This
abstraction exists so that strategies are interchangeable: any code
that consumes a strategy - backtesting, portfolio management,
execution, optimization, ML-driven models, or agentic
decision-making - can depend on the single ``Strategy`` interface
rather than on the internals of any particular strategy
implementation. As long as a class fulfills this contract, it can be
swapped in or out, combined with other strategies, or driven by
automated components without those components needing to know how the
underlying trading logic works.

The interface passes both the current row of indicators and the
previous row, rather than the current row alone. Many trading signals
are not determined by a single row's values in isolation but by how
those values change from one row to the next - for example, an MACD
line crossing above its signal line, RSI crossing above or below a
threshold, a fast moving average crossing a slow one, a Bollinger Band
breakout, or future candlestick patterns. Giving every strategy access
to both rows lets it detect these transitions directly, instead of
strategies having to maintain their own history of prior rows.
"""

from abc import ABC, abstractmethod

import pandas as pd

from backend.app.strategy.signals import Signal


class Strategy(ABC):
    """The contract that every concrete trading strategy must follow.

    A ``Strategy`` is any class that can look at the current row of
    already-generated technical indicators, together with the
    previous row, and decide what action to take. Every concrete
    strategy must implement :meth:`generate_signal`, returning a
    :class:`Signal` value. This class defines that contract only; it
    contains no trading logic of its own and cannot be instantiated
    directly.
    """

    @abstractmethod
    def generate_signal(
        self,
        current_row: pd.Series,
        previous_row: pd.Series,
    ) -> Signal:
        """Generate a trading signal from the current and previous rows.

        Args:
            current_row: A pandas Series of already-generated technical
                indicators for today, i.e. the row being evaluated.
                Typical keys may include ``Close``, ``SMA20``,
                ``SMA50``, ``EMA20``, ``EMA50``, ``Daily_Return_Pct``,
                ``RSI14``, ``MACD``, ``MACD_Signal``,
                ``MACD_Histogram``, ``ATR14``, ``BB_Middle``,
                ``BB_Upper``, ``BB_Lower``, and ``OBV``. The base class
                makes no assumption about which keys are present and
                performs no validation; it is the responsibility of
                each concrete strategy to work with whichever keys it
                needs.
            previous_row: A pandas Series of the same indicators for
                yesterday, i.e. the row immediately preceding
                ``current_row``. Providing this alongside
                ``current_row`` allows a strategy to detect
                transitions and crossovers - such as a value moving
                from below a threshold to above it - rather than being
                limited to reasoning about the current state alone.

        Returns:
            The trading decision for this row, as one of
            ``Signal.BUY``, ``Signal.SELL``, or ``Signal.HOLD``.

        Raises:
            NotImplementedError: Always, when called on the abstract
                base class itself. This method must be overridden by
                every concrete subclass of ``Strategy``.
        """
        raise NotImplementedError