"""Portfolio performance analysis for backtest results.

This module is responsible for a single task: analyzing the output of
a :class:`~backend.app.backtesting.backtester.Backtester` run and
computing common portfolio performance metrics from its
``Portfolio_Value`` history.

This module does NOT:
    - execute trades
    - generate trading signals
    - calculate technical indicators
    - download data
    - plot results

Only pandas and the standard library ``math`` module are used; no
NumPy or external finance libraries are required or permitted.
"""

import math

import pandas as pd


class PerformanceAnalyzer:
    """Computes standard performance metrics from a backtest's results.

    A ``PerformanceAnalyzer`` wraps the DataFrame produced by
    :meth:`Backtester.run`, which must contain a ``Portfolio_Value``
    column representing the simulated portfolio's value over time.
    Each metric method reads directly from that column and returns a
    single float; :meth:`summary` collects all of them into one
    dictionary. Nothing here mutates the input DataFrame or performs
    any trading of its own - this class is purely an analysis layer
    on top of an already-completed backtest.
    """

    def __init__(self, results: pd.DataFrame) -> None:
        """Store and validate the backtest results to be analyzed.

        Args:
            results: The DataFrame returned by ``Backtester.run()``,
                containing at least a ``Portfolio_Value`` column.

        Raises:
            ValueError: If ``results`` does not contain a
                ``Portfolio_Value`` column.
        """
        if "Portfolio_Value" not in results.columns:
            raise ValueError(
                "results DataFrame must contain a 'Portfolio_Value' column."
            )

        self.results = results

    def total_return(self) -> float:
        """Calculate the portfolio's total return over the backtest.

        Total return compares the very first and very last portfolio
        values, ignoring everything in between:

            ``((final_value / initial_value) - 1) * 100``

        Returns:
            The total return as a percentage.
        """
        initial_value = self.results["Portfolio_Value"].iloc[0]
        final_value = self.results["Portfolio_Value"].iloc[-1]

        return ((final_value / initial_value) - 1) * 100

    def cagr(self) -> float:
        """Calculate the Compound Annual Growth Rate (CAGR).

        CAGR expresses the total return as if it had grown at a
        constant, compounding rate every trading day, then annualizes
        that rate assuming 252 trading days per year:

            ``(final_value / initial_value) ** (252 / trading_periods) - 1``

        where ``trading_periods`` is the number of return intervals in
        the backtest (i.e. the number of rows minus one, since a
        return is measured *between* two consecutive rows rather than
        at each individual row). This lets backtests of different
        lengths be compared on an apples-to-apples, per-year basis.

        Returns:
            The CAGR as a percentage, or ``0.0`` if the results
            contain fewer than two rows (i.e. no complete return
            period to annualize).
        """
        trading_periods = len(self.results) - 1

        if trading_periods <= 0:
            return 0.0

        initial_value = self.results["Portfolio_Value"].iloc[0]
        final_value = self.results["Portfolio_Value"].iloc[-1]

        growth_factor = final_value / initial_value
        annualized_growth_rate = growth_factor ** (252 / trading_periods) - 1

        return annualized_growth_rate * 100

    def daily_volatility(self) -> float:
        """Calculate the standard deviation of daily portfolio returns.

        Daily percentage returns are computed with
        ``Portfolio_Value.pct_change()``, and volatility is the
        standard deviation of those returns - a measure of how much
        the portfolio's day-to-day performance fluctuates, regardless
        of direction.

        Returns:
            The standard deviation of daily returns as a percentage,
            or ``0.0`` if there are fewer than two valid daily returns
            to compute a standard deviation from.
        """
        daily_returns = self.results["Portfolio_Value"].pct_change().dropna()

        if len(daily_returns) < 2:
            return 0.0

        return daily_returns.std() * 100

    def max_drawdown(self) -> float:
        """Calculate the maximum drawdown of the portfolio.

        For each row, the "running peak" is the highest portfolio
        value seen so far (``Portfolio_Value.cummax()``), and the
        drawdown at that row is how far below that peak the portfolio
        currently sits:

            ``drawdown = (Portfolio_Value - running_peak) / running_peak``

        The maximum drawdown is the single worst (most negative) of
        these values across the whole backtest - the largest
        peak-to-trough decline the portfolio experienced.

        Returns:
            The maximum drawdown as a percentage. This value is less
            than or equal to zero, since a drawdown of zero means the
            portfolio was at a new peak.
        """
        portfolio_value = self.results["Portfolio_Value"]
        running_peak = portfolio_value.cummax()
        drawdown = (portfolio_value - running_peak) / running_peak

        return drawdown.min() * 100

    def sharpe_ratio(self) -> float:
        """Calculate the annualized Sharpe ratio, assuming a 0% risk-free rate.

        The Sharpe ratio measures return earned per unit of risk
        taken. Daily returns are computed with
        ``Portfolio_Value.pct_change()``; their mean divided by their
        standard deviation gives a daily Sharpe ratio, which is then
        annualized by multiplying by the square root of 252 (the
        assumed number of trading days per year):

            ``(mean(daily_returns) / std(daily_returns)) * sqrt(252)``

        Because the risk-free rate is assumed to be 0, daily returns
        are used directly as excess returns.

        Returns:
            The annualized Sharpe ratio, or ``0.0`` if there are fewer
            than two valid daily returns, or if their standard
            deviation is zero (which would otherwise cause a
            division by zero).
        """
        daily_returns = self.results["Portfolio_Value"].pct_change().dropna()

        if len(daily_returns) < 2:
            return 0.0

        volatility = daily_returns.std()

        if volatility == 0:
            return 0.0

        mean_daily_return = daily_returns.mean()

        return (mean_daily_return / volatility) * math.sqrt(252)

    def summary(self) -> dict[str, float]:
        """Collect all performance metrics into a single dictionary.

        Each value is taken directly from its corresponding method
        rather than being recomputed, so this dictionary always stays
        consistent with calling the individual methods yourself.

        Returns:
            A dictionary with the following keys: ``"Total Return (%)"``,
            ``"CAGR (%)"``, ``"Daily Volatility (%)"``,
            ``"Maximum Drawdown (%)"``, and ``"Sharpe Ratio"``.
        """
        return {
            "Total Return (%)": self.total_return(),
            "CAGR (%)": self.cagr(),
            "Daily Volatility (%)": self.daily_volatility(),
            "Maximum Drawdown (%)": self.max_drawdown(),
            "Sharpe Ratio": self.sharpe_ratio(),
        }