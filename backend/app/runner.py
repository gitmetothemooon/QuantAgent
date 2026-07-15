"""High-level orchestration of the QuantAgent trading pipeline.

This module provides a single entry point for running the project's
full pipeline end to end: loading historical data, generating
technical indicators, preparing the feature set, executing a trading
strategy through the backtester, and computing performance metrics.

The :class:`Runner` in this module contains no trading logic and
performs no technical analysis, data loading, or metric calculations
of its own. Its only job is to call the project's existing,
independently-developed modules in the right order and pass their
outputs to one another, hiding that plumbing behind a single ``run()``
call. Each stage remains owned by its dedicated module - this module
simply coordinates them.
"""

from backend.app.backtesting.backtester import Backtester
from backend.app.backtesting.performance import PerformanceAnalyzer
from backend.app.data.load_data import load_stock_data
from backend.app.features.feature_engineering import generate_features
from backend.app.features.feature_quality import prepare_features
from backend.app.strategy.base import Strategy


class Runner:
    """Orchestrates the full pipeline for a single strategy and ticker.

    A ``Runner`` is configured once with a trading strategy and a
    starting cash balance, and can then execute the entire pipeline -
    data loading, feature generation, feature preparation, backtesting,
    and performance analysis - for any ticker via :meth:`run`. It
    exists purely to hide the plumbing between the project's
    independent modules; it does not implement any data loading,
    indicator, strategy, backtesting, or performance logic itself.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: float = 100000.0,
    ) -> None:
        """Configure the runner with a strategy and starting cash.

        Args:
            strategy: The trading strategy to execute during the
                backtest.
            initial_cash: The starting cash balance for the simulated
                portfolio. Defaults to 100,000.0.
        """
        self.strategy = strategy
        self.initial_cash = initial_cash

    def run(self, ticker: str) -> dict[str, object]:
        """Run the full pipeline for a single ticker.

        This method runs the pipeline stages in order, passing each
        stage's output directly to the next: loading historical data
        with ``load_stock_data()``, generating technical indicators
        with ``generate_features()``, preparing the feature set with
        ``prepare_features()``, running the backtest with
        ``Backtester``, and computing performance metrics with
        ``PerformanceAnalyzer``. No exceptions raised by any of these
        stages are caught here; they propagate naturally to the
        caller.

        Args:
            ticker: The stock ticker symbol to run the pipeline for.

        Returns:
            A dictionary with the following keys:

            - ``"raw_data"``: The raw historical OHLCV DataFrame.
            - ``"features"``: The DataFrame with generated technical
              indicators.
            - ``"prepared_features"``: The DataFrame after feature
              quality preparation.
            - ``"backtest_results"``: The DataFrame produced by the
              backtest, including the ``Portfolio_Value`` column.
            - ``"performance"``: The performance summary dictionary
              returned by ``PerformanceAnalyzer.summary()``.
        """
        raw_df = load_stock_data(ticker)
        features_df = generate_features(raw_df)
        prepared_df = prepare_features(features_df)

        backtester = Backtester(
            strategy=self.strategy,
            initial_cash=self.initial_cash,
        )
        backtest_results_df = backtester.run(prepared_df)

        performance_analyzer = PerformanceAnalyzer(backtest_results_df)
        performance_summary = performance_analyzer.summary()

        return {
            "raw_data": raw_df,
            "features": features_df,
            "prepared_features": prepared_df,
            "backtest_results": backtest_results_df,
            "performance": performance_summary,
        }