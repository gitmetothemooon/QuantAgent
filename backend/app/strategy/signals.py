"""Standard trading signals for the QuantAgent strategy layer.

This module standardizes trading decisions across the project: every
strategy implemented in ``backend.app.strategy`` should return one of
the values defined in :class:`Signal` rather than an ad-hoc string or
integer. Centralizing these values here avoids magic strings scattered
throughout the codebase (e.g. comparing against ``"buy"`` in one place
and ``"BUY"`` in another), and gives downstream consumers such as
backtesting, execution, and reporting code a single, type-checked
vocabulary to work against.
"""

from enum import Enum, auto


class Signal(Enum):
    """The set of trading decisions a strategy can produce.

    Attributes:
        BUY: The strategy recommends entering or adding to a long
            position.
        SELL: The strategy recommends exiting a long position or
            entering a short position.
        HOLD: The strategy recommends taking no action and maintaining
            the current position.
    """

    BUY = auto()
    SELL = auto()
    HOLD = auto()