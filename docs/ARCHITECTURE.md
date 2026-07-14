# QuantAgent Architecture

## Objective

Build a deterministic paper-trading recommendation engine for Indian equities.

The system should:

- Recommend stocks to buy.
- Recommend quantity.
- Recommend target price.
- Recommend stop loss.
- Explain every recommendation.
- Track paper trades.
- Measure portfolio performance.
- Never execute trades automatically.

---

## Layer 1 — Data

Responsible for obtaining and validating market data.

Modules:

- download_data
- load_data
- validate_data

---

## Layer 2 — Feature Engineering

Responsible for transforming raw OHLCV data into meaningful indicators.

Modules:

- feature_engineering

---

## Layer 3 — Signals

Generate buy/hold/sell signals.

(Not implemented)

---

## Layer 4 — Risk

Position sizing.

Maximum loss.

Portfolio constraints.

(Not implemented)

---

## Layer 5 — Portfolio

Track holdings.

Track realized profit.

Track unrealized profit.

(Not implemented)

---

## Layer 6 — Recommendation

Explain every recommendation.

(Not implemented)
