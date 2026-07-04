"""Portfolio weight optimization (Modern Portfolio Theory).

Inputs are **asset price returns** — never DCA portfolio values, whose daily
changes are contaminated by contribution cash flows (the core input bug of the
legacy optimizer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _annualized(returns: pd.DataFrame, periods_per_year: float) -> tuple[np.ndarray, np.ndarray]:
    mean = returns.mean().to_numpy() * periods_per_year
    cov = returns.cov().to_numpy() * periods_per_year
    return mean, cov


def _bounds(
    n: int,
    min_weight: float,
    max_weight: float,
    prev_weights: np.ndarray | None,
    max_change: float | None,
) -> list[tuple[float, float]]:
    """Box bounds, optionally tightened by a per-rebalance turnover limit.

    If the turnover-tightened box cannot contain a valid portfolio
    (sum of uppers < 1 or sum of lowers > 1), fall back to the plain box.
    """
    if prev_weights is not None and max_change is not None:
        lows = np.maximum(min_weight, prev_weights - max_change)
        highs = np.minimum(max_weight, prev_weights + max_change)
        if highs.sum() >= 1.0 and lows.sum() <= 1.0:
            return list(zip(lows, highs))
    return [(min_weight, max_weight)] * n


def optimize_weights(
    returns: pd.DataFrame,
    objective: str = "sharpe",
    risk_free_rate: float = 0.02,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    prev_weights: pd.Series | None = None,
    max_change: float | None = None,
    periods_per_year: float = 252.0,
) -> pd.Series:
    """Long-only weights maximizing Sharpe (or minimizing variance).

    Raises ValueError when the constraint box cannot sum to 1 (e.g. two assets
    with max_weight=0.4) — the legacy code silently returned an invalid
    portfolio in that case.
    """
    tickers = list(returns.columns)
    n = len(tickers)
    if n == 1:
        return pd.Series([1.0], index=tickers)
    if max_weight * n < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible: {n} assets with max_weight={max_weight} cannot sum to 1"
        )

    mean, cov = _annualized(returns, periods_per_year)
    prev = prev_weights.reindex(tickers).fillna(0).to_numpy() if prev_weights is not None else None
    bounds = _bounds(n, min_weight, max_weight, prev, max_change)
    x0 = prev if prev is not None else np.full(n, 1.0 / n)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    if x0.sum() > 0:
        x0 = x0 / x0.sum()

    def portfolio_vol(w: np.ndarray) -> float:
        return float(np.sqrt(w @ cov @ w))

    if objective == "min_var":
        target = portfolio_vol
    else:

        def target(w: np.ndarray) -> float:
            vol = portfolio_vol(w)
            if vol <= 0:
                return 0.0
            return -(w @ mean - risk_free_rate) / vol

    result = minimize(
        target,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500},
    )
    weights = result.x if result.success else x0
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=tickers)


def walk_forward_weights(
    prices: pd.DataFrame,
    rebalance_days: pd.DatetimeIndex,
    lookback_years: float,
    risk_free_rate: float = 0.02,
    min_weight: float = 0.0,
    max_weight: float = 0.4,
    max_change: float | None = 0.10,
    objective: str = "sharpe",
) -> pd.DataFrame:
    """Target weights re-optimized at each rebalance date.

    At date *d* only prices strictly **before** *d* (within the lookback
    window) are used — no look-ahead. Returns a DataFrame indexed by
    rebalance date.
    """
    history: dict[pd.Timestamp, pd.Series] = {}
    prev: pd.Series | None = None
    for day in rebalance_days:
        window_start = day - pd.DateOffset(years=lookback_years)
        window = prices.loc[(prices.index >= window_start) & (prices.index < day)]
        if len(window) < 60:  # not enough history to estimate anything
            continue
        returns = window.pct_change().dropna()
        span_years = max((window.index[-1] - window.index[0]).days / 365.25, 1e-9)
        ppy = len(returns) / span_years
        prev = optimize_weights(
            returns,
            objective=objective,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            prev_weights=prev,
            max_change=max_change if prev is not None else None,
            periods_per_year=ppy,
        )
        history[day] = prev
    return pd.DataFrame(history).T
