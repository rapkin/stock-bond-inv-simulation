"""DCA simulation engines (vectorized) and benchmarks.

All engines take a price series/matrix and a contribution schedule (Series of
dollar amounts indexed by execution day, see :mod:`invsim.schedule`) and
return a daily DataFrame with at least ``value``, ``invested``, and
``contribution`` columns. Purchases execute at that day's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import align_monthly
from .optimize import walk_forward_weights


def simulate_dca(prices: pd.Series, contributions: pd.Series) -> pd.DataFrame:
    """Buy ``contribution / price`` shares on each contribution day."""
    flows = contributions.reindex(prices.index, fill_value=0.0)
    shares = (flows / prices).cumsum()
    frame = pd.DataFrame(
        {
            "price": prices,
            "contribution": flows,
            "invested": flows.cumsum(),
            "shares": shares,
            "value": shares * prices,
        }
    )
    frame.index.name = "date"
    return frame


def simulate_portfolio(
    prices: pd.DataFrame, contributions: pd.Series, weights: pd.Series
) -> pd.DataFrame:
    """Fixed-weight DCA: each contribution is split by ``weights`` (no selling)."""
    weights = weights.reindex(prices.columns).fillna(0.0)
    weights = weights / weights.sum()
    flows = contributions.reindex(prices.index, fill_value=0.0)
    per_asset_flows = pd.DataFrame(
        np.outer(flows.to_numpy(), weights.to_numpy()), index=prices.index, columns=prices.columns
    )
    shares = (per_asset_flows / prices).cumsum()
    frame = pd.DataFrame(
        {
            "contribution": flows,
            "invested": flows.cumsum(),
            "value": (shares * prices).sum(axis=1),
        }
    )
    frame.index.name = "date"
    return frame


def simulate_dynamic_portfolio(
    prices: pd.DataFrame,
    contributions: pd.Series,
    lookback_years: float = 3.0,
    rebalance_months: int = 3,
    max_weight: float = 0.4,
    min_weight: float = 0.0,
    max_change: float | None = 0.10,
    risk_free_rate: float = 0.02,
    rebalance_holdings: bool = True,
    objective: str = "sharpe",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward optimized DCA portfolio.

    Weights are re-optimized every ``rebalance_months`` using only the
    trailing ``lookback_years`` of prices before the rebalance date (no
    look-ahead). Between rebalances, contributions are split by the current
    target weights; if ``rebalance_holdings``, existing holdings are also
    traded back to target at each rebalance.

    Investing starts after the first lookback window (warm-up), so callers
    comparing strategies should restrict all of them to the same
    contribution window (see :func:`investing_start`).

    Returns ``(daily frame, weight history indexed by rebalance date)``.
    """
    trading_days = prices.index
    start_day = investing_start(trading_days, lookback_years)
    rebalance_days = _rebalance_days(trading_days, start_day, rebalance_months)

    weight_history = walk_forward_weights(
        prices,
        rebalance_days,
        lookback_years,
        risk_free_rate=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight,
        max_change=max_change,
        objective=objective,
    )
    if weight_history.empty:
        raise ValueError("Not enough history for the requested lookback window")

    buys = contributions[contributions.index >= weight_history.index[0]]
    price_lookup = prices.to_numpy()
    day_positions = {day: i for i, day in enumerate(trading_days)}

    holdings = np.zeros(len(prices.columns))
    events: dict[pd.Timestamp, np.ndarray] = {}
    rebalance_iter = iter(weight_history.iterrows())
    next_rebalance, current_weights = next(rebalance_iter)

    for day in trading_days[trading_days >= weight_history.index[0]]:
        day_prices = price_lookup[day_positions[day]]
        while next_rebalance is not None and day >= next_rebalance:
            target = weight_history.loc[next_rebalance].to_numpy()
            if rebalance_holdings and holdings.sum() > 0:
                holdings = (holdings @ day_prices) * target / day_prices
            current_weights = target
            next_rebalance = next(rebalance_iter, (None, None))[0]
            events[day] = holdings.copy()
        if day in buys.index:
            holdings = holdings + buys.loc[day] * np.asarray(current_weights) / day_prices
            events[day] = holdings.copy()

    shares = (
        pd.DataFrame.from_dict(events, orient="index", columns=prices.columns)
        .reindex(trading_days)
        .ffill()
        .fillna(0.0)
    )
    flows = buys.reindex(trading_days, fill_value=0.0)
    frame = pd.DataFrame(
        {
            "contribution": flows,
            "invested": flows.cumsum(),
            "value": (shares * prices).sum(axis=1),
        }
    )
    frame.index.name = "date"
    return frame, weight_history


def investing_start(trading_days: pd.DatetimeIndex, lookback_years: float) -> pd.Timestamp:
    """First trading day after the lookback warm-up window."""
    cutoff = trading_days[0] + pd.DateOffset(years=lookback_years)
    eligible = trading_days[trading_days >= cutoff]
    if len(eligible) == 0:
        raise ValueError(f"Price history shorter than lookback of {lookback_years} years")
    return eligible[0]


def _rebalance_days(
    trading_days: pd.DatetimeIndex, start: pd.Timestamp, months: int
) -> pd.DatetimeIndex:
    calendar = pd.date_range(start, trading_days[-1], freq=pd.DateOffset(months=months))
    positions = np.searchsorted(trading_days.values, calendar.values, side="left")
    positions = np.unique(positions[positions < len(trading_days)])
    return pd.DatetimeIndex(trading_days.values[positions])


# --- benchmarks ---------------------------------------------------------------


def simulate_tbill(
    contributions: pd.Series, tbill_rates: pd.Series, index: pd.DatetimeIndex
) -> pd.Series:
    """Value of the same contributions rolled into 3-month T-bills.

    Interest accrues over actual calendar-day gaps (ACT/365), so weekends and
    holidays earn interest too — the legacy version only compounded on trading
    days and understated the benchmark.
    """
    rates = align_monthly(tbill_rates, index)
    day_gaps = index.to_series().diff().dt.days.fillna(0.0)
    growth = np.exp((np.log1p(rates) * day_gaps / 365.0).cumsum())
    flows = contributions.reindex(index, fill_value=0.0)
    return growth * (flows / growth).cumsum()


def real_factor(cpi: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Deflator expressing nominal dollars in start-of-period purchasing power."""
    cpi_daily = align_monthly(cpi, index)
    return cpi_daily.iloc[0] / cpi_daily


def add_benchmarks(
    frame: pd.DataFrame, tbill_rates: pd.Series, cpi: pd.Series
) -> pd.DataFrame:
    """Attach T-bill benchmark and inflation-adjusted (real) columns."""
    frame = frame.copy()
    frame["tbill_value"] = simulate_tbill(frame["contribution"], tbill_rates, frame.index)
    deflator = real_factor(cpi, frame.index)
    for column in ("value", "invested", "tbill_value"):
        frame[f"{column}_real"] = frame[column] * deflator
    frame["profit"] = frame["value"] - frame["invested"]
    frame["profit_real"] = frame["value_real"] - frame["invested_real"]
    return frame
