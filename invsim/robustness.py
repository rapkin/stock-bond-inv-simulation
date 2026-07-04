"""Rolling-window robustness analysis.

A single backtest is one draw from history: DCA outcomes are extremely
sensitive to the start date. This module simulates the same DCA plan over
*every* rolling window of a given length and reports the distribution of
outcomes — median/worst/best money-weighted return, how often the plan beat
T-bills, and how often it beat inflation — instead of a point estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import flow_adjusted_returns, max_drawdown, xirr
from .schedule import contribution_schedule
from .simulation import NO_COSTS, Costs, real_factor, simulate_dca, simulate_tbill


def window_bounds(
    index: pd.DatetimeIndex, window_years: float, step_months: int = 3
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) pairs of every rolling window that fits in ``index``."""
    last_start = index[-1] - pd.DateOffset(years=window_years)
    starts = pd.date_range(index[0], last_start, freq=pd.DateOffset(months=step_months))
    return [(s, s + pd.DateOffset(years=window_years)) for s in starts]


def rolling_dca(
    prices: pd.Series,
    tbill_rates: pd.Series,
    cpi: pd.Series,
    window_years: float,
    amount: float,
    step_months: int = 3,
    costs: Costs = NO_COSTS,
) -> pd.DataFrame:
    """One row of outcome stats per rolling window."""
    rows = []
    for start, end in window_bounds(prices.index, window_years, step_months):
        window_prices = prices.loc[start:end]
        if len(window_prices) < 60:
            continue
        contributions = contribution_schedule(window_prices.index, amount)
        if len(contributions) < 3:
            continue
        frame = simulate_dca(window_prices, contributions, costs)
        final_value = float(frame["value"].iloc[-1])
        invested = float(frame["invested"].iloc[-1])

        flows = -contributions.copy()
        flows = pd.concat([flows, pd.Series([final_value], index=[frame.index[-1]])])
        window_xirr = xirr(flows.groupby(level=0).sum())

        returns = flow_adjusted_returns(frame["value"], contributions)
        mdd, _ = max_drawdown((1 + returns).cumprod())

        tbill_final = float(
            simulate_tbill(contributions, tbill_rates, window_prices.index).iloc[-1]
        )
        deflator_end = float(real_factor(cpi, window_prices.index).iloc[-1])

        rows.append(
            {
                "window_start": start.date().isoformat(),
                "window_end": frame.index[-1].date().isoformat(),
                "invested": round(invested, 2),
                "final_value": round(final_value, 2),
                "xirr_pct": round(window_xirr * 100, 2) if np.isfinite(window_xirr) else None,
                "total_return_pct": round((final_value / invested - 1) * 100, 2),
                "max_drawdown_pct": round(mdd * 100, 2),
                "beat_tbills": final_value > tbill_final,
                "beat_inflation": final_value * deflator_end > invested,
            }
        )
    return pd.DataFrame(rows)


def summarize_windows(windows: pd.DataFrame, label: str = "") -> dict:
    """Distribution summary across all rolling windows of one asset."""
    if windows.empty:
        return {}
    xirr_values = windows["xirr_pct"].dropna()
    return {
        "label": label,
        "windows": len(windows),
        "median_xirr_pct": round(float(xirr_values.median()), 2),
        "p10_xirr_pct": round(float(xirr_values.quantile(0.10)), 2),
        "worst_xirr_pct": round(float(xirr_values.min()), 2),
        "best_xirr_pct": round(float(xirr_values.max()), 2),
        "worst_drawdown_pct": round(float(windows["max_drawdown_pct"].min()), 2),
        "beat_tbills_pct": round(float(windows["beat_tbills"].mean() * 100), 1),
        "beat_inflation_pct": round(float(windows["beat_inflation"].mean() * 100), 1),
    }
