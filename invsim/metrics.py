"""Performance and risk metrics.

Two complementary views of a DCA simulation:

- **Time-weighted (TWR)**: daily returns with contribution cash flows removed,
  i.e. the market performance of the strategy. Volatility, Sharpe, Sortino,
  max drawdown, and CAGR are computed on these returns. (The legacy code
  computed them on raw portfolio values, so every contribution registered as a
  fake positive return and inflated every ratio.)
- **Money-weighted (XIRR)**: the internal rate of return of the actual cash
  flows — what the investor's dollars earned given *when* they were invested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def flow_adjusted_returns(value: pd.Series, contributions: pd.Series) -> pd.Series:
    """Daily returns of the portfolio with external cash flows stripped out.

    Contributions execute at that day's close, so the market return earned by
    yesterday's holdings is ``(V_t - F_t) / V_{t-1} - 1``. For a single asset
    this equals the asset's price return exactly.
    """
    flows = contributions.reindex(value.index, fill_value=0.0)
    prev = value.shift(1)
    returns = (value - flows) / prev - 1.0
    return returns[prev > 0].dropna()


def xirr(cashflows: pd.Series) -> float:
    """Annualized internal rate of return of dated cash flows.

    Convention: investments negative, proceeds positive. Returns NaN if no
    root exists in (-99.99%, +10000%).
    """
    if len(cashflows) < 2:
        return float("nan")
    t0 = cashflows.index[0]
    years = np.array([(d - t0).days / 365.25 for d in cashflows.index])
    amounts = cashflows.to_numpy(dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    try:
        return brentq(npv, -0.9999, 100.0)
    except ValueError:
        return float("nan")


def max_drawdown(wealth_index: pd.Series) -> tuple[float, int]:
    """(max drawdown as a negative fraction, longest underwater spell in days).

    The underwater spell is measured in calendar days from a peak until the
    index regains it.
    """
    peak = wealth_index.cummax()
    drawdown = wealth_index / peak - 1.0
    mdd = float(drawdown.min()) if len(drawdown) else 0.0

    underwater = drawdown < 0
    longest = 0
    spell_start = None
    for day, is_under in underwater.items():
        if is_under and spell_start is None:
            spell_start = day
        elif not is_under and spell_start is not None:
            longest = max(longest, (day - spell_start).days)
            spell_start = None
    if spell_start is not None:
        longest = max(longest, (underwater.index[-1] - spell_start).days)
    return mdd, longest


def risk_reward_score(sortino: float, max_dd: float, total_return: float) -> float:
    """Heuristic combined score (kept from the legacy project, documented).

    ``Sortino × √(1 − |MaxDD|) × √(max(Return, 0))`` with a 0.7 "suspicion"
    penalty for implausibly clean results (Sortino > 7, or drawdown < 15%
    alongside return > 50%). All inputs are fractions, not percent.
    """
    dd_factor = np.sqrt(np.clip(1.0 - min(abs(max_dd), 1.0), 0.0, 1.0))
    return_factor = np.sqrt(max(total_return, 0.0))
    suspicion = 1.0
    if sortino > 7:
        suspicion *= 0.7
    if abs(max_dd) < 0.15 and total_return > 0.5:
        suspicion *= 0.7
    return float(sortino * dd_factor * return_factor * suspicion)


def performance_metrics(
    value: pd.Series,
    invested: pd.Series,
    contributions: pd.Series,
    rf_annual: pd.Series | float,
    label: str = "",
) -> dict:
    """Full metric set for one simulated DCA series.

    ``rf_annual`` is the annual risk-free rate: a scalar or a daily-aligned
    series (e.g. T-bill rates as-of each date).
    """
    active = value[invested.reindex(value.index).fillna(0) > 0]
    if len(active) < 10:
        return {}
    value = active
    invested = invested.reindex(value.index).ffill()

    returns = flow_adjusted_returns(value, contributions)
    if len(returns) < 10:
        return {}

    span_days = (value.index[-1] - value.index[0]).days or 1
    span_years = span_days / 365.25
    periods_per_year = len(returns) / span_years  # ~252 equities, ~365 crypto

    if isinstance(rf_annual, pd.Series):
        rf_daily = rf_annual.reindex(returns.index).ffill().bfill() / periods_per_year
    else:
        rf_daily = pd.Series(rf_annual / periods_per_year, index=returns.index)
    excess = returns - rf_daily

    volatility = float(returns.std(ddof=1))
    annual_vol = volatility * np.sqrt(periods_per_year)
    sharpe = float(excess.mean() / volatility * np.sqrt(periods_per_year)) if volatility > 0 else 0.0

    # Sortino with proper downside deviation (2nd lower partial moment over
    # *all* observations, not the std of only the negative ones).
    downside_dev = float(np.sqrt(np.mean(np.square(np.minimum(excess, 0.0)))))
    sortino = (
        float(excess.mean() / downside_dev * np.sqrt(periods_per_year))
        if downside_dev > 0
        else 0.0
    )

    wealth = (1.0 + returns).cumprod()
    mdd, underwater_days = max_drawdown(wealth)
    cagr = float(wealth.iloc[-1] ** (1.0 / span_years) - 1.0)
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0

    total_invested = float(invested.iloc[-1])
    final_value = float(value.iloc[-1])
    total_return = final_value / total_invested - 1.0 if total_invested > 0 else 0.0

    flows = -contributions[contributions.index <= value.index[-1]]
    flows = pd.concat([flows, pd.Series([final_value], index=[value.index[-1]])])
    money_weighted = xirr(flows.groupby(level=0).sum())

    return {
        "label": label,
        "start": value.index[0].date().isoformat(),
        "end": value.index[-1].date().isoformat(),
        "total_invested": round(total_invested, 2),
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return * 100, 2),
        "xirr_pct": round(money_weighted * 100, 2) if np.isfinite(money_weighted) else None,
        "cagr_pct": round(cagr * 100, 2),
        "annual_volatility_pct": round(annual_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(mdd * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 1),
        "max_underwater_days": underwater_days,
        "risk_reward_score": round(risk_reward_score(sortino, mdd, total_return), 3),
    }
