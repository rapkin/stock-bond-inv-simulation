"""DCA simulation engines (vectorized), benchmarks, and trading costs.

All engines take a price series/matrix and a contribution schedule (Series of
dollar amounts indexed by execution day, see :mod:`invsim.schedule`) and
return a daily DataFrame with at least ``value``, ``invested``, and
``contribution`` columns. Purchases execute at that day's close.

Costs model (:class:`Costs`, all default to zero):

- ``commission_pct`` / ``commission_fixed`` — charged on every trade leg
  (each asset bought or sold is one leg).
- ``annual_fee_pct`` — continuous ACT/365 drag (advisory fee, or the expense
  ratio of a raw index that has none priced in; ETF adjusted closes already
  include their expense ratio). Implemented by simulating against
  fee-adjusted prices ``p̃_t = p_t · (1 − fee)^(days/365)``, which is exactly
  equivalent to holdings decaying at the fee rate.
- ``capital_gains_tax_pct`` — tax on gains realized when the dynamic
  portfolio sells at a rebalance, using average-cost basis per asset.
  (Final liquidation is not taxed; results are pre-exit-tax.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import align_monthly
from .optimize import walk_forward_weights


@dataclass(frozen=True)
class Costs:
    """Trading costs. Rates are fractions: 0.001 = 0.1%."""

    commission_pct: float = 0.0
    commission_fixed: float = 0.0
    annual_fee_pct: float = 0.0
    capital_gains_tax_pct: float = 0.0

    def net_of_commission(self, gross):
        """Amount actually invested from a gross trade leg (array-safe)."""
        gross = np.asarray(gross, dtype=float)
        net = np.where(
            gross > 0, gross * (1.0 - self.commission_pct) - self.commission_fixed, 0.0
        )
        return np.maximum(net, 0.0)

    def commission_on(self, trade_value: float) -> float:
        """Commission for one trade leg of ``trade_value`` dollars."""
        if trade_value <= 0:
            return 0.0
        return trade_value * self.commission_pct + self.commission_fixed


NO_COSTS = Costs()


def fee_adjusted(prices: pd.Series | pd.DataFrame, annual_fee_pct: float):
    """Prices with a continuous ACT/365 fee drag baked in (see module doc)."""
    if annual_fee_pct <= 0:
        return prices
    days = (prices.index - prices.index[0]).days.to_numpy()
    decay = (1.0 - annual_fee_pct) ** (days / 365.0)
    if isinstance(prices, pd.DataFrame):
        return prices.mul(decay, axis=0)
    return prices * decay


def simulate_dca(
    prices: pd.Series, contributions: pd.Series, costs: Costs = NO_COSTS
) -> pd.DataFrame:
    """Buy ``net contribution / price`` shares on each contribution day."""
    effective = fee_adjusted(prices, costs.annual_fee_pct)
    flows = contributions.reindex(prices.index, fill_value=0.0)
    net_flows = pd.Series(costs.net_of_commission(flows), index=prices.index)
    shares = (net_flows / effective).cumsum()
    frame = pd.DataFrame(
        {
            "price": prices,
            "contribution": flows,
            "invested": flows.cumsum(),
            "shares": shares,
            "value": shares * effective,
        }
    )
    frame.index.name = "date"
    return frame


def simulate_portfolio(
    prices: pd.DataFrame,
    contributions: pd.Series,
    weights: pd.Series,
    costs: Costs = NO_COSTS,
) -> pd.DataFrame:
    """Fixed-weight DCA: each contribution is split by ``weights`` (no selling).

    Every asset bought on a contribution day is a separate trade leg for
    commission purposes.
    """
    weights = weights.reindex(prices.columns).fillna(0.0)
    weights = weights / weights.sum()
    effective = fee_adjusted(prices, costs.annual_fee_pct)
    flows = contributions.reindex(prices.index, fill_value=0.0)
    gross_legs = np.outer(flows.to_numpy(), weights.to_numpy())
    net_legs = pd.DataFrame(
        costs.net_of_commission(gross_legs), index=prices.index, columns=prices.columns
    )
    shares = (net_legs / effective).cumsum()
    frame = pd.DataFrame(
        {
            "contribution": flows,
            "invested": flows.cumsum(),
            "value": (shares * effective).sum(axis=1),
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
    costs: Costs = NO_COSTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward optimized DCA portfolio.

    Weights are re-optimized every ``rebalance_months`` using only the
    trailing ``lookback_years`` of **raw** prices before the rebalance date
    (no look-ahead). Between rebalances, contributions are split by the
    current target weights; if ``rebalance_holdings``, existing holdings are
    also traded back to target at each rebalance, paying commissions on every
    leg and capital-gains tax on realized gains (average-cost basis).

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

    effective = fee_adjusted(prices, costs.annual_fee_pct)
    buys = contributions[contributions.index >= weight_history.index[0]]
    price_lookup = effective.to_numpy()
    day_positions = {day: i for i, day in enumerate(trading_days)}

    n_assets = len(prices.columns)
    holdings = np.zeros(n_assets)
    cost_basis = np.zeros(n_assets)  # dollars paid for current holdings, per asset
    events: dict[pd.Timestamp, np.ndarray] = {}
    rebalance_iter = iter(weight_history.iterrows())
    next_rebalance, current_weights = next(rebalance_iter)

    for day in trading_days[trading_days >= weight_history.index[0]]:
        day_prices = price_lookup[day_positions[day]]
        while next_rebalance is not None and day >= next_rebalance:
            target = weight_history.loc[next_rebalance].to_numpy()
            if rebalance_holdings and holdings.sum() > 0:
                holdings, cost_basis = _rebalance(
                    holdings, cost_basis, day_prices, target, costs
                )
            current_weights = target
            next_rebalance = next(rebalance_iter, (None, None))[0]
            events[day] = holdings.copy()
        if day in buys.index:
            gross_legs = buys.loc[day] * np.asarray(current_weights)
            net_legs = costs.net_of_commission(gross_legs)
            holdings = holdings + net_legs / day_prices
            cost_basis = cost_basis + net_legs
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
            "value": (shares * effective).sum(axis=1),
        }
    )
    frame.index.name = "date"
    return frame, weight_history


def _rebalance(
    holdings: np.ndarray,
    cost_basis: np.ndarray,
    day_prices: np.ndarray,
    target: np.ndarray,
    costs: Costs,
) -> tuple[np.ndarray, np.ndarray]:
    """Trade holdings to target weights, deducting commissions and CGT.

    Costs are computed from the ideal (pre-cost) trade sizes and deducted
    from the total before reallocating — a one-pass approximation that avoids
    solving for the exact post-cost trade fixed point.
    """
    values = holdings * day_prices
    total = values.sum()
    trades = total * target - values

    sell_values = np.maximum(-trades, 0.0)
    buy_values = np.maximum(trades, 0.0)
    commissions = sum(costs.commission_on(v) for v in sell_values + buy_values)

    safe_values = np.where(values > 0, values, 1.0)
    sold_fraction = np.clip(sell_values / safe_values, 0.0, 1.0)

    realized_gains = sold_fraction * (values - cost_basis)
    tax = costs.capital_gains_tax_pct * np.maximum(realized_gains, 0.0).sum()

    new_total = max(total - commissions - tax, 0.0)
    new_holdings = new_total * target / day_prices

    # Average-cost basis: reduce by the sold fraction, add net buys.
    new_values = new_holdings * day_prices
    net_buys = np.maximum(new_values - values * (1 - sold_fraction), 0.0)
    new_basis = cost_basis * (1 - sold_fraction) + net_buys
    return new_holdings, new_basis


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
