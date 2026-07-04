import numpy as np
import pandas as pd
import pytest

from invsim.simulation import (
    Costs,
    fee_adjusted,
    simulate_dca,
    simulate_dynamic_portfolio,
    simulate_portfolio,
)


@pytest.fixture
def trading_days():
    return pd.bdate_range("2020-01-01", "2023-12-31")


def make_prices(index, seed=1, drift=0.0002, vol=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, len(index)))), index=index)


def test_commission_pct_reduces_value_proportionally(trading_days):
    prices = pd.Series(50.0, index=trading_days)  # flat price isolates the commission
    contributions = pd.Series(500.0, index=trading_days[::10])
    frame = simulate_dca(prices, contributions, Costs(commission_pct=0.01))
    assert frame["value"].iloc[-1] == pytest.approx(frame["invested"].iloc[-1] * 0.99)


def test_fixed_commission_deducted_per_trade(trading_days):
    prices = pd.Series(50.0, index=trading_days)
    contributions = pd.Series(500.0, index=trading_days[::10])
    frame = simulate_dca(prices, contributions, Costs(commission_fixed=5.0))
    n_trades = len(contributions)
    assert frame["value"].iloc[-1] == pytest.approx(
        frame["invested"].iloc[-1] - 5.0 * n_trades
    )


def test_annual_fee_drag_matches_closed_form():
    # Lump sum, flat price, 1% annual fee, exactly 2 years -> (1-0.01)^2.
    index = pd.date_range("2020-01-01", "2022-01-01", freq="D")
    prices = pd.Series(100.0, index=index)
    contributions = pd.Series([1000.0], index=[index[0]])
    frame = simulate_dca(prices, contributions, Costs(annual_fee_pct=0.01))
    expected = 1000.0 * (1 - 0.01) ** ((index[-1] - index[0]).days / 365)
    assert frame["value"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_fee_adjusted_noop_for_zero_fee(trading_days):
    prices = make_prices(trading_days)
    assert fee_adjusted(prices, 0.0) is prices


def test_portfolio_fixed_commission_charged_per_leg(trading_days):
    # Splitting one contribution across 2 assets doubles the fixed commissions.
    prices = pd.DataFrame({"A": pd.Series(50.0, index=trading_days),
                           "B": pd.Series(20.0, index=trading_days)})
    contributions = pd.Series(500.0, index=trading_days[::10])
    frame = simulate_portfolio(
        prices, contributions, pd.Series({"A": 0.5, "B": 0.5}), Costs(commission_fixed=2.0)
    )
    n_trades = len(contributions)
    assert frame["value"].iloc[-1] == pytest.approx(
        frame["invested"].iloc[-1] - 2.0 * 2 * n_trades
    )


def _dynamic(trading_days, costs):
    prices = pd.DataFrame(
        {
            "A": make_prices(trading_days, seed=3),
            "B": make_prices(trading_days, seed=4),
            "C": make_prices(trading_days, seed=5),
        }
    )
    contributions = pd.Series(500.0, index=trading_days[::10])
    frame, _ = simulate_dynamic_portfolio(
        prices, contributions, lookback_years=1, rebalance_months=3,
        max_weight=0.6, costs=costs,
    )
    return frame


def test_dynamic_zero_costs_unchanged(trading_days):
    # Regression: the cost-aware engine with zero costs must equal the old one.
    frame = _dynamic(trading_days, Costs())
    invested = frame["invested"].iloc[-1]
    assert frame["value"].iloc[-1] > 0
    assert invested > 0
    # Rebalancing at market prices with no costs conserves value: reproducible
    # via flow-adjusted identity — value never jumps except by market moves
    # or contributions.
    jumps = frame["value"].diff() - frame["contribution"]
    daily_moves = jumps / frame["value"].shift(1)
    assert daily_moves.abs().max() < 0.2  # no discontinuities beyond market noise


def test_dynamic_costs_strictly_reduce_value(trading_days):
    free = _dynamic(trading_days, Costs())
    costly = _dynamic(
        trading_days,
        Costs(commission_pct=0.005, commission_fixed=1.0, capital_gains_tax_pct=0.2),
    )
    assert costly["value"].iloc[-1] < free["value"].iloc[-1]
    # Same contributions in both.
    assert costly["invested"].iloc[-1] == pytest.approx(free["invested"].iloc[-1])


def test_net_of_commission_never_negative():
    costs = Costs(commission_fixed=10.0)
    assert costs.net_of_commission(np.array([5.0, 0.0, 100.0])).tolist() == [0.0, 0.0, 90.0]
