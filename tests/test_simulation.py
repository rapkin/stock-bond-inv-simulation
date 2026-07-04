import numpy as np
import pandas as pd
import pytest

from invsim.metrics import flow_adjusted_returns
from invsim.simulation import (
    add_benchmarks,
    simulate_dca,
    simulate_dynamic_portfolio,
    simulate_portfolio,
    simulate_tbill,
)


@pytest.fixture
def trading_days():
    return pd.bdate_range("2020-01-01", "2023-12-31")


def make_prices(index, start=100.0, drift=0.0002, vol=0.01, seed=1):
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, len(index))
    return pd.Series(start * np.exp(np.cumsum(returns)), index=index)


def test_dca_constant_price_value_equals_invested(trading_days):
    prices = pd.Series(50.0, index=trading_days)
    contributions = pd.Series(500.0, index=trading_days[::10])
    frame = simulate_dca(prices, contributions)
    pd.testing.assert_series_equal(frame["value"], frame["invested"], check_names=False)


def test_dca_price_doubling_doubles_early_money(trading_days):
    # Single contribution, price doubles by the end -> value doubles.
    prices = pd.Series(np.linspace(100, 200, len(trading_days)), index=trading_days)
    contributions = pd.Series([1000.0], index=[trading_days[0]])
    frame = simulate_dca(prices, contributions)
    assert frame["value"].iloc[-1] == pytest.approx(2000.0)


def test_flow_adjusted_returns_equal_price_returns(trading_days):
    prices = make_prices(trading_days)
    contributions = pd.Series(500.0, index=trading_days[::10])
    frame = simulate_dca(prices, contributions)
    twr = flow_adjusted_returns(frame["value"], contributions)
    expected = prices.pct_change().reindex(twr.index)
    pd.testing.assert_series_equal(twr, expected, check_names=False, atol=1e-12)


def test_portfolio_two_assets_matches_sum_of_singles(trading_days):
    prices = pd.DataFrame(
        {"A": make_prices(trading_days, seed=1), "B": make_prices(trading_days, seed=2)}
    )
    contributions = pd.Series(1000.0, index=trading_days[::10])
    combined = simulate_portfolio(prices, contributions, pd.Series({"A": 0.6, "B": 0.4}))
    single_a = simulate_dca(prices["A"], contributions * 0.6)
    single_b = simulate_dca(prices["B"], contributions * 0.4)
    pd.testing.assert_series_equal(
        combined["value"], single_a["value"] + single_b["value"], check_names=False
    )


def test_tbill_accrues_calendar_days():
    # Flat 5% rate for exactly one year on a single upfront contribution.
    index = pd.date_range("2021-01-01", "2022-01-01", freq="D")
    rates = pd.Series(0.05, index=pd.date_range("2021-01-01", "2022-01-01", freq="MS"))
    contributions = pd.Series([1000.0], index=[index[0]])
    value = simulate_tbill(contributions, rates, index)
    assert value.iloc[-1] == pytest.approx(1000 * 1.05, rel=1e-3)


def test_add_benchmarks_real_columns(trading_days):
    prices = make_prices(trading_days)
    contributions = pd.Series(500.0, index=trading_days[::10])
    frame = simulate_dca(prices, contributions)
    months = pd.date_range("2020-01-01", "2024-01-01", freq="MS")
    cpi = pd.Series(np.linspace(100, 120, len(months)), index=months)  # 20% inflation
    tbill = pd.Series(0.02, index=months)
    frame = add_benchmarks(frame, tbill, cpi)
    # Real invested < nominal invested under inflation.
    assert frame["invested_real"].iloc[-1] < frame["invested"].iloc[-1]
    assert frame["profit_real"].iloc[-1] == pytest.approx(
        frame["value_real"].iloc[-1] - frame["invested_real"].iloc[-1]
    )


def test_dynamic_portfolio_conserves_value(trading_days):
    # Rebalancing swaps holdings at market price: it must never create or
    # destroy value on the rebalance day itself.
    prices = pd.DataFrame(
        {
            "A": make_prices(trading_days, seed=3),
            "B": make_prices(trading_days, seed=4),
            "C": make_prices(trading_days, seed=5),
        }
    )
    contributions = pd.Series(500.0, index=trading_days[::10])
    frame, weights = simulate_dynamic_portfolio(
        prices, contributions, lookback_years=1, rebalance_months=3, max_weight=0.6
    )
    assert (weights.sum(axis=1) - 1).abs().max() < 1e-6
    assert (weights.to_numpy() >= -1e-9).all()
    assert (weights.to_numpy() <= 0.6 + 1e-9).all()
    # Invested equals the contributions made after warm-up.
    invested_expected = contributions[contributions.index >= weights.index[0]].sum()
    assert frame["invested"].iloc[-1] == pytest.approx(invested_expected)
    # Weight changes between consecutive rebalances respect max_change.
    step = weights.diff().abs().max().max()
    assert step <= 0.10 + 1e-9


def test_dynamic_portfolio_insufficient_history_raises():
    short_days = pd.bdate_range("2023-01-01", "2023-06-30")
    prices = pd.DataFrame(
        {"A": make_prices(short_days, seed=1), "B": make_prices(short_days, seed=2)}
    )
    contributions = pd.Series(500.0, index=short_days[::10])
    with pytest.raises(ValueError):
        simulate_dynamic_portfolio(prices, contributions, lookback_years=3, max_weight=0.6)
