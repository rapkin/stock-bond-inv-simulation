import numpy as np
import pandas as pd
import pytest

from invsim.optimize import optimize_weights, walk_forward_weights


def make_returns(n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "GOOD": rng.normal(0.001, 0.01, n_days),   # high Sharpe
            "BAD": rng.normal(-0.0005, 0.02, n_days),  # negative drift, noisy
            "MEH": rng.normal(0.0002, 0.01, n_days),
        },
        index=index,
    )


def test_weights_sum_to_one_and_respect_bounds():
    weights = optimize_weights(make_returns(), max_weight=0.5)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= -1e-9).all()
    assert (weights <= 0.5 + 1e-9).all()


def test_optimizer_prefers_high_sharpe_asset():
    weights = optimize_weights(make_returns(), max_weight=1.0)
    assert weights["GOOD"] > weights["BAD"]
    assert weights["GOOD"] > 0.5


def test_infeasible_constraints_raise():
    # 2 assets with max_weight 0.4 cannot sum to 1 — legacy silently returned garbage.
    returns = make_returns()[["GOOD", "BAD"]]
    with pytest.raises(ValueError, match="Infeasible"):
        optimize_weights(returns, max_weight=0.4)


def test_single_asset_gets_full_weight():
    weights = optimize_weights(make_returns()[["GOOD"]])
    assert weights.tolist() == [1.0]


def test_turnover_limit_respected():
    returns = make_returns()
    prev = pd.Series({"GOOD": 0.34, "BAD": 0.33, "MEH": 0.33})
    weights = optimize_weights(returns, max_weight=1.0, prev_weights=prev, max_change=0.05)
    assert ((weights - prev).abs() <= 0.05 + 1e-9).all()


def test_walk_forward_uses_only_past_data():
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2018-01-01", "2023-12-31")
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(index)))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, len(index)))),
        },
        index=index,
    )
    rebalance_days = pd.DatetimeIndex(["2020-06-01", "2021-06-01", "2022-06-01"])
    history = walk_forward_weights(prices, rebalance_days, lookback_years=2, max_weight=1.0)
    assert len(history) == 3
    assert (history.sum(axis=1) - 1).abs().max() < 1e-6

    # Corrupting all data on/after a rebalance date must not change the
    # weights computed for that date (proof there is no look-ahead).
    corrupted = prices.copy()
    corrupted.loc[corrupted.index >= "2022-06-01"] = 1e9
    history2 = walk_forward_weights(corrupted, rebalance_days, lookback_years=2, max_weight=1.0)
    pd.testing.assert_frame_equal(history, history2)
