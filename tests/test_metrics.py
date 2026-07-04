import numpy as np
import pandas as pd
import pytest

from invsim.metrics import (
    max_drawdown,
    performance_metrics,
    risk_reward_score,
    xirr,
)


def test_xirr_single_year_lump_sum():
    flows = pd.Series(
        [-1000.0, 1100.0],
        index=pd.DatetimeIndex(["2020-01-01", "2021-01-01"]),
    )
    assert xirr(flows) == pytest.approx(0.10, abs=1e-3)


def test_xirr_two_installments():
    # $100 now and $100 in 6 months, $210 after a year -> IRR ~ 6.6%.
    flows = pd.Series(
        [-100.0, -100.0, 210.0],
        index=pd.DatetimeIndex(["2020-01-01", "2020-07-01", "2021-01-01"]),
    )
    rate = xirr(flows)
    # NPV at the found rate must be ~0.
    t0 = flows.index[0]
    years = np.array([(d - t0).days / 365.25 for d in flows.index])
    assert np.sum(flows.values / (1 + rate) ** years) == pytest.approx(0, abs=1e-6)
    assert 0.05 < rate < 0.08


def test_max_drawdown_known_series():
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    wealth = pd.Series([100, 120, 60, 90, 130], index=index, dtype=float)
    mdd, underwater_days = max_drawdown(wealth)
    assert mdd == pytest.approx(-0.5)
    assert underwater_days == 2  # underwater on days 3-4, recovered on day 5


def test_risk_reward_score_matches_legacy_formula():
    # sortino=2, dd=-19%, return=+80% -> 2 * sqrt(0.81) * sqrt(0.8)
    expected = 2 * np.sqrt(0.81) * np.sqrt(0.8)
    assert risk_reward_score(2.0, -0.19, 0.8) == pytest.approx(expected)
    # Suspicion penalties
    assert risk_reward_score(8.0, -0.19, 0.8) == pytest.approx(
        8 * np.sqrt(0.81) * np.sqrt(0.8) * 0.7
    )
    assert risk_reward_score(2.0, -0.10, 0.8) == pytest.approx(
        2 * np.sqrt(0.90) * np.sqrt(0.8) * 0.7
    )


def test_performance_metrics_contributions_do_not_inflate_returns():
    """The legacy bug: cash inflows counted as returns. A flat-price asset
    with regular contributions must show ~zero return and volatility."""
    days = pd.bdate_range("2020-01-01", "2022-12-31")
    value_flat = pd.Series(0.0, index=days)
    contributions = pd.Series(500.0, index=days[::10])
    invested = contributions.reindex(days, fill_value=0).cumsum()
    value_flat = invested.copy()  # price never moves -> value == invested

    row = performance_metrics(value_flat, invested, contributions, rf_annual=0.0)
    assert row["annual_volatility_pct"] == pytest.approx(0.0, abs=1e-9)
    assert row["cagr_pct"] == pytest.approx(0.0, abs=1e-9)
    assert row["total_return_pct"] == pytest.approx(0.0, abs=1e-9)
    assert row["xirr_pct"] == pytest.approx(0.0, abs=1e-6)


def test_performance_metrics_sane_on_trending_asset():
    days = pd.bdate_range("2020-01-01", "2023-12-31")
    prices = pd.Series(100 * 1.0003 ** np.arange(len(days)), index=days)
    contributions = pd.Series(500.0, index=days[::10])
    flows = contributions.reindex(days, fill_value=0)
    shares = (flows / prices).cumsum()
    value = shares * prices
    invested = flows.cumsum()

    row = performance_metrics(value, invested, contributions, rf_annual=0.02, label="X")
    assert row["label"] == "X"
    assert row["total_return_pct"] > 0
    assert row["xirr_pct"] > 0
    assert row["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-9)
    # Deterministic exponential growth: ~7.85%/yr
    assert row["cagr_pct"] == pytest.approx(100 * (1.0003 ** 252 - 1), rel=0.05)
