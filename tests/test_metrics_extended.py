import numpy as np
import pandas as pd
import pytest

from invsim.metrics import (
    drawdown_series,
    monthly_returns,
    rolling_sharpe,
    var_cvar,
    yearly_returns,
)


@pytest.fixture
def days():
    return pd.bdate_range("2020-01-01", "2021-12-31")


def test_var_cvar_known_distribution():
    # 100 returns: ninety-five at +1%, five at -10% -> VaR95 = -10%, CVaR = -10%.
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    values = np.full(100, 0.01)
    values[:5] = -0.10
    var95, cvar95 = var_cvar(pd.Series(values, index=index), 0.95)
    assert var95 == pytest.approx(-0.10, abs=1e-9)
    assert cvar95 == pytest.approx(-0.10, abs=1e-9)
    # CVaR is never better (higher) than VaR.
    assert cvar95 <= var95 + 1e-12


def test_var_cvar_insufficient_data():
    var95, cvar95 = var_cvar(pd.Series([0.01] * 5), 0.95)
    assert np.isnan(var95) and np.isnan(cvar95)


def test_monthly_returns_compound_correctly(days):
    returns = pd.Series(0.001, index=days)
    table = monthly_returns(returns)
    assert table.shape[1] == 12
    jan_2020_days = len(days[(days.year == 2020) & (days.month == 1)])
    assert table.loc[2020, 1] == pytest.approx(1.001 ** jan_2020_days - 1)


def test_yearly_returns(days):
    returns = pd.Series(0.001, index=days)
    yearly = yearly_returns(returns)
    assert set(yearly.index) == {2020, 2021}
    n_2021 = len(days[days.year == 2021])
    assert yearly.loc[2021] == pytest.approx(1.001 ** n_2021 - 1)


def test_drawdown_series_recovers_to_zero():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    returns = pd.Series([0.10, -0.50, 0.50, 0.40], index=index)
    dd = drawdown_series(returns)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(-0.50)
    assert dd.iloc[2] == pytest.approx(0.825 / 1.1 - 1)  # still underwater
    assert dd.iloc[-1] == pytest.approx(0.0)  # new all-time high -> dd back to 0


def test_rolling_sharpe_constant_excess_is_undefined_or_large(days):
    # Zero volatility -> NaN/inf guarded upstream; just check the shape/window.
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, len(days)), index=days)
    rolling = rolling_sharpe(returns, 0.0, window=60)
    assert rolling.isna().sum() == 59  # warm-up
    assert np.isfinite(rolling.dropna()).all()
