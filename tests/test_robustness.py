import numpy as np
import pandas as pd
import pytest

from invsim.robustness import rolling_dca, summarize_windows, window_bounds


@pytest.fixture
def ten_years():
    return pd.bdate_range("2010-01-01", "2019-12-31")


def monthly(series_start, series_end, value):
    idx = pd.date_range(series_start, series_end, freq="MS")
    return pd.Series(value, index=idx)


def test_window_bounds_count_and_span(ten_years):
    bounds = window_bounds(ten_years, window_years=5, step_months=6)
    # Starts every 6 months from 2010-01-01 while a full 5-year window fits
    # (last valid start is 2014-12-31) -> 10 windows.
    assert len(bounds) == 10
    for start, end in bounds:
        assert end == start + pd.DateOffset(years=5)
        assert end <= ten_years[-1] + pd.Timedelta(days=1)


def test_deterministic_growth_gives_identical_xirr(ten_years):
    # Pure exponential price: every window must see the same annual return.
    prices = pd.Series(100 * 1.0003 ** np.arange(len(ten_years)), index=ten_years)
    tbill = monthly("2010-01-01", "2019-12-01", 0.0)
    cpi = monthly("2010-01-01", "2019-12-01", 100.0)
    windows = rolling_dca(prices, tbill, cpi, window_years=3, amount=500, step_months=12)
    assert len(windows) >= 6
    xirr_spread = windows["xirr_pct"].max() - windows["xirr_pct"].min()
    assert xirr_spread < 1.0  # all windows nearly identical
    assert windows["beat_tbills"].all()      # positive drift vs 0% t-bills
    assert windows["beat_inflation"].all()   # flat CPI
    assert (windows["max_drawdown_pct"] == 0).all()


def test_declining_asset_loses_to_benchmarks(ten_years):
    prices = pd.Series(100 * 0.9997 ** np.arange(len(ten_years)), index=ten_years)
    tbill = monthly("2010-01-01", "2019-12-01", 0.02)
    cpi = monthly("2010-01-01", "2019-12-01", 100.0)
    windows = rolling_dca(prices, tbill, cpi, window_years=3, amount=500, step_months=12)
    assert not windows["beat_tbills"].any()
    assert (windows["xirr_pct"] < 0).all()


def test_summarize_windows(ten_years):
    prices = pd.Series(100 * 1.0003 ** np.arange(len(ten_years)), index=ten_years)
    tbill = monthly("2010-01-01", "2019-12-01", 0.0)
    cpi = monthly("2010-01-01", "2019-12-01", 100.0)
    windows = rolling_dca(prices, tbill, cpi, window_years=3, amount=500, step_months=12)
    summary = summarize_windows(windows, label="X")
    assert summary["label"] == "X"
    assert summary["windows"] == len(windows)
    assert summary["beat_tbills_pct"] == 100.0
    assert summary["worst_xirr_pct"] <= summary["median_xirr_pct"] <= summary["best_xirr_pct"]


def test_summarize_empty():
    assert summarize_windows(pd.DataFrame()) == {}
