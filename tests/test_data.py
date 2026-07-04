import pandas as pd
import pytest

from invsim.data import MarketData, align_monthly


class RecordingMarketData(MarketData):
    """Test double: counts fetches via a stub fetch function."""


def make_year_series(year: int) -> pd.Series:
    index = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="MS")
    return pd.Series(float(year), index=index)


def test_completed_years_are_cached(tmp_path):
    market = MarketData(cache_dir=tmp_path)
    calls = []

    def fetch(year):
        calls.append(year)
        return make_year_series(year)

    first = market._load_yearly("test", "X", 2020, 2022, fetch)
    assert calls == [2020, 2021, 2022]
    assert len(first) == 36

    calls.clear()
    second = market._load_yearly("test", "X", 2020, 2022, fetch)
    assert calls == []  # all served from cache
    pd.testing.assert_series_equal(first, second)


def test_current_year_never_cached(tmp_path):
    from datetime import date

    current = date.today().year
    market = MarketData(cache_dir=tmp_path)
    calls = []

    def fetch(year):
        calls.append(year)
        return make_year_series(year)

    market._load_yearly("test", "X", current, current, fetch)
    market._load_yearly("test", "X", current, current, fetch)
    assert calls == [current, current]  # refetched both times
    assert not list(tmp_path.rglob("*.pkl"))


def test_duplicate_index_entries_deduped(tmp_path):
    market = MarketData(cache_dir=tmp_path)

    def fetch(year):
        # Overlapping boundary observation on Jan 1.
        index = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="MS")
        return pd.Series(1.0, index=index)

    merged = market._load_yearly("test", "X", 2020, 2021, fetch)
    assert merged.index.is_unique
    assert merged.index.is_monotonic_increasing


def test_align_monthly_forward_fills():
    monthly = pd.Series(
        [1.0, 2.0], index=pd.DatetimeIndex(["2023-01-01", "2023-02-01"])
    )
    daily = pd.DatetimeIndex(["2023-01-15", "2023-02-01", "2023-03-10"])
    aligned = align_monthly(monthly, daily)
    assert aligned.tolist() == [1.0, 2.0, 2.0]
    assert list(aligned.index) == list(daily)
