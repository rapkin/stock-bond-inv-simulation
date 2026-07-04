"""Market data access with per-year on-disk caching.

Data sources:
- Prices: Yahoo Finance (adjusted close — includes splits and dividends).
- CPI (CPIAUCSL) and 3-month T-bill rate (TB3MS): FRED, stored raw (monthly);
  interpolation/alignment happens at the point of use, never in the cache.

Caching rules:
- One pickle per (kind, key, calendar year) under ``cache/<kind>/``.
- Only completed calendar years are cached; the current year is always
  fetched fresh so partial data never goes stale on disk.
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# Index shorthands accepted on the CLI, mapped to Yahoo Finance symbols.
TICKER_ALIASES = {
    "GSPC": "^GSPC",
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "DJI": "^DJI",
    "IXIC": "^IXIC",
}


class MarketData:
    """Fetches and caches prices, CPI, and T-bill rates."""

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    # --- generic yearly cache -------------------------------------------------

    def _cache_path(self, kind: str, key: str, year: int) -> Path:
        return self.cache_dir / kind / f"{key}_{year}.pkl"

    def _load_yearly(
        self,
        kind: str,
        key: str,
        start_year: int,
        end_year: int,
        fetch_year: Callable[[int], pd.Series],
    ) -> pd.Series:
        parts: list[pd.Series] = []
        for year in range(start_year, end_year + 1):
            path = self._cache_path(kind, key, year)
            year_is_complete = year < date.today().year
            if year_is_complete and path.exists():
                parts.append(pd.read_pickle(path))
                continue
            data = fetch_year(year)
            if year_is_complete and not data.empty:
                path.parent.mkdir(parents=True, exist_ok=True)
                data.to_pickle(path)
            parts.append(data)

        parts = [p for p in parts if p is not None and not p.empty]
        if not parts:
            return pd.Series(dtype=float)
        merged = pd.concat(parts).sort_index()
        return merged[~merged.index.duplicated(keep="first")]

    # --- prices ---------------------------------------------------------------

    def prices(self, ticker: str, start_year: int, end_year: int) -> pd.Series:
        """Daily adjusted close prices for one ticker."""
        import yfinance as yf

        yf_symbol = TICKER_ALIASES.get(ticker.upper(), ticker)
        cache_key = ticker.upper().replace("^", "").replace("-", "_")

        def fetch_year(year: int) -> pd.Series:
            # yf.download's `end` is exclusive: ask for Jan 1 of the next year
            # so Dec 31 is included.
            raw = yf.download(
                yf_symbol,
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
                progress=False,
                auto_adjust=True,
            )
            if raw is None or raw.empty:
                return pd.Series(dtype=float)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"].dropna()
            return close[close > 0]

        series = self._load_yearly("prices", cache_key, start_year, end_year, fetch_year)
        if series.empty:
            raise ValueError(f"No price data for {ticker!r} ({start_year}-{end_year})")
        series.name = ticker
        return series

    def price_matrix(self, tickers: list[str], start_year: int, end_year: int) -> pd.DataFrame:
        """Prices for several tickers on their common trading days."""
        columns = [self.prices(t, start_year, end_year) for t in tickers]
        return pd.concat(columns, axis=1, join="inner").dropna()

    # --- FRED series ----------------------------------------------------------

    def cpi(self, start_year: int, end_year: int) -> pd.Series:
        """Monthly CPI (CPIAUCSL). Synthetic 3%/year fallback if FRED is down."""

        def fetch_year(year: int) -> pd.Series:
            try:
                from pandas_datareader import data as pdr

                raw = pdr.DataReader("CPIAUCSL", "fred", f"{year}-01-01", f"{year}-12-31")
                return raw["CPIAUCSL"].dropna()
            except Exception as exc:  # network/API failure
                warnings.warn(f"FRED CPI unavailable for {year} ({exc}); using synthetic 3%/yr")
                idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="MS")
                # Anchored to a fixed epoch so consecutive fallback years line up.
                epoch = pd.Timestamp("1990-01-01")
                return pd.Series(
                    [100 * 1.03 ** ((d - epoch).days / 365.25) for d in idx], index=idx
                )

        return self._load_yearly("cpi", "CPIAUCSL", start_year, end_year, fetch_year)

    def tbill_rate(self, start_year: int, end_year: int) -> pd.Series:
        """Monthly 3-month T-bill annual rate (TB3MS) as a decimal fraction."""

        def fetch_year(year: int) -> pd.Series:
            try:
                from pandas_datareader import data as pdr

                raw = pdr.DataReader("TB3MS", "fred", f"{year}-01-01", f"{year}-12-31")
                return raw["TB3MS"].dropna() / 100.0
            except Exception as exc:
                warnings.warn(f"FRED T-bill unavailable for {year} ({exc}); using flat 2%")
                idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="MS")
                return pd.Series(0.02, index=idx)

        return self._load_yearly("tbill", "TB3MS", start_year, end_year, fetch_year)


def align_monthly(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Align a monthly series onto a daily index by forward-fill (as-of join)."""
    combined = series.reindex(series.index.union(index)).ffill().bfill()
    return combined.reindex(index)
