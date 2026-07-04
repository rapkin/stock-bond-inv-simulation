"""Contribution schedules: biweekly paydays aligned to trading days."""

from __future__ import annotations

import numpy as np
import pandas as pd

FRIDAY = 4  # datetime.weekday()


def biweekly_paydays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Every second Friday from the first Friday on/after ``start`` to ``end``."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    first_friday = start + pd.Timedelta(days=(FRIDAY - start.weekday()) % 7)
    if first_friday > end:
        return pd.DatetimeIndex([])
    return pd.date_range(first_friday, end, freq="14D")


def align_to_trading_days(
    paydays: pd.DatetimeIndex, trading_days: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Map each payday to the next trading day on/after it.

    Buying on the *next* available session (never a previous one) avoids
    look-ahead: on the payday the money exists, so it waits for the market to
    open. Paydays falling after the last trading day are dropped.
    """
    positions = np.searchsorted(trading_days.values, paydays.values, side="left")
    positions = positions[positions < len(trading_days)]
    return pd.DatetimeIndex(trading_days.values[positions])


def contribution_schedule(
    trading_days: pd.DatetimeIndex,
    amount: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Dollar contributions indexed by execution (trading) day.

    If two paydays land on the same trading day (e.g. around a market halt),
    the contributions add up rather than being silently dropped.
    """
    start = pd.Timestamp(start) if start is not None else trading_days[0]
    end = pd.Timestamp(end) if end is not None else trading_days[-1]
    buy_days = align_to_trading_days(biweekly_paydays(start, end), trading_days)
    return pd.Series(amount, index=buy_days).groupby(level=0).sum()
