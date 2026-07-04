import pandas as pd
import pytest

from invsim.schedule import align_to_trading_days, biweekly_paydays, contribution_schedule


def test_paydays_are_fridays_14_days_apart():
    paydays = biweekly_paydays(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
    assert len(paydays) > 20
    assert all(d.weekday() == 4 for d in paydays)
    assert (paydays[1:] - paydays[:-1] == pd.Timedelta(days=14)).all()


def test_paydays_empty_range():
    assert len(biweekly_paydays(pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03"))) == 0


def test_alignment_moves_forward_only():
    trading_days = pd.DatetimeIndex(["2023-01-03", "2023-01-04", "2023-01-09"])
    paydays = pd.DatetimeIndex(["2023-01-02", "2023-01-06"])  # Mon holiday, Fri holiday
    aligned = align_to_trading_days(paydays, trading_days)
    assert list(aligned) == [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-09")]


def test_alignment_drops_paydays_after_last_session():
    trading_days = pd.DatetimeIndex(["2023-01-03"])
    paydays = pd.DatetimeIndex(["2023-01-02", "2023-06-01"])
    aligned = align_to_trading_days(paydays, trading_days)
    assert list(aligned) == [pd.Timestamp("2023-01-03")]


def test_contributions_on_same_day_accumulate():
    # Two paydays collapsing onto one trading day must sum, not vanish.
    trading_days = pd.DatetimeIndex(["2023-01-06", "2023-02-17"])
    schedule = contribution_schedule(trading_days, 500.0,
                                     start="2023-01-01", end="2023-02-28")
    assert schedule.sum() == pytest.approx(500.0 * 4)  # 4 biweekly paydays in range
    assert schedule.loc[pd.Timestamp("2023-02-17")] > 500  # collapsed contributions


def test_contribution_schedule_regular_weeks():
    trading_days = pd.bdate_range("2022-01-01", "2022-12-31")
    schedule = contribution_schedule(trading_days, 100.0)
    assert (schedule == 100.0).all()
    assert len(schedule) == 26  # 52 weeks / 2
