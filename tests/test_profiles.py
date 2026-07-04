import numpy as np
import pandas as pd
import pytest

from invsim.cli import build_parser, _costs_from
from invsim.profiles import BUILTIN_PROFILES, describe_profiles, get_costs, load_profiles
from invsim.simulation import Costs, simulate_dca


def test_builtin_profiles_resolve():
    assert get_costs("none") == Costs()
    ua = get_costs("ukraine-ibkr")
    assert ua.capital_gains_tax_pct == 0.23
    assert ua.exit_tax_pct == 0.23
    assert ua.commission_min == 1.0
    assert ua.fx_fee_min == 2.0
    cy = get_costs("cyprus-ibkr")
    assert cy.capital_gains_tax_pct == 0.0
    assert cy.commission_min == 1.0


def test_unknown_profile_lists_available():
    with pytest.raises(KeyError, match="ukraine-ibkr"):
        get_costs("mars-ibkr")


def test_toml_overrides_builtin_field_by_field(tmp_path):
    config = tmp_path / "profiles.toml"
    config.write_text('[ukraine-ibkr]\ncapital_gains_tax_pct = 0.195\n')
    costs = get_costs("ukraine-ibkr", config)
    assert costs.capital_gains_tax_pct == 0.195
    assert costs.commission_min == 1.0  # other built-in fields kept


def test_toml_adds_new_profile(tmp_path):
    config = tmp_path / "profiles.toml"
    config.write_text(
        '[my-broker]\ndescription = "test"\ncommission_pct = 0.003\nexit_tax_pct = 0.1\n'
    )
    costs = get_costs("my-broker", config)
    assert costs.commission_pct == 0.003
    assert costs.exit_tax_pct == 0.1
    assert "my-broker" in describe_profiles(config)


def test_toml_unknown_field_rejected(tmp_path):
    config = tmp_path / "profiles.toml"
    config.write_text('[bad]\nnot_a_field = 1\n')
    with pytest.raises(ValueError, match="unknown fields"):
        load_profiles(config)


def test_missing_explicit_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_profiles(tmp_path / "nope.toml")


def test_cli_profile_with_flag_override():
    parser = build_parser()
    args = parser.parse_args(
        ["simulate", "-t", "QQQ", "--profile", "ukraine-ibkr", "--cgt", "0.10"]
    )
    costs = _costs_from(args)
    assert costs.capital_gains_tax_pct == 0.10  # flag wins
    assert costs.exit_tax_pct == 0.23           # rest from profile
    assert costs.commission_min == 1.0


def test_commission_min_dominates_small_orders():
    days = pd.bdate_range("2020-01-01", "2021-12-31")
    prices = pd.Series(50.0, index=days)
    contributions = pd.Series(500.0, index=days[::10])
    costs = Costs(commission_pct=0.00001, commission_min=1.0)
    frame = simulate_dca(prices, contributions, costs)
    n = len(contributions)
    assert frame["value"].iloc[-1] == pytest.approx(frame["invested"].iloc[-1] - n * 1.0, rel=1e-6)


def test_fx_fee_min_per_contribution():
    days = pd.bdate_range("2020-01-01", "2021-12-31")
    prices = pd.Series(50.0, index=days)
    contributions = pd.Series(500.0, index=days[::10])
    costs = Costs(fx_fee_pct=0.00002, fx_fee_min=2.0)
    frame = simulate_dca(prices, contributions, costs)
    n = len(contributions)  # $2 min beats 0.002% of $500
    assert frame["value"].iloc[-1] == pytest.approx(frame["invested"].iloc[-1] - n * 2.0)


def test_after_exit_tax():
    costs = Costs(exit_tax_pct=0.23)
    assert costs.after_exit_tax(200_000, 100_000) == pytest.approx(200_000 - 23_000)
    assert costs.after_exit_tax(80_000, 100_000) == 80_000  # no tax on losses


def test_all_builtin_specs_are_valid_costs_fields():
    for name in BUILTIN_PROFILES:
        assert isinstance(get_costs(name), Costs)
