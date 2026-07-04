"""Named cost/tax profiles: jurisdiction + broker in one switch.

A profile is just a :class:`~invsim.simulation.Costs` plus a description.
Built-ins cover the common cases; a ``profiles.toml`` next to where you run
the CLI (or passed via ``--profiles-file``) adds or overrides profiles
without touching code:

    [ukraine-ibkr]                       # overrides the built-in
    capital_gains_tax_pct = 0.195        # e.g. if the military levy changes
    exit_tax_pct = 0.195

    [my-broker]                          # a brand-new profile
    description = "Local broker, 0.3% per trade, 10% CGT"
    commission_pct = 0.003
    capital_gains_tax_pct = 0.10
    exit_tax_pct = 0.10

All rates are fractions (0.001 = 0.1%). Fields are the ``Costs`` fields:
commission_pct, commission_fixed, commission_min, annual_fee_pct,
capital_gains_tax_pct, exit_tax_pct, fx_fee_pct, fx_fee_min.

The built-in numbers are honest approximations, not legal advice — check
your broker's pricing page and your tax code, then pin your own values in
``profiles.toml``.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

from .simulation import Costs

_COST_FIELDS = {f.name for f in dataclasses.fields(Costs)}

# IBKR "Fixed" pricing for US stocks/ETFs is $0.005/share with a $1 minimum
# per order; at DCA order sizes (a few hundred $ per leg) the $1 minimum is
# what you actually pay, so it is modeled as commission_min. IBKR currency
# conversion costs 0.002% with a $2 minimum — charged once per contribution
# for anyone earning in a non-USD currency.
_IBKR_FEES = {
    "commission_min": 1.0,
    "fx_fee_pct": 0.00002,
    "fx_fee_min": 2.0,
}

BUILTIN_PROFILES: dict[str, dict] = {
    "none": {
        "description": "No costs or taxes (raw market outcome)",
    },
    "ukraine-ibkr": {
        "description": (
            "Ukraine tax resident trading via IBKR: $1 min/order, $2 min FX, "
            "18% PIT + 5% military levy on realized gains (23% total)"
        ),
        **_IBKR_FEES,
        "capital_gains_tax_pct": 0.23,
        "exit_tax_pct": 0.23,
    },
    "cyprus-ibkr": {
        "description": (
            "Cyprus tax resident trading via IBKR: $1 min/order, $2 min FX, "
            "no capital gains tax on securities"
        ),
        **_IBKR_FEES,
    },
}


def load_profiles(profiles_file: Path | str | None = None) -> dict[str, dict]:
    """Built-in profiles merged with (and overridden by) the TOML file.

    When ``profiles_file`` is None, ``./profiles.toml`` is used if it exists.
    File entries merge field-by-field into same-named built-ins.
    """
    merged = {name: dict(spec) for name, spec in BUILTIN_PROFILES.items()}
    path = Path(profiles_file) if profiles_file else Path("profiles.toml")
    if not path.exists():
        if profiles_file:  # an explicitly requested file must exist
            raise FileNotFoundError(f"Profiles file not found: {path}")
        return merged

    with path.open("rb") as fh:
        user_profiles = tomllib.load(fh)
    for name, spec in user_profiles.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Profile [{name}] must be a table of fields")
        unknown = set(spec) - _COST_FIELDS - {"description"}
        if unknown:
            raise ValueError(f"Profile [{name}] has unknown fields: {sorted(unknown)}")
        merged.setdefault(name, {})
        merged[name].update(spec)
        merged[name].setdefault("description", f"User profile from {path.name}")
    return merged


def get_costs(name: str, profiles_file: Path | str | None = None) -> Costs:
    """Resolve a profile name to a ``Costs`` instance."""
    profiles = load_profiles(profiles_file)
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown profile {name!r}. Available: {available}")
    spec = {k: v for k, v in profiles[name].items() if k in _COST_FIELDS}
    return Costs(**spec)


def describe_profiles(profiles_file: Path | str | None = None) -> str:
    """Human-readable listing of all available profiles."""
    lines = []
    for name, spec in sorted(load_profiles(profiles_file).items()):
        lines.append(f"{name}")
        lines.append(f"  {spec.get('description', '')}")
        fields = {k: v for k, v in spec.items() if k in _COST_FIELDS and v}
        if fields:
            lines.append(
                "  " + ", ".join(f"{k}={v}" for k, v in sorted(fields.items()))
            )
        lines.append("")
    return "\n".join(lines).rstrip()
