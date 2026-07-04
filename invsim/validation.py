"""Walk-forward validation for dynamic-portfolio hyperparameters.

An in-sample grid search picks whatever won the *past* — classic overfitting.
This module does anchored walk-forward cross-validation instead:

- The evaluation span (everything after the longest warm-up) is split into
  ``folds + 1`` equal chunks; the first chunk extends the initial train span.
- For each fold: every parameter combination is scored on data strictly
  before the fold (train), the best one is selected, and only then evaluated
  on the unseen fold (test), alongside an equal-weight baseline.

The gap between train and test scores is the honest measure of how much of
the "optimal" parameters' edge is real versus curve-fit.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import performance_metrics
from .simulation import (
    NO_COSTS,
    Costs,
    investing_start,
    simulate_dynamic_portfolio,
    simulate_portfolio,
)


@dataclass(frozen=True)
class GridPoint:
    lookback_years: float
    rebalance_months: int
    max_change: float

    def as_dict(self) -> dict:
        return {
            "lookback": self.lookback_years,
            "rebalance": self.rebalance_months,
            "max_change": self.max_change,
        }


def build_grid(lookbacks, rebalances, max_changes) -> list[GridPoint]:
    return [
        GridPoint(lb, rb, mc)
        for lb, rb, mc in itertools.product(lookbacks, rebalances, max_changes)
    ]


def _score_segment(
    prices: pd.DataFrame,
    contributions: pd.Series,
    point: GridPoint,
    segment_start: pd.Timestamp,
    segment_end: pd.Timestamp,
    rf_daily: pd.Series,
    risk_free_rate: float,
    max_weight: float,
    costs: Costs,
    score_key: str,
) -> float | None:
    """Run the strategy so it invests only inside [segment_start, segment_end).

    Prices before the segment provide the lookback history; prices after the
    segment are excluded entirely.
    """
    history_start = segment_start - pd.DateOffset(years=point.lookback_years)
    window = prices.loc[(prices.index >= history_start) & (prices.index < segment_end)]
    flows = contributions[
        (contributions.index >= segment_start) & (contributions.index < segment_end)
    ]
    if len(flows) < 3:
        return None
    try:
        frame, _ = simulate_dynamic_portfolio(
            window,
            flows,
            lookback_years=point.lookback_years,
            rebalance_months=point.rebalance_months,
            max_weight=max_weight,
            max_change=point.max_change,
            risk_free_rate=risk_free_rate,
            costs=costs,
        )
    except ValueError:
        return None
    row = performance_metrics(frame["value"], frame["invested"], flows, rf_daily)
    return row.get(score_key) if row else None


def walk_forward_validate(
    prices: pd.DataFrame,
    contributions: pd.Series,
    grid: list[GridPoint],
    folds: int = 3,
    rf_daily: pd.Series | float = 0.02,
    risk_free_rate: float = 0.02,
    max_weight: float = 0.4,
    costs: Costs = NO_COSTS,
    score_key: str = "sharpe_ratio",
    log=print,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anchored walk-forward CV over the hyperparameter grid.

    Returns ``(fold_results, combo_scores)``:

    - ``fold_results``: one row per fold — the params chosen on train, their
      train and test scores, and the equal-weight baseline's test score.
    - ``combo_scores``: per-combination train/test scores per fold, for
      stability inspection.
    """
    if not isinstance(rf_daily, pd.Series):
        rf_daily = pd.Series(rf_daily, index=prices.index)

    max_lookback = max(p.lookback_years for p in grid)
    span_start = investing_start(prices.index, max_lookback)
    span_end = prices.index[-1]
    boundaries = pd.date_range(span_start, span_end, periods=folds + 2)
    # boundaries[0] .. [1] extend the initial train span; each subsequent
    # chunk is one test fold.

    equal_weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)
    fold_rows, combo_rows = [], []

    for fold in range(folds):
        test_start, test_end = boundaries[fold + 1], boundaries[fold + 2]
        train_scores: dict[GridPoint, float] = {}
        for point in grid:
            score = _score_segment(
                prices, contributions, point, span_start, test_start,
                rf_daily, risk_free_rate, max_weight, costs, score_key,
            )
            if score is not None:
                train_scores[point] = score
                combo_rows.append(
                    {**point.as_dict(), "fold": fold + 1, "phase": "train", score_key: score}
                )
        if not train_scores:
            log(f"fold {fold + 1}: no valid combination on train span, skipped")
            continue

        best = max(train_scores, key=train_scores.get)
        test_score = _score_segment(
            prices, contributions, best, test_start, test_end,
            rf_daily, risk_free_rate, max_weight, costs, score_key,
        )
        combo_rows.append(
            {**best.as_dict(), "fold": fold + 1, "phase": "test", score_key: test_score}
        )

        # Equal-weight baseline over the identical test window.
        flows = contributions[
            (contributions.index >= test_start) & (contributions.index < test_end)
        ]
        window = prices.loc[(prices.index >= test_start) & (prices.index < test_end)]
        baseline_score = None
        if len(flows) >= 3 and len(window) >= 60:
            frame = simulate_portfolio(window, flows, equal_weights, costs)
            row = performance_metrics(frame["value"], frame["invested"], flows, rf_daily)
            baseline_score = row.get(score_key) if row else None

        fold_rows.append(
            {
                "fold": fold + 1,
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                **{f"chosen_{k}": v for k, v in best.as_dict().items()},
                f"train_{score_key}": round(train_scores[best], 3),
                f"test_{score_key}": round(test_score, 3) if test_score is not None else None,
                f"equal_weight_{score_key}": (
                    round(baseline_score, 3) if baseline_score is not None else None
                ),
            }
        )
        log(
            f"fold {fold + 1}: chose lookback={best.lookback_years} "
            f"rebalance={best.rebalance_months} max_change={best.max_change:.2f} "
            f"| train {score_key}={train_scores[best]:.3f} "
            f"-> test {score_key}={test_score if test_score is not None else float('nan'):.3f} "
            f"(equal-weight {baseline_score if baseline_score is not None else float('nan'):.3f})"
        )

    return pd.DataFrame(fold_rows), pd.DataFrame(combo_rows)
