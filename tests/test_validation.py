import numpy as np
import pandas as pd
import pytest

from invsim.validation import GridPoint, build_grid, walk_forward_validate


def make_prices(seed=11, start="2016-01-01", end="2023-12-31"):
    index = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, len(index)))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.015, len(index)))),
            "C": 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(index)))),
        },
        index=index,
    )


def contributions_for(prices):
    return pd.Series(500.0, index=prices.index[::10])


GRID = build_grid([1, 2], [3], [0.10, 0.20])


def test_build_grid_cartesian():
    assert len(GRID) == 4
    assert GRID[0] == GridPoint(1, 3, 0.10)


def test_walk_forward_produces_folds_and_scores():
    prices = make_prices()
    folds, combos = walk_forward_validate(
        prices, contributions_for(prices), GRID, folds=2, max_weight=0.6, log=lambda *_: None
    )
    assert len(folds) == 2
    assert set(folds.columns) >= {
        "fold", "test_start", "test_end", "chosen_lookback",
        "train_sharpe_ratio", "test_sharpe_ratio", "equal_weight_sharpe_ratio",
    }
    # Test windows are consecutive and non-overlapping.
    assert folds["test_end"].iloc[0] == folds["test_start"].iloc[1]
    # Every fold's train scores cover the grid (all combos valid on this data).
    train_rows = combos[combos["phase"] == "train"]
    assert len(train_rows) == len(GRID) * 2


def test_chosen_params_maximize_train_score():
    prices = make_prices()
    folds, combos = walk_forward_validate(
        prices, contributions_for(prices), GRID, folds=2, max_weight=0.6, log=lambda *_: None
    )
    for _, fold_row in folds.iterrows():
        train = combos[(combos["fold"] == fold_row["fold"]) & (combos["phase"] == "train")]
        best = train.loc[train["sharpe_ratio"].idxmax()]
        assert fold_row["chosen_lookback"] == best["lookback"]
        assert fold_row["chosen_rebalance"] == best["rebalance"]
        assert fold_row["chosen_max_change"] == best["max_change"]
        assert fold_row["train_sharpe_ratio"] == pytest.approx(best["sharpe_ratio"], abs=1e-3)


def test_no_leakage_from_test_data_into_selection():
    """Corrupting data inside the FINAL test fold must not change which
    parameters earlier folds choose (their selection uses only prior data)."""
    prices = make_prices()
    flows = contributions_for(prices)
    folds_clean, _ = walk_forward_validate(
        prices, flows, GRID, folds=2, max_weight=0.6, log=lambda *_: None
    )
    corrupted = prices.copy()
    last_test_start = pd.Timestamp(folds_clean["test_start"].iloc[-1])
    corrupted.loc[corrupted.index >= last_test_start] *= np.linspace(
        1, 5, (corrupted.index >= last_test_start).sum()
    )[:, None]
    folds_corrupt, _ = walk_forward_validate(
        corrupted, flows, GRID, folds=2, max_weight=0.6, log=lambda *_: None
    )
    chosen_cols = ["chosen_lookback", "chosen_rebalance", "chosen_max_change",
                   "train_sharpe_ratio"]
    pd.testing.assert_frame_equal(
        folds_clean[chosen_cols], folds_corrupt[chosen_cols]
    )
