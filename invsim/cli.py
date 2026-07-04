"""Command-line interface.

Subcommands:
  simulate   DCA simulation for a single ticker
  run        full pipeline: all tickers -> comparison -> portfolio -> report
  portfolio  optimize + simulate dynamic/static/equal-weight portfolios
  grid       hyperparameter grid search for the dynamic portfolio
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

from . import plots, report
from .data import MarketData, align_monthly
from .metrics import flow_adjusted_returns, performance_metrics
from .optimize import optimize_weights
from .robustness import rolling_dca, summarize_windows
from .schedule import contribution_schedule
from .simulation import (
    Costs,
    add_benchmarks,
    simulate_dca,
    simulate_dynamic_portfolio,
    simulate_portfolio,
)
from .validation import build_grid, walk_forward_validate

DEFAULT_TICKERS = [
    "GSPC", "DJI", "IXIC", "QQQ",       # US indices
    "EZU", "EEM",                        # regions
    "TLT",                               # bonds
    "GLD",                               # gold
    "VNQ",                               # real estate
    "BTC-USD", "ETH-USD", "SOL-USD",     # crypto
]


def _slug(ticker: str, args) -> str:
    return f"{ticker}_{args.start}-{args.end}_${int(args.amount)}"


def _rf_series(market: MarketData, args) -> pd.Series:
    return market.tbill_rate(args.start, args.end)


def _costs_from(args) -> Costs:
    return Costs(
        commission_pct=args.commission,
        commission_fixed=args.commission_fixed,
        annual_fee_pct=args.annual_fee,
        capital_gains_tax_pct=args.cgt,
    )


def _simulate_one(market: MarketData, ticker: str, args, output_root: Path) -> dict:
    """Simulate one ticker; write dashboard + CSVs; return its metrics row."""
    prices = market.prices(ticker, args.start, args.end)
    contributions = contribution_schedule(prices.index, args.amount)
    frame = simulate_dca(prices, contributions, _costs_from(args))
    frame = add_benchmarks(frame, market.tbill_rate(args.start, args.end), market.cpi(args.start, args.end))

    rf = align_monthly(_rf_series(market, args), frame.index)
    row = performance_metrics(frame["value"], frame["invested"], contributions, rf, label=ticker)

    out_dir = output_root / _slug(ticker, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "simulation_data.csv")
    if row:
        pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    returns = flow_adjusted_returns(frame["value"], contributions)
    plots.tearsheet(frame, returns, rf, ticker, out_dir / "tearsheet.png")
    return row


def cmd_simulate(args) -> int:
    market = MarketData(args.cache_dir)
    output_root = Path(args.output)
    row = _simulate_one(market, args.ticker, args, output_root)
    if not row:
        print(f"Not enough data to compute metrics for {args.ticker}", file=sys.stderr)
        return 1
    _print_metrics_table(pd.DataFrame([row]))
    print(f"\nResults: {output_root / _slug(args.ticker, args)}")
    return 0


def cmd_run(args) -> int:
    market = MarketData(args.cache_dir)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    rows, failed, asset_pages = [], [], {}
    for ticker in args.tickers:
        print(f"── Simulating {ticker} ...")
        try:
            row = _simulate_one(market, ticker, args, output_root)
            if row:
                rows.append(row)
                asset_pages[ticker] = f"{_slug(ticker, args)}/tearsheet.png"
        except Exception as exc:
            failed.append(ticker)
            print(f"   FAILED: {exc}", file=sys.stderr)

    comparison = None
    if rows:
        comparison = (
            pd.DataFrame(rows)
            .sort_values("risk_reward_score", ascending=False)
            .reset_index(drop=True)
        )
        comparison.to_csv(output_root / "comparison.csv", index=False)
        print("\n=== Asset comparison (by risk-reward score) ===")
        _print_metrics_table(comparison)

    portfolio_metrics = weights = None
    usable = [t for t in args.tickers if t not in failed]
    if len(usable) >= 2:
        print("\n── Optimizing portfolio ...")
        try:
            portfolio_metrics, weights = _run_portfolio(market, usable, args, output_root)
            _print_metrics_table(portfolio_metrics)
        except Exception as exc:
            print(f"   Portfolio step failed: {exc}", file=sys.stderr)

        try:
            asset_returns = (
                market.price_matrix(usable, args.start, args.end).pct_change().dropna()
            )
            plots.correlation_heatmap(asset_returns, output_root / "correlation.png")
        except Exception as exc:
            print(f"   Correlation chart failed: {exc}", file=sys.stderr)

    if comparison is not None:
        points = [
            {"label": r["label"], "annual_volatility_pct": r["annual_volatility_pct"],
             "cagr_pct": r["cagr_pct"], "kind": "asset"}
            for _, r in comparison.iterrows()
        ]
        if portfolio_metrics is not None:
            points += [
                {"label": r["label"], "annual_volatility_pct": r["annual_volatility_pct"],
                 "cagr_pct": r["cagr_pct"], "kind": "strategy"}
                for _, r in portfolio_metrics.iterrows()
            ]
        plots.risk_return_scatter(pd.DataFrame(points), output_root / "risk_return.png")

    report_path = report.generate_report(
        output_root,
        comparison,
        portfolio_metrics,
        weights,
        {"period": f"{args.start}–{args.end}", "amount": args.amount},
        asset_pages=asset_pages,
    )
    print(f"\nReport: {report_path}")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}", file=sys.stderr)
    return 0 if rows else 1


def _run_portfolio(
    market: MarketData, tickers: list[str], args, output_root: Path
) -> tuple[pd.DataFrame, pd.Series]:
    prices = market.price_matrix(tickers, args.start, args.end)
    tbill = market.tbill_rate(args.start, args.end)
    cpi = market.cpi(args.start, args.end)
    rf_mean = float(tbill.mean())

    contributions = contribution_schedule(prices.index, args.amount)
    costs = _costs_from(args)

    dynamic_frame, weight_history = simulate_dynamic_portfolio(
        prices,
        contributions,
        lookback_years=args.lookback,
        rebalance_months=args.rebalance,
        max_weight=args.max_weight,
        max_change=args.max_change,
        risk_free_rate=rf_mean,
        costs=costs,
    )

    # Fair comparison: static and equal-weight invest over the same window as
    # the dynamic strategy (after warm-up). Static weights come from the
    # warm-up window only — no look-ahead.
    start = weight_history.index[0]
    window_contributions = contributions[contributions.index >= start]
    warmup_returns = prices.loc[prices.index < start].pct_change().dropna()
    baseline_weights = optimize_weights(
        warmup_returns,
        risk_free_rate=rf_mean,
        max_weight=args.max_weight,
    )
    equal_weights = pd.Series(1.0 / len(tickers), index=prices.columns)

    frames = {
        "Dynamic (walk-forward)": dynamic_frame,
        "Static (warm-up optimal)": simulate_portfolio(
            prices, window_contributions, baseline_weights, costs
        ),
        "Equal weight": simulate_portfolio(prices, window_contributions, equal_weights, costs),
    }

    rf_daily = align_monthly(tbill, prices.index)
    rows, strategy_returns = [], {}
    for name, frame in frames.items():
        frames[name] = frame = add_benchmarks(frame, tbill, cpi)
        strategy_returns[name] = flow_adjusted_returns(frame["value"], window_contributions)
        row = performance_metrics(
            frame["value"], frame["invested"], window_contributions, rf_daily, label=name
        )
        if row:
            rows.append(row)

    dynamic_name = "Dynamic (walk-forward)"
    plots.tearsheet(
        frames[dynamic_name],
        strategy_returns[dynamic_name],
        rf_daily,
        "Dynamic portfolio",
        output_root / "portfolio_tearsheet.png",
    )
    plots.strategy_drawdowns(strategy_returns, output_root / "strategy_drawdowns.png")

    weight_history.to_csv(output_root / "portfolio_weights.csv")
    baseline_weights.rename("weight").to_csv(output_root / "optimal_portfolio.csv")
    combined = pd.concat(
        {name: frame[["value", "invested", "profit"]] for name, frame in frames.items()}, axis=1
    )
    combined.to_csv(output_root / "portfolio_simulation.csv")
    plots.portfolio_comparison(frames, output_root / "portfolio_comparison.png")
    plots.weights_chart(weight_history, output_root / "portfolio_weights.png")

    return pd.DataFrame(rows), baseline_weights


def cmd_portfolio(args) -> int:
    market = MarketData(args.cache_dir)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics, _ = _run_portfolio(market, args.tickers, args, output_root)
    _print_metrics_table(metrics)
    return 0


def cmd_grid(args) -> int:
    market = MarketData(args.cache_dir)
    prices = market.price_matrix(args.tickers, args.start, args.end)
    tbill = market.tbill_rate(args.start, args.end)
    rf_daily = align_monthly(tbill, prices.index)
    contributions = contribution_schedule(prices.index, args.amount)
    costs = _costs_from(args)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.folds > 0:
        return _grid_walk_forward(args, prices, contributions, tbill, rf_daily, costs, output_root)
    return _grid_in_sample(args, prices, contributions, tbill, rf_daily, costs, output_root)


def _grid_walk_forward(args, prices, contributions, tbill, rf_daily, costs, output_root) -> int:
    grid = build_grid(args.lookback_grid, args.rebalance_grid, args.max_change_grid)
    print(
        f"Walk-forward validation: {len(grid)} combinations, {args.folds} test folds, "
        f"{len(prices.columns)} assets"
    )
    fold_results, combo_scores = walk_forward_validate(
        prices,
        contributions,
        grid,
        folds=args.folds,
        rf_daily=rf_daily,
        risk_free_rate=float(tbill.mean()),
        max_weight=args.max_weight,
        costs=costs,
    )
    if fold_results.empty:
        print("No fold produced a valid result", file=sys.stderr)
        return 1

    fold_results.to_csv(output_root / "grid_validation_folds.csv", index=False)
    combo_scores.to_csv(output_root / "grid_validation_combos.csv", index=False)

    print("\n=== Walk-forward results (tuned on train, scored on unseen test) ===")
    print(fold_results.to_string(index=False))

    test_col = "test_sharpe_ratio"
    baseline_col = "equal_weight_sharpe_ratio"
    scored = fold_results.dropna(subset=[test_col, baseline_col])
    if not scored.empty:
        tuned = scored[test_col].mean()
        baseline = scored[baseline_col].mean()
        degradation = (fold_results["train_sharpe_ratio"] - fold_results[test_col]).mean()
        print(f"\nMean out-of-sample Sharpe: tuned {tuned:.3f} vs equal-weight {baseline:.3f}")
        print(f"Mean train->test degradation: {degradation:.3f}")
        if tuned <= baseline:
            print("Verdict: tuning does NOT beat the equal-weight baseline out of sample.")
        else:
            print("Verdict: tuned parameters kept an edge out of sample — but check stability across folds.")
    print(f"\nSaved: {output_root / 'grid_validation_folds.csv'}")
    return 0


def _grid_in_sample(args, prices, contributions, tbill, rf_daily, costs, output_root) -> int:
    combos = list(itertools.product(args.lookback_grid, args.rebalance_grid, args.max_change_grid))
    print(f"In-sample grid search ({len(combos)} combinations) — beware of overfitting; "
          f"prefer --folds 3")
    rows = []
    for i, (lookback, rebalance, max_change) in enumerate(combos, 1):
        try:
            frame, history = simulate_dynamic_portfolio(
                prices,
                contributions,
                lookback_years=lookback,
                rebalance_months=rebalance,
                max_weight=args.max_weight,
                max_change=max_change,
                risk_free_rate=float(tbill.mean()),
                costs=costs,
            )
        except ValueError as exc:
            print(f"[{i}/{len(combos)}] lookback={lookback} rebalance={rebalance} "
                  f"max_change={max_change:.2f} -> skipped ({exc})")
            continue
        window = contributions[contributions.index >= history.index[0]]
        row = performance_metrics(frame["value"], frame["invested"], window, rf_daily)
        if row:
            row.update({"lookback": lookback, "rebalance": rebalance, "max_change": max_change})
            rows.append(row)
            print(f"[{i}/{len(combos)}] lookback={lookback} rebalance={rebalance} "
                  f"max_change={max_change:.2f} -> XIRR {row['xirr_pct']}%, "
                  f"Sharpe {row['sharpe_ratio']}")

    if not rows:
        print("No successful combinations", file=sys.stderr)
        return 1
    results = pd.DataFrame(rows).drop(columns=["label", "start", "end"])
    front = ["lookback", "rebalance", "max_change"]
    results = results[front + [c for c in results.columns if c not in front]]
    results = results.sort_values("risk_reward_score", ascending=False)

    output = output_root / "grid_search_results.csv"
    results.to_csv(output, index=False)
    print("\n=== Top 10 by risk-reward score ===")
    print(results.head(10).to_string(index=False))
    print(f"\nSaved: {output}")
    return 0


def cmd_rolling(args) -> int:
    market = MarketData(args.cache_dir)
    tbill = market.tbill_rate(args.start, args.end)
    cpi = market.cpi(args.start, args.end)
    costs = _costs_from(args)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries, per_ticker, failed = [], {}, []
    for ticker in args.tickers:
        try:
            prices = market.prices(ticker, args.start, args.end)
            windows = rolling_dca(
                prices, tbill, cpi, args.window, args.amount,
                step_months=args.step, costs=costs,
            )
        except Exception as exc:
            failed.append(ticker)
            print(f"{ticker}: FAILED ({exc})", file=sys.stderr)
            continue
        if windows.empty:
            failed.append(ticker)
            print(f"{ticker}: no full {args.window}-year window in {args.start}-{args.end}",
                  file=sys.stderr)
            continue
        per_ticker[ticker] = windows
        summaries.append(summarize_windows(windows, label=ticker))
        print(f"{ticker}: {len(windows)} windows")

    if not summaries:
        return 1

    summary_df = pd.DataFrame(summaries).sort_values("median_xirr_pct", ascending=False)
    print(f"\n=== {args.window}-year DCA outcomes across all start dates "
          f"({args.start}-{args.end}, step {args.step}mo) ===")
    print(summary_df.to_string(index=False))
    print("\nColumns: median/p10/worst/best XIRR across windows; beat_tbills_pct / "
          "beat_inflation_pct = share of windows that beat the benchmark.")

    all_windows = pd.concat(
        [w.assign(ticker=t) for t, w in per_ticker.items()], ignore_index=True
    )
    all_windows.to_csv(output_root / "rolling_windows.csv", index=False)
    summary_df.to_csv(output_root / "rolling_summary.csv", index=False)
    plots.rolling_xirr_chart(per_ticker, output_root / "rolling_xirr.png")
    print(f"\nSaved: {output_root / 'rolling_summary.csv'}, rolling_windows.csv, rolling_xirr.png")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}", file=sys.stderr)
    return 0


def _print_metrics_table(df: pd.DataFrame) -> None:
    columns = [c for c in df.columns if c not in ("start", "end")]
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(df[columns].to_string(index=False))


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", "-s", type=int, default=2020, help="first year (default 2020)")
    parser.add_argument("--end", "-e", type=int, default=2025, help="last year (default 2025)")
    parser.add_argument("--amount", "-a", type=float, default=500.0,
                        help="contribution every 2 weeks in $ (default 500)")
    parser.add_argument("--output", "-o", default="./simulation_results",
                        help="output directory (default ./simulation_results)")
    parser.add_argument("--cache-dir", default=None, help="data cache directory")
    parser.add_argument("--commission", type=float, default=0.0,
                        help="commission per trade leg as a fraction, e.g. 0.001 = 0.1%% (default 0)")
    parser.add_argument("--commission-fixed", type=float, default=0.0,
                        help="fixed $ commission per trade leg (default 0)")
    parser.add_argument("--annual-fee", type=float, default=0.0,
                        help="annual fee drag as a fraction, e.g. 0.0075 = 0.75%%/yr (default 0)")
    parser.add_argument("--cgt", type=float, default=0.0,
                        help="capital gains tax on rebalance sales as a fraction, "
                             "e.g. 0.18 = 18%% (default 0)")


def _add_portfolio_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lookback", type=float, default=3.0,
                        help="optimization lookback in years (default 3)")
    parser.add_argument("--rebalance", type=int, default=3,
                        help="rebalance every N months (default 3)")
    parser.add_argument("--max-weight", type=float, default=0.4,
                        help="max weight per asset (default 0.4)")
    parser.add_argument("--max-change", type=float, default=0.10,
                        help="max weight change per rebalance (default 0.10)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invsim", description="Biweekly DCA investment simulator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="simulate a single ticker")
    p.add_argument("--ticker", "-t", default="GSPC")
    _add_common(p)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("run", help="full pipeline: simulate, compare, optimize, report")
    p.add_argument("--tickers", "-t", nargs="+", default=DEFAULT_TICKERS)
    _add_common(p)
    _add_portfolio_options(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("portfolio", help="portfolio optimization and comparison")
    p.add_argument("--tickers", "-t", nargs="+", default=DEFAULT_TICKERS)
    _add_common(p)
    _add_portfolio_options(p)
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("grid", help="grid search over dynamic-portfolio hyperparameters "
                                    "(walk-forward validated by default)")
    p.add_argument("--tickers", "-t", nargs="+", default=DEFAULT_TICKERS)
    _add_common(p)
    p.add_argument("--max-weight", type=float, default=0.4)
    p.add_argument("--lookback-grid", type=float, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--rebalance-grid", type=int, nargs="+", default=[1, 3, 6, 12])
    p.add_argument("--max-change-grid", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--folds", type=int, default=3,
                   help="walk-forward test folds; 0 = in-sample only (default 3)")
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("rolling", help="rolling-window robustness: DCA outcome distribution "
                                       "across all start dates")
    p.add_argument("--tickers", "-t", nargs="+", default=DEFAULT_TICKERS)
    _add_common(p)
    p.add_argument("--window", "-w", type=float, default=5.0,
                   help="window length in years (default 5)")
    p.add_argument("--step", type=int, default=3,
                   help="months between window starts (default 3)")
    p.set_defaults(func=cmd_rolling)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "start") and args.start > args.end:
        print("error: start year must not be after end year", file=sys.stderr)
        return 2
    if hasattr(args, "amount") and args.amount <= 0:
        print("error: amount must be positive", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
