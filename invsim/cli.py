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
from .metrics import performance_metrics
from .optimize import optimize_weights
from .schedule import contribution_schedule
from .simulation import (
    add_benchmarks,
    investing_start,
    simulate_dca,
    simulate_dynamic_portfolio,
    simulate_portfolio,
)

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


def _simulate_one(market: MarketData, ticker: str, args, output_root: Path) -> dict:
    """Simulate one ticker; write dashboard + CSVs; return its metrics row."""
    prices = market.prices(ticker, args.start, args.end)
    contributions = contribution_schedule(prices.index, args.amount)
    frame = simulate_dca(prices, contributions)
    frame = add_benchmarks(frame, market.tbill_rate(args.start, args.end), market.cpi(args.start, args.end))

    rf = align_monthly(_rf_series(market, args), frame.index)
    row = performance_metrics(frame["value"], frame["invested"], contributions, rf, label=ticker)

    out_dir = output_root / _slug(ticker, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "simulation_data.csv")
    if row:
        pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    plots.asset_dashboard(frame, ticker, out_dir / "dashboard.png")
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

    rows, failed = [], []
    for ticker in args.tickers:
        print(f"── Simulating {ticker} ...")
        try:
            row = _simulate_one(market, ticker, args, output_root)
            if row:
                rows.append(row)
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

    report_path = report.generate_report(
        output_root,
        comparison,
        portfolio_metrics,
        weights,
        {"period": f"{args.start}–{args.end}", "amount": args.amount},
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

    dynamic_frame, weight_history = simulate_dynamic_portfolio(
        prices,
        contributions,
        lookback_years=args.lookback,
        rebalance_months=args.rebalance,
        max_weight=args.max_weight,
        max_change=args.max_change,
        risk_free_rate=rf_mean,
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
        "Static (warm-up optimal)": simulate_portfolio(prices, window_contributions, baseline_weights),
        "Equal weight": simulate_portfolio(prices, window_contributions, equal_weights),
    }

    rf_daily = align_monthly(tbill, prices.index)
    rows = []
    for name, frame in frames.items():
        frames[name] = frame = add_benchmarks(frame, tbill, cpi)
        row = performance_metrics(
            frame["value"], frame["invested"], window_contributions, rf_daily, label=name
        )
        if row:
            rows.append(row)

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

    combos = list(itertools.product(args.lookback_grid, args.rebalance_grid, args.max_change_grid))
    print(f"Grid search: {len(combos)} combinations over {len(prices.columns)} assets")
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

    output = Path(args.output) / "grid_search_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print("\n=== Top 10 by risk-reward score ===")
    print(results.head(10).to_string(index=False))
    print(f"\nSaved: {output}")
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

    p = sub.add_parser("grid", help="grid search over dynamic-portfolio hyperparameters")
    p.add_argument("--tickers", "-t", nargs="+", default=DEFAULT_TICKERS)
    _add_common(p)
    p.add_argument("--max-weight", type=float, default=0.4)
    p.add_argument("--lookback-grid", type=float, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--rebalance-grid", type=int, nargs="+", default=[1, 3, 6, 12])
    p.add_argument("--max-change-grid", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p.set_defaults(func=cmd_grid)

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
