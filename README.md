# invsim — Investment Simulation

**Goal:** answer the question an ordinary person with a salary asks about investing —
*"If I had put a fixed amount of every paycheck into asset X, what would I actually
have today, in real purchasing power, and what risk would I have lived through?"*

The simulator models **dollar-cost averaging (DCA)**: a fixed dollar amount invested
every second Friday (payday), and compares the outcome against the realistic
alternatives:

- **Cash under the mattress** — loses purchasing power to inflation (real CPI data);
- **3-month T-bills** — the risk-free baseline (real FRED rates);
- **Other assets** — same methodology for every ticker, so results are comparable;
- **Portfolios** — Modern Portfolio Theory weights, including a walk-forward
  dynamically rebalanced portfolio with no look-ahead.

It is an **analysis tool, not financial advice**. It only looks at history, and does
not model taxes, broker fees, or commissions.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e '.[dev]'

# Full pipeline: simulate all default tickers, compare, optimize, HTML report
invsim run -s 2018 -e 2025 -a 500

open simulation_results/report.html
```

## Commands

| Command | What it does |
|---------|--------------|
| `invsim simulate -t QQQ -s 2020 -e 2025 -a 500` | DCA simulation of one ticker: dashboard chart, CSVs, risk metrics |
| `invsim run -t "GSPC QQQ GLD TLT" ...` | Everything: per-ticker sims → comparison table → portfolio optimization → `report.html` |
| `invsim portfolio --lookback 3 --rebalance 3 --max-weight 0.4 --max-change 0.10` | Dynamic vs static vs equal-weight portfolio comparison |
| `invsim rolling -s 2010 -e 2025 --window 5 --step 3` | **Robustness**: DCA outcome distribution across *all* rolling start dates — median/worst XIRR, % of windows beating T-bills/inflation |
| `invsim grid --lookback-grid 2 3 4 --rebalance-grid 1 3 6` | Hyperparameter search, **walk-forward validated** by default (`--folds 3`; `--folds 0` for raw in-sample) |

Common options: `--start/-s`, `--end/-e` (years), `--amount/-a` ($ per 2 weeks),
`--output/-o` (default `./simulation_results`), `--tickers/-t`.

Trading costs (all default to 0, rates are fractions): `--commission 0.001`
(0.1% per trade leg), `--commission-fixed 1` ($ per leg), `--annual-fee 0.0075`
(0.75%/yr drag), `--cgt 0.18` (18% tax on gains realized at rebalances).

### Why `rolling` and `--folds` matter

A single backtest is one draw from history. On 2018–2025 gold looks like the
best asset; across **all** 5-year windows since 2010 it beat inflation in only
~41% of them. Likewise, hyperparameters tuned in-sample usually lose their edge
on unseen data — `invsim grid` now tunes on a training span and scores on held-out
folds, reporting the degradation and an equal-weight baseline honestly.

Default tickers: `GSPC DJI IXIC QQQ` (US indices), `EZU EEM` (regions), `TLT`
(bonds), `GLD` (gold), `VNQ` (REITs), `BTC-USD ETH-USD SOL-USD` (crypto).
Index shorthands (`GSPC`, `DJI`, `IXIC`) map to Yahoo Finance `^`-symbols
automatically.

## Metrics

All risk metrics are computed on **flow-adjusted (time-weighted) returns** — daily
returns with contribution cash flows stripped out — so contributions can never
masquerade as market gains. The money outcome is reported separately as **XIRR**
(money-weighted annual return of the actual cash flows).

| Metric | Meaning |
|--------|---------|
| `total_return_pct` | Final value vs total contributed |
| `xirr_pct` | Money-weighted annual return of your actual dollars |
| `cagr_pct` | Time-weighted annual growth (market performance of the strategy) |
| `annual_volatility_pct`, `sharpe_ratio`, `sortino_ratio` | Risk-adjusted quality, vs the real T-bill rate |
| `max_drawdown_pct`, `max_underwater_days`, `calmar_ratio` | Worst-case pain |
| `risk_reward_score` | Combined heuristic ranking (see [docs/METHODOLOGY.md](docs/METHODOLOGY.md)) |

## Project layout

```
invsim/
├── data.py        # Yahoo Finance + FRED access, per-year cache (raw data only)
├── schedule.py    # biweekly paydays aligned to next trading day
├── simulation.py  # vectorized DCA engines, costs model, T-bill/inflation benchmarks
├── metrics.py     # TWR metrics, XIRR, drawdowns, risk-reward score
├── optimize.py    # max-Sharpe / min-variance weights, walk-forward optimizer
├── robustness.py  # rolling-window outcome distributions
├── validation.py  # walk-forward CV for hyperparameters
├── plots.py       # dashboards and portfolio charts
├── report.py      # self-contained HTML report
└── cli.py         # invsim simulate | run | portfolio | rolling | grid
tests/             # pytest suite (run: pytest)
docs/METHODOLOGY.md    # formulas, assumptions, data sources
docs/LEGACY_REVIEW.md  # what was wrong in the old scripts and what changed
legacy/                # previous implementation, kept for reference
```

## Data & caching

- Prices: Yahoo Finance **adjusted close** (splits + dividends included).
- CPI (`CPIAUCSL`) and 3-month T-bill rate (`TB3MS`): FRED, with a clearly-warned
  synthetic fallback (3% inflation / 2% rate) if FRED is unreachable.
- Cached per calendar year under `cache/`; only completed years are cached, the
  current year is always fetched fresh. Delete `cache/` to force a refetch.

## Limitations

- Costs are opt-in and simplified: flat commission per trade leg, a continuous
  annual fee drag, and capital-gains tax only on rebalance sales (average-cost
  basis, no final-liquidation tax). Dividend taxes are not modeled.
- CPI and T-bill data are US-only.
- Historical simulation only — nothing here predicts the future.
