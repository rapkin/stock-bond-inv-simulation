# Legacy code review — what was wrong and what changed

The original project (now in `legacy/`) was a working set of experiments:
`investment_simulation.py`, `aggregate_results.py`, `optimize_portfolio.py`,
`grid_search.py`, `generate_report.py`, orchestrated by `run_simulations.sh`
passing CSVs between scripts. It produced useful output but had correctness,
precision, and structure problems. This file records them and how the rewrite
(`invsim/`) addresses each — it's also the honest changelog for why new numbers
differ from old ones.

## Correctness / financial-math bugs

| # | Legacy problem | Fix in `invsim` |
|---|----------------|------------------|
| 1 | **Contributions counted as returns.** All risk metrics (Sharpe, Sortino, volatility, drawdown) were computed on the raw DCA portfolio-value series, so every biweekly $500 deposit registered as a positive "return". Ratios were systematically inflated, most severely early in the period when the portfolio was small. | Metrics are computed on flow-adjusted (time-weighted) returns: `(V_t − F_t)/V_{t−1} − 1`. Verified by test: a flat-price asset with contributions shows exactly 0 return and 0 volatility. |
| 2 | **MPT optimized on contaminated returns.** `optimize_portfolio.load_returns_data` fed portfolio-value pct-changes (deposits included) into the mean/covariance estimation. | Optimizer input is always asset price returns. |
| 3 | **Look-ahead bias in the "optimal" portfolio.** Static weights were optimized over the *full* period, then the simulation pretended to have known them from day one, unfairly beating the dynamic strategy. | Static weights come only from the warm-up window (data before the first investment). Walk-forward weights use only data strictly before each rebalance date — proven by a test that corrupts future data and checks weights don't change. |
| 4 | **Wrong CAGR for DCA.** `(final_value / total_invested)^(1/years)` treats gradually-invested money as if it were all invested on day one, understating the true rate. | Money-weighted return is reported as proper XIRR; time-weighted CAGR is computed from the flow-adjusted wealth index. |
| 5 | **Wrong Sortino denominator.** `std(returns[returns < 0])` ignores loss frequency (an asset that rarely but catastrophically loses looks safe). | Second lower partial moment over all observations: `√(mean(min(r − rf, 0)²))`. |
| 6 | **Dec 31 silently dropped every year.** `yf.download(end="YYYY-12-31")` is exclusive of `end`, so each cached year lost its last trading day. | Each year is fetched with `end = Jan 1` of the next year. |
| 7 | **T-bills only compounded on trading days.** Daily rate `r/365` was applied only on days present in the stock index — weekends/holidays earned nothing, understating the benchmark by ~30%. | ACT/365 accrual over actual calendar-day gaps. Verified: $1000 at flat 5% for one year → $1050. |
| 8 | **Contributions silently skipped on holidays.** The portfolio simulators bought only when a raw Friday exactly matched a trading day; holiday paydays vanished, and different strategies could end up investing different totals (making their comparison meaningless). | Paydays map to the next trading day on/after; colliding paydays sum; all compared strategies use the identical contribution schedule. |
| 9 | **Infeasible constraints silently ignored.** With 2 assets and `max_weight=0.4`, weights can't sum to 1; SLSQP returned an invalid portfolio with only a printed warning. | Raises `ValueError` immediately. |
| 10 | **Hardcoded 2% risk-free rate** in the optimizer and its Sharpe reporting, while the simulator downloaded real T-bill rates — two inconsistent Sharpe definitions across the project. | One metrics module; the real FRED T-bill series (or its period mean for optimization) is used everywhere. |
| 11 | **Interpolated CPI cached per year.** Monthly CPI was linearly interpolated to daily and *then* cached year-by-year, creating discontinuities at year boundaries; the synthetic fallback restarted at 100 every year, making multi-year fallbacks nonsense. | Raw monthly series are cached; alignment happens at the point of use. The synthetic fallback is anchored to a fixed epoch, so consecutive years are continuous. |
| 12 | **Annualization assumed 252 days for everything**, including crypto that trades ~365 days/year. | Periods-per-year is measured from the data. |

## Precision / quality improvements

- **Adjusted close** (dividends reinvested) instead of raw close — the old numbers
  ignored dividends entirely for ETFs like VNQ/TLT where they are a large share of
  total return.
- **Weight history correctness**: the legacy weights chart zipped a per-day list
  against a possibly-shifted date list; weights are now indexed by rebalance date.
- **"Portfolio drawdown" was Σ wᵢ·MaxDDᵢ** of individual assets (drawdowns are
  non-additive and non-simultaneous); now drawdown is computed from the simulated
  portfolio's own return series.

## Performance

- Legacy simulation loops were O(days × purchase dates) with per-cell `DataFrame.loc`
  lookups inside nested day×ticker loops (minutes for long periods). The new engines
  are vectorized (cumsum/outer products; the dynamic engine iterates only over
  buy/rebalance events) — the full 6-ticker 8-year pipeline runs in seconds.

## Structure

- One package with one implementation of each concept. Legacy had: two different
  metric implementations (`investment_simulation` vs `optimize_portfolio`), two
  near-identical CSV loaders, three DCA loops, and ~200 lines of dead code
  (`generate_efficient_frontier`, unused `is_purchase_day`/`purchase_dates_passed`,
  superseded `print_summary`/visualization variants).
- Scripts communicated via intermediate CSVs on disk; the pipeline now passes data
  in memory and writes CSVs only as *outputs*.
- One CLI (`invsim`) with subcommands instead of four scripts plus a bash wrapper.
- HTML report: templated small functions, `html.escape` on all data, guarded
  divisions, and it now includes the dynamic portfolio (the legacy report silently
  omitted the headline strategy).
- Tests target the new engines directly with synthetic data; no network needed.

## Deliberate simplifications

- One 2×2 dashboard PNG per asset instead of six near-duplicate charts.
- English throughout (code, CLI, report); the legacy was mixed Ukrainian/English.
- The `risk_reward_score` heuristic (including the 0.7 "suspicion" penalties) is
  kept as the default ranking for continuity, but is documented as a heuristic in
  `docs/METHODOLOGY.md`.
