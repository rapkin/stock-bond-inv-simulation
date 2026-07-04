# Methodology

Formulas and assumptions used by `invsim`. Everything here applies identically to
every asset and portfolio, which is what makes the results comparable.

## Contribution schedule

- Paydays are **every second Friday** starting from the first Friday of the period.
- Each payday's contribution executes at the close of the **next trading day on or
  after** the payday (never an earlier one — the money doesn't exist before payday,
  and buying "yesterday" would be look-ahead).
- If two paydays collapse onto one trading day (market halt), the amounts add up.

## DCA accounting

On each contribution day: `shares += amount / close_price`. Portfolio value is
`shares × price` daily. Fractional shares are allowed; fills at the daily close.
Prices are Yahoo Finance **adjusted close**, so dividends are implicitly reinvested
and splits handled.

## Two kinds of return — and why both are reported

A DCA portfolio's raw value series mixes two things: market movement and your own
deposits. Metrics must separate them.

**Time-weighted returns (TWR)** — market performance with cash flows removed.
Contributions execute at the close, so the return earned by yesterday's holdings is

```
r_t = (V_t − F_t) / V_{t−1} − 1        F_t = contribution on day t
```

For a single asset this equals the asset's price return exactly. Volatility, Sharpe,
Sortino, max drawdown, CAGR, and win rate are all computed on these returns.

**Money-weighted return (XIRR)** — the annualized internal rate of return of the
actual dated cash flows (each contribution negative, final value positive):

```
Σ CF_i / (1 + r)^{years_i} = 0   →  solve for r  (Brent's method)
```

This is "what my dollars earned given *when* I invested them". For assets that
rallied late in the period, XIRR is higher than CAGR; for early rallies, lower.

## Risk metrics

With `p` = observed trading periods per year (measured from the data — ≈252 for
equities, ≈365 for crypto) and `rf_t` the daily-aligned FRED T-bill rate:

```
volatility   = std(r) × √p
sharpe       = mean(r − rf) / std(r) × √p
sortino      = mean(r − rf) / downside_dev × √p
downside_dev = √( mean( min(r − rf, 0)² ) )      ← 2nd lower partial moment,
                                                    averaged over ALL days
max_drawdown = min( W_t / cummax(W_t) − 1 ),  W = cumprod(1 + r)
CAGR         = W_final^(1 / years) − 1           (calendar years)
calmar       = CAGR / |max_drawdown|
```

Note the Sortino denominator: the classic error (present in the legacy code) is
`std(r[r<0])`, which ignores how *often* losses happen and rewards assets that lose
rarely but catastrophically.

## Benchmarks

**T-bills**: the same contributions accrue at the FRED `TB3MS` annual rate,
compounded over actual calendar-day gaps (ACT/365) — weekends and holidays earn
interest:

```
balance_t = growth_t × Σ_{buy ≤ t} amount / growth_buy,
growth_t = exp( Σ ln(1 + rate) × Δdays / 365 )
```

**Cash / inflation**: real (start-of-period purchasing power) values use the CPI
deflator `real_t = nominal_t × CPI_start / CPI_t`, with monthly `CPIAUCSL` joined
as-of each date (no interpolation is ever cached). "Cash under the mattress" is
just the contributions deflated — what doing nothing costs.

## Risk-reward score (heuristic)

Kept from the original project as the default ranking, now documented and centrally
implemented (fractions, not percent):

```
score = Sortino × √(1 − |MaxDD|) × √(max(TotalReturn, 0)) × suspicion
suspicion = 0.7 if Sortino > 7        (implausibly smooth)
          × 0.7 if |MaxDD| < 15% and TotalReturn > 50%   (implausibly clean)
```

The square roots soften the penalty for drawdowns and the reward for raw return; the
suspicion factor demotes results that usually indicate a data artifact or an
unsustainable regime. It is a *heuristic*, not a standard financial measure.

## Portfolio strategies

All three strategies invest the **same contributions over the same window** (after
the warm-up period), so their results are directly comparable:

1. **Dynamic (walk-forward)** — every `rebalance` months, weights are re-optimized
   (max Sharpe, SLSQP) on the trailing `lookback` years of **asset price returns**
   ending strictly *before* the rebalance date. Constraints: long-only,
   `weight ≤ max_weight`, per-rebalance turnover `|Δweight| ≤ max_change`. Holdings
   are traded to target at each rebalance; contributions between rebalances are
   split by current target weights.
2. **Static (warm-up optimal)** — weights optimized once, using *only* the warm-up
   window (data before the first investment), then held fixed. This is the honest
   version of the legacy "optimal" portfolio, which optimized over the full period
   and then pretended to have known those weights from day one.
3. **Equal weight** — 1/N, the classic hard-to-beat baseline.

The optimizer input is always asset **price returns** — never DCA portfolio values,
whose daily changes are contaminated by contribution inflows.

The risk-free rate for optimization is the mean FRED T-bill rate over the period
(not a hardcoded constant).

## Trading costs (opt-in, default zero)

Costs are bundled into named **profiles** (jurisdiction + broker), resolved
from built-ins in `invsim/profiles.py` merged with the user's `profiles.toml`;
CLI flags override individual fields. Built-in numbers are documented
approximations (e.g. IBKR Fixed = $0.005/share min $1/order → modeled as a $1
per-leg minimum; IBKR FX = 0.002% min $2 per conversion; Ukraine = 18% PIT +
5% military levy on investment gains) — verify against current broker pricing
and tax law, then pin your values in `profiles.toml`.

- **Commission** — each asset bought or sold is one trade leg; a leg of gross
  value `g` pays `max(g·pct + fixed, min)` (never leaving a negative net). A
  portfolio buy across N assets is N legs.
- **FX conversion** — charged once per contribution before splitting into
  legs: `max(g·fx_pct, fx_min)`. Models converting a non-USD paycheck to the
  trading currency.
- **Exit tax** — an estimate of tax due on liquidating at the end:
  `exit_tax × max(final − invested, 0)`, reported as `final_value_after_tax`
  and never deducted from the running simulation. It ignores basis step-up
  from gains already taxed at rebalances (slightly conservative) and any
  loss-carryforward rules.
- **Annual fee** (advisory fee, or the expense ratio of a raw index — note ETF
  adjusted closes already include their own expense ratio, don't double-count) —
  continuous ACT/365 drag. Implemented exactly by simulating against
  fee-adjusted prices `p̃_t = p_t · (1 − fee)^(days/365)`, which is equivalent to
  holdings decaying at the fee rate.
- **Capital-gains tax** — charged when the dynamic portfolio sells at a
  rebalance: selling fraction `f` of an asset realizes `f · (value − basis)`
  using average-cost basis per asset; positive gains are taxed and the tax plus
  commissions are deducted from the portfolio before reallocating (one-pass
  approximation of the post-cost fixed point). Final liquidation is not taxed.

Costs reduce value, never `invested` — the investor still paid the full
contribution out of pocket.

## Rolling-window robustness (`invsim rolling`)

One backtest is a single draw from history; DCA outcomes are dominated by the
start date. The rolling analysis re-runs the identical plan over every window
of length `--window` years, stepping the start by `--step` months, and reports
the distribution: median / 10th-percentile / worst / best XIRR, the worst
drawdown ever encountered, and the share of windows that beat T-bills and
inflation. Read the summary as "if I had started this plan at a random point
in the past, what usually happened — and what's the realistic bad case?"

## Walk-forward hyperparameter validation (`invsim grid`)

In-sample grid search selects whatever won the past — overfitting by
construction. The default mode (`--folds 3`) does anchored walk-forward CV:

1. The evaluation span (after the longest warm-up) is split into `folds + 1`
   equal chunks; the first extends the initial training span.
2. Per fold: every combination is scored on data strictly *before* the fold
   (the strategy itself is walk-forward, so this is genuinely out-of-fold),
   the best is selected on the training score (Sharpe), and only then scored
   on the unseen fold, next to an equal-weight baseline on the same window.

The output to trust is the mean out-of-sample score of the tuned parameters
versus the equal-weight baseline, and the train→test degradation. If tuning
doesn't beat 1/N out of sample, the "optimal" parameters are noise.

## Data sources

| Series | Source | Notes |
|--------|--------|-------|
| Prices | Yahoo Finance (`yfinance`) | adjusted close; `end` is exclusive, so each year is fetched to Jan 1 of the next |
| CPI | FRED `CPIAUCSL` | monthly, cached raw |
| T-bill | FRED `TB3MS` | monthly annual %, stored as decimal |

Fallbacks when FRED is unreachable (with a loud warning): synthetic 3%/yr CPI
anchored to a fixed epoch (so consecutive fallback years are continuous), flat 2%
T-bill rate.
