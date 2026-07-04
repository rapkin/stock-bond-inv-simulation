"""Self-contained HTML report (inline CSS, base64-embedded charts).

Cross-asset charts are embedded base64 so the file is portable; the large
per-asset tearsheets are referenced by relative path (they live next to the
report inside the results directory) behind collapsible sections, so the
report stays light while every asset still gets a full risk workup.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

_CSS = """
:root { --bg:#0d0d0d; --card:#1a1a19; --text:#ffffff; --muted:#898781;
        --soft:#c3c2b7; --accent:#3987e5; --good:#0ca30c; --bad:#e66767;
        --border:#383835; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:system-ui,-apple-system,'Segoe UI',sans-serif; background:var(--bg);
       color:var(--soft); padding:2rem; max-width:1200px; margin:0 auto; }
h1 { font-size:1.6rem; margin-bottom:.25rem; color:var(--text); }
h2 { font-size:1.1rem; margin:2rem 0 .75rem; color:var(--text);
     border-bottom:1px solid var(--border); padding-bottom:.4rem; }
.muted { color:var(--muted); font-size:.85rem; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
         gap:1rem; margin:1.25rem 0; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px;
        padding:1rem 1.1rem; }
.card .k { color:var(--muted); font-size:.78rem; text-transform:uppercase;
           letter-spacing:.04em; }
.card .v { font-size:1.55rem; font-weight:700; margin-top:.3rem; color:var(--text); }
.card .s { color:var(--muted); font-size:.78rem; margin-top:.25rem; }
.good { color:var(--good); } .bad { color:var(--bad); }
.tablewrap { overflow-x:auto; background:var(--card); border:1px solid var(--border);
             border-radius:10px; }
table { border-collapse:collapse; width:100%; font-size:.83rem; }
th,td { padding:.5rem .7rem; text-align:right; white-space:nowrap;
        font-variant-numeric:tabular-nums; }
th:first-child,td:first-child { text-align:left; }
th { background:#111110; color:var(--muted); position:sticky; top:0;
     font-weight:600; }
tr:nth-child(even) td { background:rgba(255,255,255,.02); }
img { max-width:100%; border-radius:10px; margin:.5rem 0; display:block; }
details { background:var(--card); border:1px solid var(--border); border-radius:10px;
          margin:.5rem 0; }
details summary { cursor:pointer; padding:.7rem 1rem; color:var(--text);
                  font-weight:600; }
details[open] summary { border-bottom:1px solid var(--border); }
details .inner { padding: .75rem; }
footer { margin-top:2rem; color:var(--muted); font-size:.8rem; }
p.note { color:var(--muted); font-size:.82rem; margin:.5rem 0 0; }
"""

PCT_COLUMNS = {
    "total_return_pct", "xirr_pct", "cagr_pct", "max_drawdown_pct",
    "annual_volatility_pct", "win_rate_pct", "var_95_pct", "cvar_95_pct",
    "best_month_pct", "worst_month_pct",
}
MONEY_COLUMNS = {"total_invested", "final_value", "final_value_after_tax"}


def _fmt(value, kind: str = "num") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if kind == "money":
        return f"${value:,.0f}"
    if kind == "pct":
        cls = "good" if value >= 0 else "bad"
        return f'<span class="{cls}">{value:+.1f}%</span>'
    if isinstance(value, float):
        return f"{value:,.2f}"
    return html.escape(str(value))


def _table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            kind = "pct" if col in PCT_COLUMNS else "money" if col in MONEY_COLUMNS else "num"
            cells.append(f"<td>{_fmt(row[col], kind)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _embed_image(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(path.stem)}">'


def _tile(key: str, value: str, sub: str = "", cls: str = "") -> str:
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return (
        f'<div class="card"><div class="k">{html.escape(key)}</div>'
        f'<div class="v {cls}">{value}</div>{sub_html}</div>'
    )


def _hero_tiles(headline: pd.Series, config: dict, n_assets: int) -> str:
    """Stat tiles for the headline result (dynamic portfolio or best asset)."""
    xirr = headline.get("xirr_pct")
    tiles = [
        _tile(
            f"{headline['label']} — XIRR",
            f"{xirr:+.1f}%/yr" if xirr is not None and pd.notna(xirr) else "—",
            "money-weighted return of your dollars",
            "good" if (xirr or 0) >= 0 else "bad",
        ),
        _tile(
            "Final value",
            f"${headline['final_value']:,.0f}",
            f"from ${headline['total_invested']:,.0f} contributed",
        ),
        _tile(
            "Max drawdown",
            f"{headline['max_drawdown_pct']:.1f}%",
            f"{headline['max_underwater_days']} days underwater at worst",
            "bad",
        ),
        _tile(
            "Sharpe / Sortino",
            f"{headline['sharpe_ratio']:.2f} / {headline['sortino_ratio']:.2f}",
            f"daily VaR95 {headline.get('var_95_pct', float('nan')):.1f}%, "
            f"CVaR95 {headline.get('cvar_95_pct', float('nan')):.1f}%",
        ),
        _tile(
            "Plan",
            f"${config.get('amount', 0):,.0f} / 2wk",
            f"{html.escape(str(config.get('period', '')))} · {n_assets} assets",
        ),
    ]
    return f'<div class="cards">{"".join(tiles)}</div>'


def generate_report(
    results_dir: Path,
    comparison: pd.DataFrame | None,
    portfolio_metrics: pd.DataFrame | None,
    weights: pd.Series | None,
    config: dict,
    output_file: Path | None = None,
    asset_pages: dict[str, str] | None = None,
) -> Path:
    """Assemble report.html from in-memory results and saved charts."""
    output_file = output_file or results_dir / "report.html"
    sections: list[str] = []

    headline = None
    if portfolio_metrics is not None and not portfolio_metrics.empty:
        headline = portfolio_metrics.iloc[0]
    elif comparison is not None and not comparison.empty:
        headline = comparison.iloc[0]
    if headline is not None:
        n_assets = len(comparison) if comparison is not None else 0
        sections.append(_hero_tiles(headline, config, n_assets))

    if portfolio_metrics is not None and not portfolio_metrics.empty:
        sections.append("<h2>Portfolio strategies</h2>")
        sections.append(_table(portfolio_metrics))
        sections.append(
            '<p class="note">All strategies invest the same contributions over the '
            "same window (after the lookback warm-up). Static weights come from "
            "warm-up data only — no look-ahead. Sharpe, Sortino, volatility, and "
            "drawdowns use flow-adjusted (time-weighted) returns; XIRR is the "
            "money-weighted annual return.</p>"
        )
        for chart in ("portfolio_comparison.png", "strategy_drawdowns.png",
                      "portfolio_weights.png"):
            sections.append(_embed_image(results_dir / chart))
        tearsheet = results_dir / "portfolio_tearsheet.png"
        if tearsheet.exists():
            sections.append(
                "<details><summary>Dynamic portfolio — full tearsheet</summary>"
                f'<div class="inner"><img src="{tearsheet.name}" loading="lazy" '
                'alt="dynamic portfolio tearsheet"></div></details>'
            )

    if weights is not None and len(weights):
        weights_df = pd.DataFrame(
            {"ticker": weights.index, "weight_pct": (weights * 100).round(1)}
        )
        sections.append("<h2>Baseline optimized weights (from warm-up window)</h2>")
        sections.append(_table(weights_df))

    if comparison is not None and not comparison.empty:
        sections.append("<h2>Assets, ranked by risk-reward score</h2>")
        sections.append(_table(comparison))
        sections.append(
            '<p class="note">risk_reward_score = Sortino × √(1−|MaxDD|) × √(Return) '
            "× suspicion penalty — a heuristic ranking, see docs/METHODOLOGY.md. "
            "VaR95/CVaR95: the daily loss exceeded on 5% of days, and the average "
            "loss on those days.</p>"
        )
        sections.append(_embed_image(results_dir / "risk_return.png"))
        sections.append(_embed_image(results_dir / "correlation.png"))

        if asset_pages:
            sections.append("<h2>Per-asset tearsheets</h2>")
            for label, rel_path in asset_pages.items():
                sections.append(
                    f"<details><summary>{html.escape(label)}</summary>"
                    f'<div class="inner"><img src="{html.escape(rel_path)}" '
                    f'loading="lazy" alt="{html.escape(label)} tearsheet"></div></details>'
                )

    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Investment Simulation Report</title><style>{_CSS}</style></head>
<body>
<h1>Investment Simulation Report</h1>
<p class="muted">Generated {datetime.now():%Y-%m-%d %H:%M} · biweekly DCA ·
inflation-adjusted · vs T-bills and cash</p>
{''.join(sections)}
<footer>invsim — analysis tool, not financial advice. Taxes, fees, and broker
commissions are modeled only if enabled via --commission/--annual-fee/--cgt.</footer>
</body></html>"""

    output_file.write_text(document, encoding="utf-8")
    return output_file
