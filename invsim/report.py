"""Self-contained HTML report (inline CSS, base64-embedded charts)."""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path

import pandas as pd

_CSS = """
:root { --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8;
        --accent:#38bdf8; --good:#4ade80; --bad:#f87171; --border:#334155; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:-apple-system,'Segoe UI',Roboto,sans-serif; background:var(--bg);
       color:var(--text); padding:2rem; max-width:1200px; margin:0 auto; }
h1 { font-size:1.6rem; margin-bottom:.25rem; }
h2 { font-size:1.15rem; margin:1.5rem 0 .75rem; color:var(--accent); }
.muted { color:var(--muted); font-size:.85rem; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
         gap:1rem; margin:1rem 0; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px;
        padding:1rem; }
.card .v { font-size:1.4rem; font-weight:700; margin-top:.25rem; }
.good { color:var(--good); } .bad { color:var(--bad); }
.tablewrap { overflow-x:auto; background:var(--card); border:1px solid var(--border);
             border-radius:10px; }
table { border-collapse:collapse; width:100%; font-size:.85rem; }
th,td { padding:.5rem .75rem; text-align:right; white-space:nowrap; }
th:first-child,td:first-child { text-align:left; }
th { background:#0b1220; color:var(--muted); position:sticky; top:0; }
tr:nth-child(even) td { background:rgba(255,255,255,.02); }
img { max-width:100%; border-radius:10px; margin:.5rem 0; }
footer { margin-top:2rem; color:var(--muted); font-size:.8rem; }
"""


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


def _table(df: pd.DataFrame, pct_cols: set[str] = frozenset(), money_cols: set[str] = frozenset()) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            kind = "pct" if col in pct_cols else "money" if col in money_cols else "num"
            cells.append(f"<td>{_fmt(row[col], kind)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _embed_image(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(path.stem)}">'


def generate_report(
    results_dir: Path,
    comparison: pd.DataFrame | None,
    portfolio_metrics: pd.DataFrame | None,
    weights: pd.Series | None,
    config: dict,
    output_file: Path | None = None,
) -> Path:
    """Assemble report.html from in-memory results and saved charts."""
    output_file = output_file or results_dir / "report.html"
    pct_cols = {
        "total_return_pct", "xirr_pct", "cagr_pct", "max_drawdown_pct",
        "annual_volatility_pct", "win_rate_pct",
    }
    money_cols = {"total_invested", "final_value"}

    sections: list[str] = []

    amount = config.get("amount", 0)
    sections.append(
        '<div class="cards">'
        f'<div class="card"><div class="muted">Period</div><div class="v">{html.escape(str(config.get("period", "—")))}</div></div>'
        f'<div class="card"><div class="muted">Contribution</div><div class="v">${amount:,.0f} / 2 weeks</div></div>'
        f'<div class="card"><div class="muted">Assets</div><div class="v">{len(comparison) if comparison is not None else 0}</div></div>'
        "</div>"
    )

    if comparison is not None and not comparison.empty:
        sections.append("<h2>Assets, ranked by risk-reward score</h2>")
        sections.append(_table(comparison, pct_cols, money_cols))
        sections.append(
            '<p class="muted">risk_reward_score = Sortino × √(1−|MaxDD|) × √(Return) '
            "× suspicion penalty. Sharpe/Sortino/drawdown are computed on "
            "flow-adjusted (time-weighted) returns; XIRR is the money-weighted "
            "annual return of the actual contributions.</p>"
        )

    if weights is not None and len(weights):
        weights_df = pd.DataFrame(
            {"ticker": weights.index, "weight_pct": (weights * 100).round(1)}
        )
        sections.append("<h2>Baseline optimized weights (from warm-up window)</h2>")
        sections.append(_table(weights_df, pct_cols=set()))

    if portfolio_metrics is not None and not portfolio_metrics.empty:
        sections.append("<h2>Portfolio strategies</h2>")
        sections.append(_table(portfolio_metrics, pct_cols, money_cols))
        sections.append(
            '<p class="muted">All strategies invest over the same window (after the '
            "lookback warm-up). Static weights are optimized only on warm-up data — "
            "no look-ahead.</p>"
        )

    for chart in ("portfolio_comparison.png", "portfolio_weights.png"):
        image_html = _embed_image(results_dir / chart)
        if image_html:
            sections.append(image_html)

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
commissions are not modeled.</footer>
</body></html>"""

    output_file.write_text(document, encoding="utf-8")
    return output_file
