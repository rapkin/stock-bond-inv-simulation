"""Chart generation (matplotlib, headless).

Charts are rendered dark to sit on the report's dark surface, using the
validated reference palette: categorical slots assigned in fixed order (never
cycled — beyond 8 series the remainder folds into a muted "Other"), a
blue↔red diverging pair with a neutral midpoint for signed values, and
recessive ink for chrome. Positive/negative always follows the diverging
pair (blue gain, red loss), not the status palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from .metrics import (  # noqa: E402
    drawdown_series,
    monthly_returns,
    rolling_sharpe,
    var_cvar,
    yearly_returns,
)

# Reference palette, dark mode (see docs of the dataviz method).
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
CATEGORICAL = [
    "#3987e5",  # blue
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
    "#d55181",  # magenta
    "#d95926",  # orange
]
BLUE, AQUA, YELLOW, RED = CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[5]
DIVERGING = LinearSegmentedColormap.from_list("gain_loss", [RED, BASELINE, BLUE])

matplotlib.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "axes.axisbelow": True,
    }
)


def series_colors(names: list[str]) -> dict[str, str]:
    """Fixed-order categorical assignment; entries beyond 8 slots go muted."""
    return {
        name: CATEGORICAL[i] if i < len(CATEGORICAL) else MUTED
        for i, name in enumerate(names)
    }


def _year_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


def _pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")


def _save(fig, output_path: Path) -> None:
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- tearsheet -----------------------------------------------------------------


def tearsheet(
    frame: pd.DataFrame,
    returns: pd.Series,
    rf_annual: pd.Series | float,
    label: str,
    output_path: Path,
) -> None:
    """Multi-panel performance/risk tearsheet for one asset or strategy.

    Panels: real value vs benchmarks, underwater drawdown, rolling Sharpe,
    monthly-return heatmap, daily-return distribution with VaR/CVaR, and
    yearly returns.
    """
    fig = plt.figure(figsize=(15, 18))
    grid = fig.add_gridspec(
        4, 2, height_ratios=[1.35, 1, 1.15, 1], hspace=0.45, wspace=0.18,
        top=0.955, bottom=0.035, left=0.06, right=0.98,
    )

    ax = fig.add_subplot(grid[0, :])
    _panel_value(ax, frame, label)

    ax = fig.add_subplot(grid[1, 0])
    _panel_drawdown(ax, returns)

    ax = fig.add_subplot(grid[1, 1])
    _panel_rolling_sharpe(ax, returns, rf_annual)

    ax = fig.add_subplot(grid[2, :])
    _panel_monthly_heatmap(ax, returns)

    ax = fig.add_subplot(grid[3, 0])
    _panel_histogram(ax, returns)

    ax = fig.add_subplot(grid[3, 1])
    _panel_yearly(ax, returns)

    fig.suptitle(
        f"{label} — biweekly DCA tearsheet", fontsize=15, fontweight="bold", color=INK, y=0.995
    )
    _save(fig, output_path)


def _panel_value(ax, frame: pd.DataFrame, label: str) -> None:
    dates = frame.index
    ax.plot(dates, frame["value_real"], color=BLUE, linewidth=2, label=f"{label} (real)")
    ax.plot(dates, frame["tbill_value_real"], color=YELLOW, linewidth=2, label="T-bills (real)")
    ax.plot(dates, frame["invested_real"], color=RED, linewidth=2, label="Cash (real)")
    ax.plot(
        dates, frame["invested"], color=MUTED, linewidth=1.2, linestyle="--",
        label="Contributions (nominal)",
    )
    ax.set_title("Value in start-of-period dollars vs doing nothing")
    ax.set_ylabel("$")
    ax.legend(loc="upper left")
    _year_axis(ax)


def _panel_drawdown(ax, returns: pd.Series) -> None:
    drawdown = drawdown_series(returns) * 100
    ax.fill_between(drawdown.index, drawdown, 0, color=RED, alpha=0.35, linewidth=0)
    ax.plot(drawdown.index, drawdown, color=RED, linewidth=1.2)
    trough = drawdown.idxmin()
    ax.plot([trough], [drawdown.min()], "o", color=RED, markersize=6)
    ax.annotate(
        f"{drawdown.min():.1f}%", (trough, drawdown.min()),
        textcoords="offset points", xytext=(6, -4), fontsize=8, color=INK,
    )
    ax.set_title("Drawdown (underwater)")
    _pct_axis(ax)
    _year_axis(ax)


def _panel_rolling_sharpe(ax, returns: pd.Series, rf_annual) -> None:
    rolling = rolling_sharpe(returns, rf_annual, window=126).dropna()
    if rolling.empty:
        ax.set_axis_off()
        return
    ax.plot(rolling.index, rolling, color=BLUE, linewidth=1.6)
    ax.axhline(0, color=BASELINE, linewidth=1)
    mean = rolling.mean()
    ax.axhline(mean, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"mean {mean:.2f}", (rolling.index[-1], mean),
        textcoords="offset points", xytext=(-40, 5), fontsize=8, color=MUTED,
    )
    ax.set_title("Rolling Sharpe ratio (6-month window)")
    _year_axis(ax)


def _panel_monthly_heatmap(ax, returns: pd.Series) -> None:
    table = monthly_returns(returns) * 100
    data = table.to_numpy(dtype=float)
    vmax = max(np.nanmax(np.abs(data)), 1.0)
    mesh_data = np.ma.masked_invalid(data)
    ax.pcolormesh(
        mesh_data,
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
        edgecolors=SURFACE,
        linewidth=2,
    )
    ax.set_xticks(np.arange(12) + 0.5)
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_yticks(np.arange(len(table.index)) + 0.5)
    ax.set_yticklabels(table.index)
    ax.invert_yaxis()
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if not np.isnan(data[row, col]):
                ax.text(
                    col + 0.5, row + 0.5, f"{data[row, col]:.1f}",
                    ha="center", va="center", fontsize=7.5, color=INK,
                )
    ax.grid(False)
    ax.set_title("Monthly returns (%) — blue gain, red loss")


def _panel_histogram(ax, returns: pd.Series) -> None:
    values = returns * 100
    var95, cvar95 = var_cvar(returns, 0.95)
    ax.hist(values, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ax.axvline(var95 * 100, color=YELLOW, linewidth=1.4, linestyle="--")
    ax.axvline(cvar95 * 100, color=RED, linewidth=1.4, linestyle="--")
    top = ax.get_ylim()[1]
    ax.annotate(
        f"VaR95 {var95 * 100:.1f}%", (var95 * 100, top * 0.95),
        fontsize=8, color=YELLOW, ha="right", rotation=90, va="top",
    )
    ax.annotate(
        f"CVaR95 {cvar95 * 100:.1f}%", (cvar95 * 100, top * 0.95),
        fontsize=8, color=RED, ha="right", rotation=90, va="top",
    )
    ax.set_title("Daily returns distribution")
    ax.set_xlabel("Daily return (%)")


def _panel_yearly(ax, returns: pd.Series) -> None:
    yearly = yearly_returns(returns) * 100
    colors = [BLUE if v >= 0 else RED for v in yearly]
    ax.bar(yearly.index.astype(str), yearly, color=colors, width=0.62)
    ax.axhline(0, color=BASELINE, linewidth=1)
    for x, v in zip(yearly.index.astype(str), yearly):
        ax.annotate(
            f"{v:.0f}", (x, v), textcoords="offset points",
            xytext=(0, 4 if v >= 0 else -11), ha="center", fontsize=8, color=INK_2,
        )
    ax.set_title("Yearly returns")
    _pct_axis(ax)


# --- cross-asset charts ---------------------------------------------------------


def risk_return_scatter(points: pd.DataFrame, output_path: Path) -> None:
    """CAGR vs volatility map. ``points``: label, annual_volatility_pct,
    cagr_pct, kind ('asset' | 'strategy')."""
    fig, ax = plt.subplots(figsize=(11, 7))
    for kind, color, marker, name in (
        ("asset", BLUE, "o", "Assets"),
        ("strategy", AQUA, "D", "Strategies"),
    ):
        subset = points[points["kind"] == kind]
        if subset.empty:
            continue
        ax.scatter(
            subset["annual_volatility_pct"], subset["cagr_pct"],
            s=70, color=color, marker=marker, label=name, zorder=3,
            edgecolors=SURFACE, linewidth=1,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["label"], (row["annual_volatility_pct"], row["cagr_pct"]),
                textcoords="offset points", xytext=(7, 4), fontsize=8.5, color=INK_2,
            )
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("Annual volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Risk vs return — up and left is better", color=INK)
    if (points["kind"] == "strategy").any():
        ax.legend(loc="upper left")
    _save(fig, output_path)


def correlation_heatmap(returns: pd.DataFrame, output_path: Path) -> None:
    """Correlation matrix of daily asset returns (diverging, annotated)."""
    corr = returns.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * n, 0.8 + 0.7 * n))
    ax.pcolormesh(
        corr.to_numpy(),
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-1, vmax=1),
        edgecolors=SURFACE,
        linewidth=2,
    )
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.invert_yaxis()
    for row in range(n):
        for col in range(n):
            ax.text(
                col + 0.5, row + 0.5, f"{corr.iloc[row, col]:.2f}",
                ha="center", va="center", fontsize=7.5, color=INK,
            )
    ax.grid(False)
    ax.set_title("Correlation of daily returns — blue +1, red −1", color=INK)
    _save(fig, output_path)


def strategy_drawdowns(returns_by_name: dict[str, pd.Series], output_path: Path) -> None:
    """Underwater curves of several strategies on one axis."""
    colors = series_colors(list(returns_by_name))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for name, returns in returns_by_name.items():
        drawdown = drawdown_series(returns) * 100
        ax.plot(drawdown.index, drawdown, color=colors[name], linewidth=1.6, label=name)
        ax.fill_between(drawdown.index, drawdown, 0, color=colors[name], alpha=0.12, linewidth=0)
    ax.set_title("Strategy drawdowns", color=INK)
    ax.legend(loc="lower left")
    _pct_axis(ax)
    _year_axis(ax)
    _save(fig, output_path)


def portfolio_comparison(frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Value and profit of several portfolio strategies on shared axes."""
    colors = series_colors(list(frames))
    fig, (ax_value, ax_profit) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for name, frame in frames.items():
        ax_value.plot(frame.index, frame["value"], label=name, color=colors[name], linewidth=1.8)
        ax_profit.plot(frame.index, frame["profit"], label=name, color=colors[name], linewidth=1.8)
    first = next(iter(frames.values()))
    ax_value.plot(
        first.index, first["invested"], "--", color=MUTED, linewidth=1.2, label="Contributions"
    )
    ax_profit.axhline(0, color=BASELINE, linewidth=1)
    ax_value.set_title("Portfolio value", color=INK)
    ax_profit.set_title("Profit", color=INK)
    for ax in (ax_value, ax_profit):
        ax.set_ylabel("$")
        ax.legend(loc="upper left")
        _year_axis(ax)
    _save(fig, output_path)


def weights_chart(weight_history: pd.DataFrame, output_path: Path) -> None:
    """Stacked area chart of dynamic portfolio target weights over time.

    At most 8 assets get their own hue; the smallest of the rest fold into a
    muted "Other" band so hues are never cycled.
    """
    history = weight_history
    if len(history.columns) > len(CATEGORICAL):
        ranked = history.mean().sort_values(ascending=False)
        keep = list(ranked.index[: len(CATEGORICAL) - 1])
        other = history.drop(columns=keep).sum(axis=1).rename("Other")
        history = pd.concat([history[keep], other], axis=1)
    colors = [
        MUTED if name == "Other" else CATEGORICAL[i]
        for i, name in enumerate(history.columns)
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(
        history.index,
        [history[c] for c in history.columns],
        labels=list(history.columns),
        colors=colors,
        alpha=0.9,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Dynamic portfolio target weights", color=INK)
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", ncol=2)
    _year_axis(ax)
    _save(fig, output_path)


def rolling_xirr_chart(per_ticker: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Money-weighted return of every rolling window, by window start date.

    Shows at most 8 series (by median XIRR); the rest are dropped with a note
    in the title rather than cycling hues.
    """
    ranked = sorted(
        (item for item in per_ticker.items() if not item[1].empty),
        key=lambda item: item[1]["xirr_pct"].median(),
        reverse=True,
    )
    dropped = max(0, len(ranked) - len(CATEGORICAL))
    ranked = ranked[: len(CATEGORICAL)]
    colors = series_colors([name for name, _ in ranked])
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for name, windows in ranked:
        starts = pd.to_datetime(windows["window_start"])
        ax.plot(starts, windows["xirr_pct"], linewidth=1.8, label=name, color=colors[name])
    ax.axhline(0, color=BASELINE, linewidth=1)
    title = "DCA outcome by start date (rolling windows)"
    if dropped:
        title += f" — top {len(ranked)} by median XIRR, {dropped} omitted"
    ax.set_title(title, color=INK)
    ax.set_ylabel("XIRR (%/yr)")
    ax.set_xlabel("Window start")
    ax.legend(ncol=2)
    _year_axis(ax)
    _save(fig, output_path)
