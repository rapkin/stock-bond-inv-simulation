"""Chart generation (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

plt.style.use("seaborn-v0_8-darkgrid")


def _format_year_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


def asset_dashboard(frame: pd.DataFrame, label: str, output_path: Path) -> None:
    """2×2 dashboard: price, value vs cash, profit, real profit vs T-bills."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    dates = frame.index

    ax = axes[0, 0]
    ax.plot(dates, frame["price"], linewidth=1.2, color="tab:blue")
    ax.set_title(f"{label} price")
    ax.set_ylabel("$")

    ax = axes[0, 1]
    ax.plot(dates, frame["value"], label="Portfolio (nominal)", color="tab:blue")
    ax.plot(dates, frame["value_real"], label="Portfolio (real)", color="tab:green")
    ax.plot(dates, frame["invested"], "--", label="Contributions (nominal)", color="tab:red")
    ax.plot(
        dates,
        frame["invested_real"],
        label="Cash under mattress (real)",
        color="tab:red",
        alpha=0.6,
    )
    ax.set_title("Portfolio value vs holding cash")
    ax.set_ylabel("$")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(dates, frame["profit"], label="Profit (nominal)", color="tab:blue")
    ax.plot(dates, frame["profit_real"], label="Profit (real)", color="tab:green")
    ax.axhline(0, color="tab:red", linestyle="--", alpha=0.5)
    ax.fill_between(dates, frame["profit"], 0, where=frame["profit"] >= 0, alpha=0.2, color="green")
    ax.fill_between(dates, frame["profit"], 0, where=frame["profit"] < 0, alpha=0.2, color="red")
    ax.set_title("Profit / loss")
    ax.set_ylabel("$")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(dates, frame["value_real"], label=f"{label} (real)", color="tab:blue")
    ax.plot(dates, frame["tbill_value_real"], label="T-bills (real)", color="tab:orange")
    ax.plot(dates, frame["invested_real"], label="Cash (real)", color="tab:red", alpha=0.6)
    ax.set_title("Real value: asset vs T-bills vs cash")
    ax.set_ylabel("$")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        _format_year_axis(ax)
    fig.suptitle(f"DCA simulation: {label}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def portfolio_comparison(frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Value and profit of several portfolio strategies on shared axes."""
    fig, (ax_value, ax_profit) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for color, (name, frame) in zip(colors, frames.items()):
        ax_value.plot(frame.index, frame["value"], label=name, color=color, linewidth=1.4)
        ax_profit.plot(frame.index, frame["profit"], label=name, color=color, linewidth=1.4)
    first = next(iter(frames.values()))
    ax_value.plot(first.index, first["invested"], "--", color="tab:red", label="Contributions")
    ax_profit.axhline(0, color="tab:red", linestyle="--", alpha=0.5)
    ax_value.set_title("Portfolio value")
    ax_profit.set_title("Profit")
    for ax in (ax_value, ax_profit):
        ax.set_ylabel("$")
        ax.legend(fontsize=9)
        _format_year_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def rolling_xirr_chart(per_ticker: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Money-weighted return of every rolling window, by window start date."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, windows in per_ticker.items():
        if windows.empty:
            continue
        starts = pd.to_datetime(windows["window_start"])
        ax.plot(starts, windows["xirr_pct"], linewidth=1.4, label=name)
    ax.axhline(0, color="tab:red", linestyle="--", alpha=0.5)
    ax.set_title("DCA outcome by start date (rolling windows)")
    ax.set_ylabel("XIRR (%/yr)")
    ax.set_xlabel("Window start")
    ax.legend(fontsize=8, ncol=2)
    _format_year_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def weights_chart(weight_history: pd.DataFrame, output_path: Path) -> None:
    """Stacked area chart of dynamic portfolio target weights over time."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(
        weight_history.index,
        [weight_history[c] for c in weight_history.columns],
        labels=list(weight_history.columns),
        alpha=0.85,
    )
    ax.set_ylim(0, 1)
    ax.set_title("Dynamic portfolio target weights")
    ax.set_ylabel("Weight")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    _format_year_axis(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
