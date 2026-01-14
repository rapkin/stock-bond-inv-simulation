#!/usr/bin/env python3
"""
Generate HTML Report
Генерує HTML звіт з оглядом портфелів та результатів симуляцій.
"""

import base64
from pathlib import Path
from datetime import datetime

import pandas as pd


def image_to_base64(image_path: Path) -> str:
    """Конвертувати зображення в base64."""
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_html_report(results_dir: Path, output_file: Path = None):
    """Генерувати HTML звіт."""

    if output_file is None:
        output_file = results_dir / "report.html"

    # Завантажуємо дані
    comparison_file = results_dir / "comparison.csv"
    portfolio_file = results_dir / "optimal_portfolio.csv"
    simulation_file = results_dir / "portfolio_simulation.csv"

    comparison_df = pd.read_csv(comparison_file) if comparison_file.exists() else pd.DataFrame()
    portfolio_df = pd.read_csv(portfolio_file) if portfolio_file.exists() else pd.DataFrame()
    simulation_df = pd.read_csv(simulation_file) if simulation_file.exists() else pd.DataFrame()

    # Зображення
    portfolio_chart = results_dir / "portfolio_comparison.png"

    # Генеруємо HTML
    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Investment Portfolio Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --danger: #dc2626;
            --warning: #d97706;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--border);
        }}

        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: var(--text-muted);
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
        }}

        .card h2 {{
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .card h3 {{
            font-size: 1rem;
            margin: 1rem 0 0.5rem;
            color: var(--text-muted);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        th, td {{
            padding: 0.5rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            font-weight: 600;
            color: var(--text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
        }}

        tr:hover {{
            background: var(--bg);
        }}

        .text-right {{
            text-align: right;
        }}

        .text-center {{
            text-align: center;
        }}

        .positive {{
            color: var(--success);
            font-weight: 600;
        }}

        .negative {{
            color: var(--danger);
            font-weight: 600;
        }}

        .weight-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
        }}

        .weight-bar-fill {{
            height: 100%;
            background: var(--primary);
            border-radius: 4px;
        }}

        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
        }}

        .metric:last-child {{
            border-bottom: none;
        }}

        .metric-label {{
            color: var(--text-muted);
        }}

        .metric-value {{
            font-weight: 600;
        }}

        .chart-container {{
            margin: 2rem 0;
        }}

        .chart-container img {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .summary-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--border);
        }}

        .summary-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}

        .summary-card .label {{
            color: var(--text-muted);
            font-size: 0.85rem;
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .badge-success {{
            background: #dcfce7;
            color: var(--success);
        }}

        .badge-danger {{
            background: #fee2e2;
            color: var(--danger);
        }}

        .badge-warning {{
            background: #fef3c7;
            color: var(--warning);
        }}

        footer {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.85rem;
        }}

        @media (max-width: 768px) {{
            .summary-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Investment Portfolio Report</h1>
            <p class="subtitle">Згенеровано: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </header>
"""

    # Summary cards
    if not simulation_df.empty:
        opt_profit = simulation_df['optimal_profit'].iloc[-1]
        eq_profit = simulation_df['equal_profit'].iloc[-1]
        total_invested = simulation_df['total_invested'].iloc[-1]

        opt_pct = (opt_profit / total_invested * 100) if total_invested > 0 else 0
        eq_pct = (eq_profit / total_invested * 100) if total_invested > 0 else 0

        html += f"""
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">${total_invested:,.0f}</div>
                <div class="label">Інвестовано</div>
            </div>
            <div class="summary-card">
                <div class="value {'positive' if opt_profit > 0 else 'negative'}">${opt_profit:,.0f}</div>
                <div class="label">Прибуток (Оптимальний)</div>
            </div>
            <div class="summary-card">
                <div class="value {'positive' if eq_profit > 0 else 'negative'}">${eq_profit:,.0f}</div>
                <div class="label">Прибуток (Рівномірний)</div>
            </div>
            <div class="summary-card">
                <div class="value">{len(comparison_df)}</div>
                <div class="label">Активів</div>
            </div>
        </div>
"""

    html += '<div class="grid">'

    # Optimal Portfolio Card
    if not portfolio_df.empty:
        html += """
        <div class="card">
            <h2>🎯 Оптимальний портфель</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.9rem;">
                Maximum Sharpe Ratio оптимізація
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Актив</th>
                        <th class="text-right">Вага</th>
                        <th style="width: 100px;"></th>
                    </tr>
                </thead>
                <tbody>
"""
        for _, row in portfolio_df.iterrows():
            if row['weight_pct'] > 0.1:
                html += f"""
                    <tr>
                        <td><strong>{row['ticker']}</strong></td>
                        <td class="text-right">{row['weight_pct']:.1f}%</td>
                        <td>
                            <div class="weight-bar">
                                <div class="weight-bar-fill" style="width: {row['weight_pct']}%;"></div>
                            </div>
                        </td>
                    </tr>
"""
        html += """
                </tbody>
            </table>
        </div>
"""

    # Equal Weight Portfolio Card
    if not comparison_df.empty:
        n_assets = len(comparison_df)
        equal_weight = 100 / n_assets
        html += f"""
        <div class="card">
            <h2>⚖️ Рівномірний портфель</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.9rem;">
                Однакова вага для всіх активів
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Актив</th>
                        <th class="text-right">Вага</th>
                        <th style="width: 100px;"></th>
                    </tr>
                </thead>
                <tbody>
"""
        for ticker in comparison_df['ticker']:
            html += f"""
                    <tr>
                        <td><strong>{ticker}</strong></td>
                        <td class="text-right">{equal_weight:.1f}%</td>
                        <td>
                            <div class="weight-bar">
                                <div class="weight-bar-fill" style="width: {equal_weight}%; background: var(--warning);"></div>
                            </div>
                        </td>
                    </tr>
"""
        html += """
                </tbody>
            </table>
        </div>
"""

    html += '</div>'

    # Chart
    if portfolio_chart.exists():
        chart_b64 = image_to_base64(portfolio_chart)
        html += f"""
        <div class="card chart-container">
            <h2>📈 Порівняння портфелів</h2>
            <img src="data:image/png;base64,{chart_b64}" alt="Portfolio Comparison">
        </div>
"""

    # Assets comparison table
    if not comparison_df.empty:
        html += """
        <div class="card">
            <h2>📋 Порівняння активів (Risk-Reward Score)</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Актив</th>
                            <th class="text-right">Score</th>
                            <th class="text-right">Return</th>
                            <th class="text-right">CAGR</th>
                            <th class="text-right">Sharpe</th>
                            <th class="text-right">Sortino</th>
                            <th class="text-right">Max DD</th>
                            <th class="text-right">Volatility</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        for _, row in comparison_df.iterrows():
            score = row.get('risk_reward_score', 0)
            ret = row.get('total_return_pct', 0)
            cagr = row.get('cagr_pct', 0)
            sharpe = row.get('sharpe_ratio', 0)
            sortino = row.get('sortino_ratio', 0)
            max_dd = row.get('max_drawdown_pct', 0)
            vol = row.get('annual_volatility_pct', 0)

            ret_class = 'positive' if ret > 0 else 'negative'
            dd_class = 'negative' if max_dd < -30 else ''

            html += f"""
                        <tr>
                            <td><strong>{row['ticker']}</strong></td>
                            <td class="text-right">{score:.2f}</td>
                            <td class="text-right {ret_class}">{ret:.1f}%</td>
                            <td class="text-right">{cagr:.1f}%</td>
                            <td class="text-right">{sharpe:.2f}</td>
                            <td class="text-right">{sortino:.2f}</td>
                            <td class="text-right {dd_class}">{max_dd:.1f}%</td>
                            <td class="text-right">{vol:.1f}%</td>
                        </tr>
"""
        html += """
                    </tbody>
                </table>
            </div>
            <p style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted);">
                📊 Risk-Reward Score = Sortino × √(1-|MaxDD|/100) × √(Return/100) × suspicion_penalty
            </p>
        </div>
"""

    # Simulation results
    if not simulation_df.empty:
        opt_final = simulation_df['optimal_value'].iloc[-1]
        eq_final = simulation_df['equal_value'].iloc[-1]
        total_inv = simulation_df['total_invested'].iloc[-1]

        html += f"""
        <div class="card">
            <h2>💰 Результати симуляції</h2>
            <p style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.9rem;">
                Інвестиція $500 кожні 2 тижні
            </p>
            <div class="grid" style="grid-template-columns: 1fr 1fr;">
                <div>
                    <h3>Оптимальний портфель</h3>
                    <div class="metric">
                        <span class="metric-label">Фінальна вартість</span>
                        <span class="metric-value">${opt_final:,.0f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Прибуток</span>
                        <span class="metric-value {'positive' if opt_final > total_inv else 'negative'}">
                            ${opt_final - total_inv:,.0f} ({(opt_final/total_inv - 1)*100:.1f}%)
                        </span>
                    </div>
                </div>
                <div>
                    <h3>Рівномірний портфель</h3>
                    <div class="metric">
                        <span class="metric-label">Фінальна вартість</span>
                        <span class="metric-value">${eq_final:,.0f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Прибуток</span>
                        <span class="metric-value {'positive' if eq_final > total_inv else 'negative'}">
                            ${eq_final - total_inv:,.0f} ({(eq_final/total_inv - 1)*100:.1f}%)
                        </span>
                    </div>
                </div>
            </div>
            <div style="margin-top: 1rem; padding: 1rem; background: var(--bg); border-radius: 8px;">
                <strong>{'🏆 Переможець: Рівномірний портфель' if eq_final > opt_final else '🏆 Переможець: Оптимальний портфель'}</strong>
                <span style="color: var(--text-muted);"> (різниця: ${abs(eq_final - opt_final):,.0f})</span>
            </div>
        </div>
"""

    html += """
        <footer>
            <p>Generated by Investment Simulation Tool</p>
        </footer>
    </div>
</body>
</html>
"""

    # Зберігаємо файл
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"📄 HTML звіт збережено: {output_file}")
    return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Генерація HTML звіту')
    parser.add_argument('--dir', '-d', type=str, default='./simulation_results',
                        help='Директорія з результатами')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Вихідний файл (за замовчуванням: report.html)')

    args = parser.parse_args()

    results_dir = Path(args.dir)
    output_file = Path(args.output) if args.output else None

    if not results_dir.exists():
        print(f"Директорія {results_dir} не існує")
        return

    generate_html_report(results_dir, output_file)


if __name__ == '__main__':
    main()
