#!/usr/bin/env python3
"""
Grid Search for Portfolio Hyperparameters
Пошук оптимальних гіперпараметрів для динамічного портфеля.
"""

import argparse
import itertools
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from optimize_portfolio import (
    load_price_data,
    simulate_dynamic_portfolio,
    simulate_portfolio_investment,
    load_returns_data,
    load_metrics,
    optimize_portfolio,
)


def run_single_simulation(
    prices_df: pd.DataFrame,
    lookback_years: int,
    rebalance_months: int,
    max_weight_change: float,
    investment_amount: float = 500.0,
    max_weight: float = 0.4
) -> dict:
    """
    Запустити одну симуляцію з заданими параметрами.

    Returns:
        dict з результатами або None якщо помилка
    """
    try:
        result = simulate_dynamic_portfolio(
            prices_df,
            investment_amount=investment_amount,
            lookback_years=lookback_years,
            rebalance_months=rebalance_months,
            max_weight=max_weight,
            max_weight_change=max_weight_change
        )

        if result is None:
            return None

        # Розраховуємо метрики
        final_value = result['portfolio_value'][-1]
        total_invested = result['total_invested'][-1]
        profit = result['profit'][-1]
        profit_pct = result['profit_pct'][-1]

        # Розраховуємо max drawdown
        portfolio_values = pd.Series(result['portfolio_value'])
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Розраховуємо волатильність
        returns = portfolio_values.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        # Sharpe ratio (спрощений)
        annual_return = profit_pct / (len(result['dates']) / 252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        return {
            'final_value': final_value,
            'total_invested': total_invested,
            'profit': profit,
            'profit_pct': profit_pct,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe': sharpe,
            'num_rebalances': len(result['rebalance_dates']),
            'investment_days': len(result['dates']),
        }

    except Exception as e:
        print(f"  Помилка: {e}")
        return None


def grid_search(
    prices_df: pd.DataFrame,
    lookback_range: list,
    rebalance_range: list,
    max_change_range: list,
    investment_amount: float = 500.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Провести grid search по всіх комбінаціях параметрів.

    Returns:
        DataFrame з результатами всіх комбінацій
    """
    results = []
    total_combinations = len(lookback_range) * len(rebalance_range) * len(max_change_range)

    print(f"\n{'='*70}")
    print(f"GRID SEARCH: {total_combinations} комбінацій")
    print(f"{'='*70}")
    print(f"Lookback (років):     {lookback_range}")
    print(f"Rebalance (місяців):  {rebalance_range}")
    print(f"Max change:           {max_change_range}")
    print(f"{'='*70}\n")

    for i, (lookback, rebalance, max_change) in enumerate(
        itertools.product(lookback_range, rebalance_range, max_change_range)
    ):
        if verbose:
            print(f"[{i+1}/{total_combinations}] lookback={lookback}, rebalance={rebalance}, max_change={max_change:.2f}", end=" ")

        result = run_single_simulation(
            prices_df,
            lookback_years=lookback,
            rebalance_months=rebalance,
            max_weight_change=max_change,
            investment_amount=investment_amount
        )

        if result:
            result['lookback'] = lookback
            result['rebalance'] = rebalance
            result['max_change'] = max_change
            results.append(result)

            if verbose:
                print(f"-> profit: {result['profit_pct']:.1f}%, sharpe: {result['sharpe']:.2f}")
        else:
            if verbose:
                print("-> SKIP (недостатньо даних)")

    return pd.DataFrame(results)


def print_results(df: pd.DataFrame, top_n: int = 10):
    """Вивести топ результатів."""

    print(f"\n{'='*90}")
    print(f"ТОП-{top_n} ЗА ПРИБУТКОМ (%)")
    print(f"{'='*90}")

    top_profit = df.nlargest(top_n, 'profit_pct')
    print(top_profit[['lookback', 'rebalance', 'max_change', 'profit_pct', 'max_drawdown', 'sharpe']].to_string(index=False))

    print(f"\n{'='*90}")
    print(f"ТОП-{top_n} ЗА SHARPE RATIO")
    print(f"{'='*90}")

    top_sharpe = df.nlargest(top_n, 'sharpe')
    print(top_sharpe[['lookback', 'rebalance', 'max_change', 'profit_pct', 'max_drawdown', 'sharpe']].to_string(index=False))

    print(f"\n{'='*90}")
    print(f"ТОП-{top_n} ЗА RISK-ADJUSTED (profit / |max_dd|)")
    print(f"{'='*90}")

    df['risk_adjusted'] = df['profit_pct'] / df['max_drawdown'].abs()
    top_risk = df.nlargest(top_n, 'risk_adjusted')
    print(top_risk[['lookback', 'rebalance', 'max_change', 'profit_pct', 'max_drawdown', 'risk_adjusted']].to_string(index=False))

    # Найкращі параметри
    best_profit = df.loc[df['profit_pct'].idxmax()]
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    best_risk = df.loc[df['risk_adjusted'].idxmax()]

    print(f"\n{'='*90}")
    print("РЕКОМЕНДОВАНІ ПАРАМЕТРИ")
    print(f"{'='*90}")

    print(f"\n🏆 Максимальний прибуток:")
    print(f"   lookback={int(best_profit['lookback'])}, rebalance={int(best_profit['rebalance'])}, max_change={best_profit['max_change']:.2f}")
    print(f"   Прибуток: {best_profit['profit_pct']:.1f}%, Max DD: {best_profit['max_drawdown']:.1f}%")

    print(f"\n📊 Найкращий Sharpe:")
    print(f"   lookback={int(best_sharpe['lookback'])}, rebalance={int(best_sharpe['rebalance'])}, max_change={best_sharpe['max_change']:.2f}")
    print(f"   Прибуток: {best_sharpe['profit_pct']:.1f}%, Sharpe: {best_sharpe['sharpe']:.2f}")

    print(f"\n⚖️ Найкращий risk-adjusted:")
    print(f"   lookback={int(best_risk['lookback'])}, rebalance={int(best_risk['rebalance'])}, max_change={best_risk['max_change']:.2f}")
    print(f"   Прибуток: {best_risk['profit_pct']:.1f}%, Max DD: {best_risk['max_drawdown']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Grid Search для оптимізації гіперпараметрів портфеля',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  python grid_search.py
  python grid_search.py --lookback 2 3 4 5 --rebalance 1 3 6
  python grid_search.py --max-change 0.05 0.10 0.15 0.20
        """
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='./simulation_results',
        help='Директорія з результатами симуляцій'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        nargs='+',
        default=[2, 3, 4, 5],
        help='Значення lookback для перебору (роки)'
    )

    parser.add_argument(
        '--rebalance',
        type=int,
        nargs='+',
        default=[1, 3, 6, 12],
        help='Значення rebalance для перебору (місяці)'
    )

    parser.add_argument(
        '--max-change',
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.15, 0.20],
        help='Значення max_change для перебору'
    )

    parser.add_argument(
        '--amount',
        type=float,
        default=500.0,
        help='Сума інвестиції кожні 2 тижні'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='grid_search_results.csv',
        help='Файл для збереження результатів'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Тихий режим (менше виводу)'
    )

    args = parser.parse_args()

    results_dir = Path(args.dir)

    if not results_dir.exists():
        print(f"Директорія {results_dir} не існує")
        print("Спочатку запустіть ./run_simulations.sh")
        return

    # Завантажуємо дані
    print("Завантаження даних...")
    prices_df = load_price_data(results_dir)

    if prices_df.empty:
        print("Не знайдено даних для оптимізації")
        return

    print(f"Знайдено {len(prices_df.columns)} активів: {', '.join(prices_df.columns)}")
    print(f"Період: {len(prices_df)} торгових днів")
    print(f"Дати: {prices_df.index[0].strftime('%Y-%m-%d')} - {prices_df.index[-1].strftime('%Y-%m-%d')}")

    # Grid search
    results_df = grid_search(
        prices_df,
        lookback_range=args.lookback,
        rebalance_range=args.rebalance,
        max_change_range=args.max_change,
        investment_amount=args.amount,
        verbose=not args.quiet
    )

    if results_df.empty:
        print("Немає результатів")
        return

    # Виводимо результати
    print_results(results_df)

    # Зберігаємо
    output_path = results_dir / args.output
    results_df.to_csv(output_path, index=False)
    print(f"\n📊 Результати збережено в: {output_path}")


if __name__ == '__main__':
    main()
