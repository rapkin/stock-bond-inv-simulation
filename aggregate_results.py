#!/usr/bin/env python3
"""
Aggregate Results Script
Агрегує метрики з усіх симуляцій в одну таблицю для порівняння.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_metrics_files(results_dir: Path) -> list[Path]:
    """Знайти всі файли metrics.csv в директорії результатів."""
    return list(results_dir.glob('*/metrics.csv'))


def aggregate_metrics(results_dir: Path) -> pd.DataFrame:
    """
    Агрегувати всі метрики в один DataFrame.

    Args:
        results_dir: Директорія з результатами симуляцій

    Returns:
        DataFrame з усіма метриками
    """
    metrics_files = find_metrics_files(results_dir)

    if not metrics_files:
        print(f"Не знайдено файлів metrics.csv в {results_dir}")
        return pd.DataFrame()

    dfs = []
    for file in metrics_files:
        try:
            df = pd.read_csv(file)
            df['source_dir'] = file.parent.name
            dfs.append(df)
        except Exception as e:
            print(f"Помилка читання {file}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def calculate_risk_reward_score(df: pd.DataFrame) -> pd.Series:
    """
    Розрахувати комбіновану метрику Risk-Reward Score.

    Score = Sortino × (1 - |max_drawdown|/100) × ln(1 + total_return/100)

    Враховує:
    - Sortino Ratio: дохідність відносно негативної волатильності
    - Штраф за великі просадки (max drawdown)
    - Логарифм прибутку (зменшує вплив екстремальних значень)
    """
    sortino = df['sortino_ratio'].fillna(0)
    max_dd = df['max_drawdown_pct'].abs().fillna(100)
    total_return = df['total_return_pct'].fillna(0)

    # Drawdown penalty: 0% DD = 1.0, 50% DD = 0.5, 100% DD = 0.0
    dd_factor = 1 - (max_dd / 100)

    # Log return factor: 0% = 0, 100% = 0.69, 1000% = 2.4
    return_factor = np.log1p(total_return.clip(lower=0) / 100)

    score = sortino * dd_factor * return_factor

    return score


def format_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Форматувати таблицю для зручного порівняння."""
    if df.empty:
        return df

    # Розраховуємо Risk-Reward Score
    if all(c in df.columns for c in ['sortino_ratio', 'max_drawdown_pct', 'total_return_pct']):
        df = df.copy()
        df['risk_reward_score'] = calculate_risk_reward_score(df)

    # Порядок колонок (risk_reward_score на початку після базових)
    priority_columns = [
        'ticker',
        'start_year',
        'end_year',
        'risk_reward_score',
    ]

    # Всі інші колонки крім службових
    exclude = ['source_dir', 'investment_amount', 'risk_free_rate_pct']
    other_columns = [c for c in df.columns if c not in priority_columns and c not in exclude]

    columns = priority_columns + other_columns
    columns = [c for c in columns if c in df.columns]

    df = df[columns].copy()

    # Сортуємо за Risk-Reward Score (найкращі зверху)
    if 'risk_reward_score' in df.columns:
        df = df.sort_values('risk_reward_score', ascending=False)
    elif 'sharpe_ratio' in df.columns:
        df = df.sort_values('sharpe_ratio', ascending=False)

    return df


def print_comparison_table(df: pd.DataFrame):
    """Вивести таблицю порівняння."""
    if df.empty:
        print("Немає даних для порівняння")
        return

    sort_col = 'risk_reward_score' if 'risk_reward_score' in df.columns else 'sharpe_ratio'
    print("\n" + "=" * 120)
    print(f"ПОРІВНЯННЯ СИМУЛЯЦІЙ (відсортовано за {sort_col})")
    print("=" * 120)

    # Форматуємо числові колонки
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    print(df.to_string(index=False))
    print("=" * 120)

    # Рекомендації
    if len(df) > 1 and 'risk_reward_score' in df.columns:
        best = df.iloc[0]
        print(f"\n🏆 Найкращий за Risk-Reward Score: {best['ticker']}")
        print(f"   Score: {best['risk_reward_score']:.2f}, "
              f"Return: {best['total_return_pct']:.1f}%, "
              f"Sortino: {best['sortino_ratio']:.2f}, "
              f"Max DD: {best['max_drawdown_pct']:.1f}%")
        print(f"\n📊 Risk-Reward Score = Sortino × (1 - |MaxDD|/100) × ln(1 + Return/100)")


def main():
    parser = argparse.ArgumentParser(
        description='Агрегація метрик з усіх симуляцій',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  python aggregate_results.py
  python aggregate_results.py --dir ./my_results
  python aggregate_results.py --output comparison.csv
        """
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='./simulation_results',
        help='Директорія з результатами симуляцій. За замовчуванням: ./simulation_results'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./simulation_results/comparison.csv',
        help='Файл для збереження агрегованих результатів'
    )

    parser.add_argument(
        '--sort', '-s',
        type=str,
        default='risk_reward_score',
        choices=['risk_reward_score', 'sharpe_ratio', 'sortino_ratio', 'cagr_pct', 'total_return_pct', 'max_drawdown_pct'],
        help='Колонка для сортування. За замовчуванням: risk_reward_score'
    )

    args = parser.parse_args()

    results_dir = Path(args.dir)

    if not results_dir.exists():
        print(f"Директорія {results_dir} не існує")
        return

    # Агрегуємо метрики
    df = aggregate_metrics(results_dir)

    if df.empty:
        print("Не знайдено метрик для агрегації")
        return

    # Форматуємо таблицю
    comparison_df = format_comparison_table(df)

    # Сортуємо за вибраною колонкою
    if args.sort in comparison_df.columns:
        ascending = args.sort == 'max_drawdown_pct'  # Для drawdown менше = краще
        comparison_df = comparison_df.sort_values(args.sort, ascending=ascending)

    # Виводимо таблицю
    print_comparison_table(comparison_df)

    # Зберігаємо результати
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    print(f"\nРезультати збережено в: {output_path}")


if __name__ == '__main__':
    main()
