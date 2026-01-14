#!/usr/bin/env python3
"""
Aggregate Results Script
Агрегує метрики з усіх симуляцій в одну таблицю для порівняння.
"""

import argparse
from pathlib import Path

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


def format_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Форматувати таблицю для зручного порівняння."""
    if df.empty:
        return df

    # Вибираємо ключові колонки для порівняння
    columns = [
        'ticker',
        'start_year',
        'end_year',
        'total_invested',
        'final_value',
        'total_return_pct',
        'cagr_pct',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown_pct',
        'calmar_ratio',
        'annual_volatility_pct',
        'win_rate_pct',
        'max_underwater_days',
    ]

    # Залишаємо тільки наявні колонки
    columns = [c for c in columns if c in df.columns]

    df = df[columns].copy()

    # Сортуємо за Sharpe Ratio (найкращі зверху)
    if 'sharpe_ratio' in df.columns:
        df = df.sort_values('sharpe_ratio', ascending=False)

    return df


def print_comparison_table(df: pd.DataFrame):
    """Вивести таблицю порівняння."""
    if df.empty:
        print("Немає даних для порівняння")
        return

    print("\n" + "=" * 100)
    print("ПОРІВНЯННЯ СИМУЛЯЦІЙ (відсортовано за Sharpe Ratio)")
    print("=" * 100)

    # Форматуємо числові колонки
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    print(df.to_string(index=False))
    print("=" * 100)

    # Рекомендації
    if len(df) > 1 and 'sharpe_ratio' in df.columns:
        best = df.iloc[0]
        print(f"\nНайкращий за Sharpe Ratio: {best['ticker']}")
        print(f"  Sharpe: {best['sharpe_ratio']:.2f}, "
              f"CAGR: {best['cagr_pct']:.2f}%, "
              f"Max DD: {best['max_drawdown_pct']:.2f}%")


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
        default='sharpe_ratio',
        choices=['sharpe_ratio', 'sortino_ratio', 'cagr_pct', 'total_return_pct', 'max_drawdown_pct'],
        help='Колонка для сортування. За замовчуванням: sharpe_ratio'
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
