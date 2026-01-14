#!/usr/bin/env python3
"""
Portfolio Optimization Script
Оптимізація портфеля на основі результатів симуляцій.

Використовує Modern Portfolio Theory (MPT) для знаходження
оптимальних ваг активів з урахуванням кореляцій.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def load_returns_data(results_dir: Path) -> pd.DataFrame:
    """
    Завантажити денні дохідності з усіх симуляцій.

    Returns:
        DataFrame з денними дохідностями для кожного тікера
    """
    returns_dict = {}

    for sim_dir in results_dir.iterdir():
        if not sim_dir.is_dir():
            continue

        data_file = sim_dir / 'simulation_data.csv'
        metrics_file = sim_dir / 'metrics.csv'

        if not data_file.exists() or not metrics_file.exists():
            continue

        try:
            # Отримуємо тікер з метрик
            metrics = pd.read_csv(metrics_file)
            ticker = metrics['ticker'].iloc[0]

            # Завантажуємо дані портфеля
            data = pd.read_csv(data_file, parse_dates=['date'])
            data = data.set_index('date')

            # Розраховуємо денні дохідності портфеля
            portfolio_values = data['portfolio_value']
            # Фільтруємо нульові значення
            portfolio_values = portfolio_values[portfolio_values > 0]
            returns = portfolio_values.pct_change().dropna()

            # Видаляємо нескінченні та занадто великі значення
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            returns = returns[returns.abs() < 1]  # Видаляємо >100% денні зміни

            if len(returns) > 100:  # Мінімум 100 днів даних
                returns_dict[ticker] = returns

        except Exception as e:
            print(f"Помилка завантаження {sim_dir.name}: {e}")

    if not returns_dict:
        return pd.DataFrame()

    # Об'єднуємо в один DataFrame
    returns_df = pd.DataFrame(returns_dict)

    # Залишаємо тільки дати де є дані для всіх активів
    returns_df = returns_df.dropna()

    return returns_df


def load_metrics(results_dir: Path) -> pd.DataFrame:
    """Завантажити метрики з comparison.csv або з файлів метрик."""
    comparison_file = results_dir / 'comparison.csv'

    if comparison_file.exists():
        return pd.read_csv(comparison_file)

    # Якщо немає comparison.csv, збираємо з окремих файлів
    metrics_list = []
    for metrics_file in results_dir.glob('*/metrics.csv'):
        try:
            df = pd.read_csv(metrics_file)
            metrics_list.append(df)
        except:
            pass

    if metrics_list:
        return pd.concat(metrics_list, ignore_index=True)

    return pd.DataFrame()


def calculate_portfolio_stats(weights: np.ndarray,
                               mean_returns: np.ndarray,
                               cov_matrix: np.ndarray) -> tuple:
    """Розрахувати очікувану дохідність та волатильність портфеля."""
    portfolio_return = np.sum(mean_returns * weights) * 252  # Річна
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return portfolio_return, portfolio_volatility


def negative_sharpe(weights: np.ndarray,
                    mean_returns: np.ndarray,
                    cov_matrix: np.ndarray,
                    risk_free_rate: float = 0.02) -> float:
    """Негативний Sharpe Ratio (для мінімізації)."""
    p_return, p_volatility = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility


def portfolio_volatility(weights: np.ndarray,
                         mean_returns: np.ndarray,
                         cov_matrix: np.ndarray) -> float:
    """Волатильність портфеля (для мінімізації)."""
    _, p_volatility = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
    return p_volatility


def optimize_portfolio(returns_df: pd.DataFrame,
                       metrics_df: pd.DataFrame,
                       optimization_target: str = 'sharpe',
                       max_weight: float = 0.4,
                       min_weight: float = 0.0) -> dict:
    """
    Оптимізувати портфель.

    Args:
        returns_df: DataFrame з денними дохідностями
        metrics_df: DataFrame з метриками
        optimization_target: 'sharpe' (макс Sharpe) або 'min_var' (мін волатильність)
        max_weight: Максимальна вага одного активу
        min_weight: Мінімальна вага одного активу

    Returns:
        dict з результатами оптимізації
    """
    tickers = returns_df.columns.tolist()
    n_assets = len(tickers)

    # Середні дохідності та коваріаційна матриця
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    # Початкові ваги (рівномірний розподіл)
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    # Обмеження
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Сума ваг = 1
    ]

    # Межі для кожного активу
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

    # Вибір цільової функції
    if optimization_target == 'sharpe':
        objective = negative_sharpe
    else:  # min_var
        objective = portfolio_volatility

    # Оптимізація
    result = minimize(
        objective,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        print(f"Попередження: оптимізація не збіглась - {result.message}")

    optimal_weights = result.x

    # Розрахунок статистики оптимального портфеля
    opt_return, opt_volatility = calculate_portfolio_stats(
        optimal_weights, mean_returns, cov_matrix
    )
    opt_sharpe = (opt_return - 0.02) / opt_volatility

    # Зважений max drawdown
    if 'max_drawdown_pct' in metrics_df.columns:
        ticker_dd = metrics_df.set_index('ticker')['max_drawdown_pct'].to_dict()
        weighted_dd = sum(
            optimal_weights[i] * ticker_dd.get(t, 0)
            for i, t in enumerate(tickers)
        )
    else:
        weighted_dd = None

    return {
        'tickers': tickers,
        'weights': optimal_weights,
        'expected_return': opt_return,
        'volatility': opt_volatility,
        'sharpe_ratio': opt_sharpe,
        'weighted_max_drawdown': weighted_dd,
        'correlation_matrix': pd.DataFrame(
            returns_df.corr(),
            index=tickers,
            columns=tickers
        ),
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
    }


def generate_efficient_frontier(returns_df: pd.DataFrame,
                                n_points: int = 50) -> pd.DataFrame:
    """Генерувати точки ефективної границі."""
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    n_assets = len(returns_df.columns)

    # Знаходимо мін та макс дохідність
    min_ret_result = minimize(
        lambda w: -np.sum(mean_returns * w) * 252,
        np.array([1.0 / n_assets] * n_assets),
        method='SLSQP',
        bounds=tuple((0, 1) for _ in range(n_assets)),
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    )

    max_ret_result = minimize(
        lambda w: np.sum(mean_returns * w) * 252,
        np.array([1.0 / n_assets] * n_assets),
        method='SLSQP',
        bounds=tuple((0, 1) for _ in range(n_assets)),
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    )

    min_return = -min_ret_result.fun
    max_return = -max_ret_result.fun

    target_returns = np.linspace(min_return, max_return, n_points)
    frontier_points = []

    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: np.sum(mean_returns * w) * 252 - t}
        ]

        result = minimize(
            portfolio_volatility,
            np.array([1.0 / n_assets] * n_assets),
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=tuple((0, 1) for _ in range(n_assets)),
            constraints=constraints
        )

        if result.success:
            vol = portfolio_volatility(result.x, mean_returns, cov_matrix)
            frontier_points.append({
                'return': target,
                'volatility': vol,
                'sharpe': (target - 0.02) / vol
            })

    return pd.DataFrame(frontier_points)


def print_optimization_results(results: dict, metrics_df: pd.DataFrame):
    """Вивести результати оптимізації."""
    print("\n" + "=" * 80)
    print("ОПТИМІЗАЦІЯ ПОРТФЕЛЯ")
    print("=" * 80)

    # Кореляційна матриця
    print("\n📊 КОРЕЛЯЦІЙНА МАТРИЦЯ:")
    print("-" * 80)
    corr = results['correlation_matrix']
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    print(corr.to_string())

    # Оптимальні ваги
    print("\n" + "=" * 80)
    print("🎯 ОПТИМАЛЬНИЙ ПОРТФЕЛЬ (Maximum Sharpe Ratio)")
    print("=" * 80)

    print("\nРозподіл активів:")
    print("-" * 50)

    # Сортуємо за вагою
    sorted_idx = np.argsort(results['weights'])[::-1]

    for idx in sorted_idx:
        ticker = results['tickers'][idx]
        weight = results['weights'][idx]
        if weight > 0.001:  # Показуємо тільки >0.1%
            # Знаходимо метрики тікера
            ticker_metrics = metrics_df[metrics_df['ticker'] == ticker]
            if not ticker_metrics.empty:
                ret = ticker_metrics['total_return_pct'].iloc[0]
                dd = ticker_metrics['max_drawdown_pct'].iloc[0]
                print(f"  {ticker:12} {weight*100:6.1f}%   (Return: {ret:6.1f}%, Max DD: {dd:6.1f}%)")
            else:
                print(f"  {ticker:12} {weight*100:6.1f}%")

    print("-" * 50)
    print(f"  {'TOTAL':12} {sum(results['weights'])*100:6.1f}%")

    # Статистика портфеля
    print("\n📈 ОЧІКУВАНІ ПОКАЗНИКИ ПОРТФЕЛЯ:")
    print("-" * 50)
    print(f"  Очікувана річна дохідність: {results['expected_return']*100:6.2f}%")
    print(f"  Річна волатильність:        {results['volatility']*100:6.2f}%")
    print(f"  Sharpe Ratio:               {results['sharpe_ratio']:6.2f}")
    if results['weighted_max_drawdown']:
        print(f"  Зважений Max Drawdown:      {results['weighted_max_drawdown']:6.2f}%")

    print("=" * 80)


def print_equal_weight_comparison(returns_df: pd.DataFrame,
                                   optimal_results: dict):
    """Порівняти з рівномірним розподілом."""
    n_assets = len(returns_df.columns)
    equal_weights = np.array([1.0 / n_assets] * n_assets)

    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    eq_return, eq_volatility = calculate_portfolio_stats(
        equal_weights, mean_returns, cov_matrix
    )
    eq_sharpe = (eq_return - 0.02) / eq_volatility

    print("\n📊 ПОРІВНЯННЯ З РІВНОМІРНИМ РОЗПОДІЛОМ:")
    print("-" * 60)
    print(f"{'Метрика':<25} {'Рівномірний':>15} {'Оптимальний':>15}")
    print("-" * 60)
    print(f"{'Очікувана дохідність':<25} {eq_return*100:>14.2f}% {optimal_results['expected_return']*100:>14.2f}%")
    print(f"{'Волатильність':<25} {eq_volatility*100:>14.2f}% {optimal_results['volatility']*100:>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {eq_sharpe:>15.2f} {optimal_results['sharpe_ratio']:>15.2f}")
    print("-" * 60)

    improvement = (optimal_results['sharpe_ratio'] / eq_sharpe - 1) * 100
    print(f"\n✨ Покращення Sharpe Ratio: {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Оптимізація портфеля на основі результатів симуляцій',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  python optimize_portfolio.py
  python optimize_portfolio.py --max-weight 0.3
  python optimize_portfolio.py --target min_var
        """
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='./simulation_results',
        help='Директорія з результатами симуляцій'
    )

    parser.add_argument(
        '--target', '-t',
        type=str,
        default='sharpe',
        choices=['sharpe', 'min_var'],
        help='Ціль оптимізації: sharpe (макс Sharpe) або min_var (мін волатильність)'
    )

    parser.add_argument(
        '--max-weight',
        type=float,
        default=0.40,
        help='Максимальна вага одного активу (за замовчуванням: 0.40)'
    )

    parser.add_argument(
        '--min-weight',
        type=float,
        default=0.0,
        help='Мінімальна вага одного активу (за замовчуванням: 0.0)'
    )

    args = parser.parse_args()

    results_dir = Path(args.dir)

    if not results_dir.exists():
        print(f"Директорія {results_dir} не існує")
        return

    print("Завантаження даних...")

    # Завантажуємо дані
    returns_df = load_returns_data(results_dir)
    metrics_df = load_metrics(results_dir)

    if returns_df.empty:
        print("Не знайдено даних для оптимізації")
        return

    print(f"Знайдено {len(returns_df.columns)} активів: {', '.join(returns_df.columns)}")
    print(f"Період: {len(returns_df)} торгових днів")

    # Оптимізація
    print(f"\nОптимізація портфеля (ціль: {'Maximum Sharpe' if args.target == 'sharpe' else 'Minimum Variance'})...")

    results = optimize_portfolio(
        returns_df,
        metrics_df,
        optimization_target=args.target,
        max_weight=args.max_weight,
        min_weight=args.min_weight
    )

    # Виводимо результати
    print_optimization_results(results, metrics_df)
    print_equal_weight_comparison(returns_df, results)

    # Зберігаємо результати
    output_file = results_dir / 'optimal_portfolio.csv'
    pd.DataFrame({
        'ticker': results['tickers'],
        'weight': results['weights'],
        'weight_pct': results['weights'] * 100
    }).to_csv(output_file, index=False)
    print(f"\nРезультати збережено в: {output_file}")


if __name__ == '__main__':
    main()
