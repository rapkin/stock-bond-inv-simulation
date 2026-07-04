#!/usr/bin/env python3
"""
Portfolio Optimization Script
Оптимізація портфеля на основі результатів симуляцій.

Використовує Modern Portfolio Theory (MPT) для знаходження
оптимальних ваг активів з урахуванням кореляцій.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from investment_simulation import get_biweekly_fridays


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


def load_price_data(results_dir: Path) -> pd.DataFrame:
    """
    Завантажити ціни активів з усіх симуляцій.

    Returns:
        DataFrame з цінами для кожного тікера
    """
    prices_dict = {}

    for sim_dir in results_dir.iterdir():
        if not sim_dir.is_dir():
            continue

        data_file = sim_dir / 'simulation_data.csv'
        metrics_file = sim_dir / 'metrics.csv'

        if not data_file.exists() or not metrics_file.exists():
            continue

        try:
            metrics = pd.read_csv(metrics_file)
            ticker = metrics['ticker'].iloc[0]

            data = pd.read_csv(data_file, parse_dates=['date'])
            data = data.set_index('date')

            if 'stock_price' in data.columns:
                prices = data['stock_price']
                prices = prices[prices > 0].dropna()
                if len(prices) > 100:
                    prices_dict[ticker] = prices

        except Exception as e:
            print(f"Помилка завантаження цін {sim_dir.name}: {e}")

    if not prices_dict:
        return pd.DataFrame()

    prices_df = pd.DataFrame(prices_dict)
    prices_df = prices_df.dropna()

    return prices_df


def calculate_optimal_weights_from_prices(
    prices_df: pd.DataFrame,
    max_weight: float = 0.4,
    min_weight: float = 0.0,
    previous_weights: np.ndarray = None,
    max_weight_change: float = 0.10
) -> tuple:
    """
    Розрахувати оптимальні ваги на основі історичних цін.

    Args:
        prices_df: DataFrame з цінами
        max_weight: Максимальна вага активу
        min_weight: Мінімальна вага активу
        previous_weights: Попередні ваги (для обмеження зміни)
        max_weight_change: Максимальна зміна ваги за один ребаланс (0.10 = 10%)

    Returns:
        tuple: (weights, tickers)
    """
    # Розраховуємо денні дохідності
    returns_df = prices_df.pct_change().dropna()
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna()
    returns_df = returns_df[(returns_df.abs() < 1).all(axis=1)]

    if len(returns_df) < 60:  # Мінімум 60 днів
        # Якщо недостатньо даних - рівномірний розподіл
        n = len(prices_df.columns)
        return np.array([1.0 / n] * n), prices_df.columns.tolist()

    tickers = returns_df.columns.tolist()
    n_assets = len(tickers)

    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    # Початкові ваги
    if previous_weights is not None and len(previous_weights) == n_assets:
        initial_weights = previous_weights.copy()
    else:
        initial_weights = np.array([1.0 / n_assets] * n_assets)

    # Обмеження
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Межі для кожного активу з урахуванням обмеження зміни
    if previous_weights is not None and len(previous_weights) == n_assets:
        bounds = tuple(
            (max(min_weight, prev - max_weight_change),
             min(max_weight, prev + max_weight_change))
            for prev in previous_weights
        )
    else:
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

    # Оптимізація за Sharpe
    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        return result.x, tickers
    else:
        return initial_weights, tickers


def simulate_dynamic_portfolio(
    prices_df: pd.DataFrame,
    investment_amount: float = 500.0,
    lookback_years: int = 3,
    rebalance_months: int = 3,
    max_weight: float = 0.4,
    max_weight_change: float = 0.10
) -> dict:
    """
    Симуляція портфеля з динамічним ребалансуванням.

    Args:
        prices_df: DataFrame з цінами активів
        investment_amount: Сума інвестиції кожні 2 тижні
        lookback_years: Кількість років для аналізу перед початком інвестування
        rebalance_months: Частота ребалансування (в місяцях), за замовчуванням 3 (квартал)
        max_weight: Максимальна вага одного активу
        max_weight_change: Максимальна зміна ваги за один ребаланс (0.10 = 10%)

    Returns:
        dict з результатами симуляції
    """
    lookback_days = lookback_years * 252  # Торгових днів

    if len(prices_df) < lookback_days + 60:
        print(f"Недостатньо даних. Потрібно мінімум {lookback_days + 60} днів, є {len(prices_df)}")
        return None

    # Дати для симуляції (після lookback періоду)
    all_dates = prices_df.index.tolist()
    investment_start_idx = lookback_days
    investment_start_date = all_dates[investment_start_idx]

    print(f"\n📅 Lookback період: {all_dates[0].strftime('%Y-%m-%d')} - {all_dates[investment_start_idx-1].strftime('%Y-%m-%d')}")
    print(f"📅 Період інвестування: {investment_start_date.strftime('%Y-%m-%d')} - {all_dates[-1].strftime('%Y-%m-%d')}")

    # Генеруємо дати інвестування (кожні 2 тижні)
    start_dt = investment_start_date.to_pydatetime() if hasattr(investment_start_date, 'to_pydatetime') else investment_start_date
    end_dt = all_dates[-1].to_pydatetime() if hasattr(all_dates[-1], 'to_pydatetime') else all_dates[-1]
    investment_dates = get_biweekly_fridays(start_dt, end_dt)

    # Ініціалізація
    tickers = prices_df.columns.tolist()
    holdings = {t: 0.0 for t in tickers}
    total_invested = 0.0
    current_weights = None
    last_rebalance_month = None

    results = {
        'dates': [],
        'portfolio_value': [],
        'total_invested': [],
        'profit': [],
        'profit_pct': [],
        'weights_history': [],
        'rebalance_dates': [],
    }

    # Симуляція
    for i, date in enumerate(all_dates):
        if i < investment_start_idx:
            continue  # Пропускаємо lookback період

        date_dt = date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date
        current_month = (date_dt.year, date_dt.month)

        # Перевіряємо чи потрібно ребалансувати
        need_rebalance = False
        if current_weights is None:
            need_rebalance = True
        elif last_rebalance_month is None:
            need_rebalance = True
        else:
            months_diff = (current_month[0] - last_rebalance_month[0]) * 12 + (current_month[1] - last_rebalance_month[1])
            if months_diff >= rebalance_months:
                need_rebalance = True

        # Ребалансування
        if need_rebalance:
            # Використовуємо дані за останні lookback_years
            lookback_start = max(0, i - lookback_days)
            lookback_prices = prices_df.iloc[lookback_start:i]

            if len(lookback_prices) >= 60:
                new_weights, _ = calculate_optimal_weights_from_prices(
                    lookback_prices,
                    max_weight=max_weight,
                    previous_weights=current_weights,
                    max_weight_change=max_weight_change
                )
                current_weights = new_weights
                last_rebalance_month = current_month
                results['rebalance_dates'].append(date)

                # Логування
                if len(results['rebalance_dates']) <= 3 or len(results['rebalance_dates']) % 4 == 0:
                    top_weights = sorted(zip(tickers, current_weights), key=lambda x: -x[1])[:3]
                    top_str = ", ".join([f"{t}: {w*100:.0f}%" for t, w in top_weights])
                    print(f"  📊 Ребаланс {date_dt.strftime('%Y-%m')}: {top_str}")

        # Інвестування (якщо це день інвестування)
        for inv_date in investment_dates:
            if abs((inv_date - date_dt).days) == 0 and current_weights is not None:
                for j, ticker in enumerate(tickers):
                    price = prices_df.loc[date, ticker]
                    if pd.notna(price) and price > 0:
                        amount = investment_amount * current_weights[j]
                        shares = amount / price
                        holdings[ticker] += shares
                total_invested += investment_amount
                break

        # Розрахунок вартості портфеля
        portfolio_value = 0.0
        for ticker in tickers:
            price = prices_df.loc[date, ticker]
            if pd.notna(price) and price > 0:
                portfolio_value += holdings[ticker] * price

        profit = portfolio_value - total_invested
        profit_pct = (profit / total_invested * 100) if total_invested > 0 else 0

        results['dates'].append(date)
        results['portfolio_value'].append(portfolio_value)
        results['total_invested'].append(total_invested)
        results['profit'].append(profit)
        results['profit_pct'].append(profit_pct)
        if current_weights is not None:
            results['weights_history'].append(dict(zip(tickers, current_weights)))

    print(f"\n  ✅ Всього ребалансувань: {len(results['rebalance_dates'])}")

    return results


def simulate_portfolio_investment(
    prices_df: pd.DataFrame,
    weights: np.ndarray,
    tickers: list,
    investment_amount: float = 500.0
) -> dict:
    """
    Симулювати інвестування в портфель кожні 2 тижні.

    Args:
        prices_df: DataFrame з цінами активів
        weights: Ваги активів в портфелі
        tickers: Список тікерів
        investment_amount: Сума інвестиції кожні 2 тижні

    Returns:
        dict з результатами симуляції
    """
    # Фільтруємо тільки тікери що є в prices_df
    available_tickers = [t for t in tickers if t in prices_df.columns]
    available_weights = []
    for t in available_tickers:
        idx = tickers.index(t)
        available_weights.append(weights[idx])

    # Нормалізуємо ваги
    available_weights = np.array(available_weights)
    available_weights = available_weights / available_weights.sum()

    # Дати інвестування
    start_date = prices_df.index.min()
    end_date = prices_df.index.max()

    if hasattr(start_date, 'to_pydatetime'):
        start_date = start_date.to_pydatetime()
    if hasattr(end_date, 'to_pydatetime'):
        end_date = end_date.to_pydatetime()

    investment_dates = get_biweekly_fridays(start_date, end_date)

    # Ініціалізація
    holdings = {t: 0.0 for t in available_tickers}
    total_invested = 0.0

    results = {
        'dates': [],
        'portfolio_value': [],
        'total_invested': [],
        'profit': [],
        'profit_pct': [],
    }

    # Симуляція
    for date in prices_df.index:
        date_dt = date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date

        # Перевіряємо чи це день інвестування
        is_investment_day = any(
            abs((inv_date - date_dt).days) <= 3
            for inv_date in investment_dates
            if inv_date <= date_dt
        )

        # Інвестуємо
        for inv_date in investment_dates:
            if abs((inv_date - date_dt).days) == 0:
                for i, ticker in enumerate(available_tickers):
                    if ticker in prices_df.columns and date in prices_df.index:
                        price = prices_df.loc[date, ticker]
                        if pd.notna(price) and price > 0:
                            amount = investment_amount * available_weights[i]
                            shares = amount / price
                            holdings[ticker] += shares
                total_invested += investment_amount
                break

        # Розраховуємо вартість портфеля
        portfolio_value = 0.0
        for ticker in available_tickers:
            if ticker in prices_df.columns and date in prices_df.index:
                price = prices_df.loc[date, ticker]
                if pd.notna(price) and price > 0:
                    portfolio_value += holdings[ticker] * price

        profit = portfolio_value - total_invested
        profit_pct = (profit / total_invested * 100) if total_invested > 0 else 0

        results['dates'].append(date)
        results['portfolio_value'].append(portfolio_value)
        results['total_invested'].append(total_invested)
        results['profit'].append(profit)
        results['profit_pct'].append(profit_pct)

    return results


def create_portfolio_visualization(
    optimal_results: dict,
    equal_results: dict,
    output_dir: Path
):
    """Створити візуалізацію порівняння портфелів."""

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    dates = optimal_results['dates']

    # 1. Вартість портфелів
    ax = axes[0, 0]
    ax.plot(dates, optimal_results['portfolio_value'], 'b-', linewidth=2, label='Оптимальний')
    ax.plot(dates, equal_results['portfolio_value'], 'orange', linewidth=2, label='Рівномірний')
    ax.plot(dates, optimal_results['total_invested'], 'r--', linewidth=1.5, label='Інвестовано')
    ax.set_title('Вартість портфелів', fontsize=14)
    ax.set_ylabel('Вартість ($)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 2. Прибуток в $
    ax = axes[0, 1]
    ax.plot(dates, optimal_results['profit'], 'b-', linewidth=2, label='Оптимальний')
    ax.plot(dates, equal_results['profit'], 'orange', linewidth=2, label='Рівномірний')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(dates, optimal_results['profit'], 0,
                    where=[p > 0 for p in optimal_results['profit']],
                    alpha=0.3, color='blue')
    ax.set_title('Прибуток ($)', fontsize=14)
    ax.set_ylabel('Прибуток ($)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 3. Прибуток в %
    ax = axes[1, 0]
    ax.plot(dates, optimal_results['profit_pct'], 'b-', linewidth=2, label='Оптимальний')
    ax.plot(dates, equal_results['profit_pct'], 'orange', linewidth=2, label='Рівномірний')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Прибуток (%)', fontsize=14)
    ax.set_ylabel('Прибуток (%)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 4. Різниця між портфелями
    ax = axes[1, 1]
    diff = [o - e for o, e in zip(optimal_results['profit'], equal_results['profit'])]
    ax.fill_between(dates, diff, 0,
                    where=[d > 0 for d in diff],
                    alpha=0.5, color='green', label='Оптимальний краще')
    ax.fill_between(dates, diff, 0,
                    where=[d < 0 for d in diff],
                    alpha=0.5, color='red', label='Рівномірний краще')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_title('Перевага оптимального портфеля ($)', fontsize=14)
    ax.set_ylabel('Різниця ($)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.suptitle('Порівняння: Оптимальний vs Рівномірний портфель\n(інвестиція $500 кожні 2 тижні)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'portfolio_comparison.png'
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"\n📈 Графік збережено в: {output_file}")


def print_simulation_summary(optimal_results: dict, equal_results: dict):
    """Вивести підсумок симуляції."""
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТИ СИМУЛЯЦІЇ (інвестиція $500 кожні 2 тижні)")
    print("=" * 80)

    opt_final = optimal_results['portfolio_value'][-1]
    opt_invested = optimal_results['total_invested'][-1]
    opt_profit = optimal_results['profit'][-1]
    opt_profit_pct = optimal_results['profit_pct'][-1]

    eq_final = equal_results['portfolio_value'][-1]
    eq_invested = equal_results['total_invested'][-1]
    eq_profit = equal_results['profit'][-1]
    eq_profit_pct = equal_results['profit_pct'][-1]

    print(f"\n{'Метрика':<25} {'Оптимальний':>15} {'Рівномірний':>15}")
    print("-" * 60)
    print(f"{'Інвестовано':<25} ${opt_invested:>13,.0f} ${eq_invested:>13,.0f}")
    print(f"{'Фінальна вартість':<25} ${opt_final:>13,.0f} ${eq_final:>13,.0f}")
    print(f"{'Прибуток':<25} ${opt_profit:>13,.0f} ${eq_profit:>13,.0f}")
    print(f"{'Прибуток %':<25} {opt_profit_pct:>14.1f}% {eq_profit_pct:>14.1f}%")
    print("-" * 60)

    advantage = opt_final - eq_final
    print(f"\n💰 Перевага оптимального портфеля: ${advantage:,.0f}")
    print("=" * 80)


def print_simulation_summary_three(dynamic_results: dict, optimal_results: dict, equal_results: dict):
    """Вивести підсумок симуляції трьох портфелів."""
    print("\n" + "=" * 90)
    print("📊 РЕЗУЛЬТАТИ СИМУЛЯЦІЇ")
    print("=" * 90)

    dyn_final = dynamic_results['portfolio_value'][-1]
    dyn_invested = dynamic_results['total_invested'][-1]
    dyn_profit = dynamic_results['profit'][-1]
    dyn_profit_pct = dynamic_results['profit_pct'][-1]

    opt_final = optimal_results['portfolio_value'][-1]
    opt_invested = optimal_results['total_invested'][-1]
    opt_profit = optimal_results['profit'][-1]
    opt_profit_pct = optimal_results['profit_pct'][-1]

    eq_final = equal_results['portfolio_value'][-1]
    eq_invested = equal_results['total_invested'][-1]
    eq_profit = equal_results['profit'][-1]
    eq_profit_pct = equal_results['profit_pct'][-1]

    print(f"\n{'Метрика':<25} {'Динамічний':>15} {'Статичний':>15} {'Рівномірний':>15}")
    print("-" * 75)
    print(f"{'Інвестовано':<25} ${dyn_invested:>13,.0f} ${opt_invested:>13,.0f} ${eq_invested:>13,.0f}")
    print(f"{'Фінальна вартість':<25} ${dyn_final:>13,.0f} ${opt_final:>13,.0f} ${eq_final:>13,.0f}")
    print(f"{'Прибуток':<25} ${dyn_profit:>13,.0f} ${opt_profit:>13,.0f} ${eq_profit:>13,.0f}")
    print(f"{'Прибуток %':<25} {dyn_profit_pct:>14.1f}% {opt_profit_pct:>14.1f}% {eq_profit_pct:>14.1f}%")
    print("-" * 75)

    # Визначаємо найкращий портфель
    best_portfolio = max([
        ('Динамічний', dyn_final),
        ('Статичний', opt_final),
        ('Рівномірний', eq_final)
    ], key=lambda x: x[1])

    print(f"\n🏆 Найкращий портфель: {best_portfolio[0]} (${best_portfolio[1]:,.0f})")
    print(f"   Перевага динамічного над статичним: ${dyn_final - opt_final:+,.0f}")
    print(f"   Перевага динамічного над рівномірним: ${dyn_final - eq_final:+,.0f}")
    print("=" * 90)


def create_portfolio_visualization_three(
    dynamic_results: dict,
    optimal_results: dict,
    equal_results: dict,
    output_dir: Path,
    lookback_years: int = 3,
    rebalance_months: int = 3,
    investment_amount: float = 500.0
):
    """Створити візуалізацію порівняння трьох портфелів."""

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    dates = dynamic_results['dates']

    # 1. Вартість портфелів
    ax = axes[0, 0]
    ax.plot(dates, dynamic_results['portfolio_value'], 'g-', linewidth=2, label='Динамічний')
    ax.plot(dates, optimal_results['portfolio_value'], 'b-', linewidth=2, label='Статичний')
    ax.plot(dates, equal_results['portfolio_value'], 'orange', linewidth=2, label='Рівномірний')
    ax.plot(dates, dynamic_results['total_invested'], 'r--', linewidth=1.5, label='Інвестовано')
    ax.set_title('Вартість портфелів', fontsize=14)
    ax.set_ylabel('Вартість ($)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 2. Прибуток в $
    ax = axes[0, 1]
    ax.plot(dates, dynamic_results['profit'], 'g-', linewidth=2, label='Динамічний')
    ax.plot(dates, optimal_results['profit'], 'b-', linewidth=2, label='Статичний')
    ax.plot(dates, equal_results['profit'], 'orange', linewidth=2, label='Рівномірний')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Прибуток ($)', fontsize=14)
    ax.set_ylabel('Прибуток ($)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 3. Прибуток в %
    ax = axes[1, 0]
    ax.plot(dates, dynamic_results['profit_pct'], 'g-', linewidth=2, label='Динамічний')
    ax.plot(dates, optimal_results['profit_pct'], 'b-', linewidth=2, label='Статичний')
    ax.plot(dates, equal_results['profit_pct'], 'orange', linewidth=2, label='Рівномірний')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Прибуток (%)', fontsize=14)
    ax.set_ylabel('Прибуток (%)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 4. Перевага динамічного портфеля
    ax = axes[1, 1]
    diff_vs_static = [d - o for d, o in zip(dynamic_results['profit'], optimal_results['profit'])]
    diff_vs_equal = [d - e for d, e in zip(dynamic_results['profit'], equal_results['profit'])]
    ax.plot(dates, diff_vs_static, 'b-', linewidth=2, label='vs Статичний')
    ax.plot(dates, diff_vs_equal, 'orange', linewidth=2, label='vs Рівномірний')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.fill_between(dates, diff_vs_static, 0,
                    where=[d > 0 for d in diff_vs_static],
                    alpha=0.3, color='green')
    ax.fill_between(dates, diff_vs_static, 0,
                    where=[d < 0 for d in diff_vs_static],
                    alpha=0.3, color='red')
    ax.set_title('Перевага динамічного портфеля ($)', fontsize=14)
    ax.set_ylabel('Різниця ($)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.suptitle(f'Порівняння: Динамічний vs Статичний vs Рівномірний портфель\n'
                 f'(lookback {lookback_years} років, ребаланс кожні {rebalance_months} міс., '
                 f'інвестиція ${investment_amount:.0f}/2 тижні)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'portfolio_comparison.png'
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"\n📈 Графік збережено в: {output_file}")


def create_weights_visualization(dynamic_results: dict, output_dir: Path):
    """Створити візуалізацію розподілу ваг портфеля в часі."""

    if 'weights_history' not in dynamic_results or not dynamic_results['weights_history']:
        print("Немає даних про ваги для візуалізації")
        return

    # Отримуємо дати ребалансування та відповідні ваги
    weights_history = dynamic_results['weights_history']
    rebalance_dates = dynamic_results.get('rebalance_dates', [])

    if not weights_history:
        return

    # Отримуємо список тікерів
    tickers = list(weights_history[0].keys())

    # Створюємо DataFrame з вагами на кожну дату
    # Беремо ваги тільки на дати ребалансування
    weights_data = []
    for i, date in enumerate(dynamic_results['dates']):
        if i < len(weights_history):
            row = {'date': date}
            row.update(weights_history[i])
            weights_data.append(row)

    weights_df = pd.DataFrame(weights_data)
    weights_df = weights_df.set_index('date')

    # Семплюємо щомісяця для кращої візуалізації
    weights_df = weights_df.resample('ME').last().dropna()

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # 1. Stacked Area Chart - розподіл ваг в часі
    ax = axes[0]

    # Сортуємо тікери за середньою вагою
    avg_weights = weights_df.mean().sort_values(ascending=False)
    sorted_tickers = avg_weights.index.tolist()

    # Створюємо stacked area
    colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_tickers)))
    ax.stackplot(weights_df.index, [weights_df[t] * 100 for t in sorted_tickers],
                 labels=sorted_tickers, colors=colors, alpha=0.8)

    ax.set_title('Розподіл портфеля в часі (динамічний)', fontsize=14)
    ax.set_ylabel('Вага (%)')
    ax.set_ylim(0, 100)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 2. Bar Chart - останній розподіл
    ax = axes[1]

    last_weights = weights_df.iloc[-1].sort_values(ascending=True)
    last_weights_pct = last_weights * 100

    # Фільтруємо тільки активи з вагою > 0.5%
    last_weights_pct = last_weights_pct[last_weights_pct > 0.5]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(last_weights_pct)))
    bars = ax.barh(last_weights_pct.index, last_weights_pct.values, color=colors)

    # Додаємо підписи на стовпці
    for bar, val in zip(bars, last_weights_pct.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)

    ax.set_title(f'Поточний розподіл портфеля ({weights_df.index[-1].strftime("%Y-%m")})', fontsize=14)
    ax.set_xlabel('Вага (%)')
    ax.set_xlim(0, max(last_weights_pct.values) * 1.15)

    plt.suptitle('Динамічний портфель - розподіл активів', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'portfolio_weights.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Графік ваг збережено в: {output_file}")


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

    parser.add_argument(
        '--lookback',
        type=int,
        default=3,
        help='Lookback період в роках для динамічного портфеля (за замовчуванням: 3)'
    )

    parser.add_argument(
        '--rebalance',
        type=int,
        default=3,
        help='Частота ребалансування в місяцях (за замовчуванням: 3 = квартал)'
    )

    parser.add_argument(
        '--max-change',
        type=float,
        default=0.10,
        help='Максимальна зміна ваги за ребаланс (за замовчуванням: 0.10 = 10%%)'
    )

    parser.add_argument(
        '--amount',
        type=float,
        default=500.0,
        help='Сума інвестиції кожні 2 тижні (за замовчуванням: 500)'
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

    # Симуляція інвестування
    print("\n" + "=" * 80)
    print("Симуляція інвестування...")
    print("=" * 80)

    prices_df = load_price_data(results_dir)

    if not prices_df.empty:
        # 1. Динамічний портфель (з ребалансуванням)
        print(f"\n🔄 ДИНАМІЧНИЙ ПОРТФЕЛЬ (lookback {args.lookback} років, ребаланс кожні {args.rebalance} міс., макс зміна {args.max_change*100:.0f}%):")
        dynamic_sim = simulate_dynamic_portfolio(
            prices_df,
            investment_amount=args.amount,
            lookback_years=args.lookback,
            rebalance_months=args.rebalance,
            max_weight=args.max_weight,
            max_weight_change=args.max_change
        )

        # 2. Статичний оптимальний портфель (ваги з повного періоду)
        print("\n📊 СТАТИЧНИЙ ОПТИМАЛЬНИЙ ПОРТФЕЛЬ:")
        if dynamic_sim:
            # Симулюємо з тієї ж дати що й динамічний
            start_idx = prices_df.index.get_loc(dynamic_sim['dates'][0])
            static_prices = prices_df.iloc[start_idx:]

            optimal_sim = simulate_portfolio_investment(
                static_prices,
                results['weights'],
                results['tickers'],
                investment_amount=args.amount
            )

            # 3. Рівномірний портфель
            print("\n⚖️ РІВНОМІРНИЙ ПОРТФЕЛЬ:")
            n_assets = len(results['tickers'])
            equal_weights = np.array([1.0 / n_assets] * n_assets)
            equal_sim = simulate_portfolio_investment(
                static_prices,
                equal_weights,
                results['tickers'],
                investment_amount=args.amount
            )

            # Підсумок всіх трьох портфелів
            print_simulation_summary_three(dynamic_sim, optimal_sim, equal_sim)

            # Візуалізація
            create_portfolio_visualization_three(
                dynamic_sim, optimal_sim, equal_sim, results_dir,
                lookback_years=args.lookback,
                rebalance_months=args.rebalance,
                investment_amount=args.amount
            )

            # Візуалізація ваг портфеля
            create_weights_visualization(dynamic_sim, results_dir)

            # Зберігаємо дані симуляції
            sim_df = pd.DataFrame({
                'date': dynamic_sim['dates'],
                'dynamic_value': dynamic_sim['portfolio_value'],
                'dynamic_profit': dynamic_sim['profit'],
                'optimal_value': optimal_sim['portfolio_value'],
                'optimal_profit': optimal_sim['profit'],
                'equal_value': equal_sim['portfolio_value'],
                'equal_profit': equal_sim['profit'],
                'total_invested': dynamic_sim['total_invested'],
            })
            sim_file = results_dir / 'portfolio_simulation.csv'
            sim_df.to_csv(sim_file, index=False)
            print(f"\n📊 Дані симуляції збережено в: {sim_file}")
        else:
            print("Недостатньо даних для динамічної симуляції")
    else:
        print("Не вдалося завантажити дані цін для симуляції")


if __name__ == '__main__':
    main()
