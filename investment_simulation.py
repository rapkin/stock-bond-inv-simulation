#!/usr/bin/env python3
"""
Investment Simulation Script
Симуляція інвестицій в акції з урахуванням інфляції та порівнянням з T-bills.

Покупки здійснюються кожної другої п'ятниці (день зарплати).
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr


def get_biweekly_fridays(start_date: datetime, end_date: datetime) -> list[datetime]:
    """Отримати список кожної другої п'ятниці в заданому діапазоні."""
    fridays = []
    current = start_date

    # Знаходимо першу п'ятницю
    while current.weekday() != 4:  # 4 = п'ятниця
        current += timedelta(days=1)

    # Додаємо кожну другу п'ятницю
    while current <= end_date:
        fridays.append(current)
        current += timedelta(days=14)  # +2 тижні

    return fridays


def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Завантажити історичні дані акцій."""
    print(f"Завантаження даних для {ticker}...")

    # Для індексів додаємо ^ якщо потрібно
    if ticker.upper() in ['GSPC', 'SPX', 'SP500']:
        ticker = '^GSPC'
    elif ticker.upper() == 'DJI':
        ticker = '^DJI'
    elif ticker.upper() == 'IXIC':
        ticker = '^IXIC'

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"Не вдалося завантажити дані для {ticker}")

    # Якщо є MultiIndex колонки, спрощуємо
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def download_inflation_data(start_date: str, end_date: str) -> pd.Series:
    """Завантажити дані CPI (Consumer Price Index) з FRED."""
    print("Завантаження даних інфляції (CPI)...")
    try:
        cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
        # Інтерполяція місячних даних до денних
        cpi = cpi.resample('D').interpolate(method='linear')
        return cpi['CPIAUCSL']
    except Exception as e:
        print(f"Помилка завантаження CPI: {e}")
        print("Використовую приблизну інфляцію 3% річних...")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Генеруємо синтетичний CPI з ~3% річної інфляції
        days = (dates - dates[0]).days
        cpi_values = 100 * (1.03 ** (days / 365))
        return pd.Series(cpi_values, index=dates)


def download_tbill_rates(start_date: str, end_date: str) -> pd.Series:
    """Завантажити ставки 3-місячних T-bills з FRED."""
    print("Завантаження ставок T-bills...")
    try:
        # TB3MS - 3-Month Treasury Bill: Secondary Market Rate
        tbill = pdr.DataReader('TB3MS', 'fred', start_date, end_date)
        # Інтерполяція до денних даних
        tbill = tbill.resample('D').interpolate(method='linear')
        return tbill['TB3MS'] / 100  # Конвертуємо % в десяткову форму
    except Exception as e:
        print(f"Помилка завантаження T-bill rates: {e}")
        print("Використовую приблизну ставку 2% річних...")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.Series(0.02, index=dates)


def simulate_investment(
    stock_data: pd.DataFrame,
    investment_dates: list[datetime],
    investment_amount: float,
    cpi_data: pd.Series,
    tbill_rates: pd.Series
) -> dict:
    """
    Симуляція інвестицій в акції.

    Returns:
        dict з результатами симуляції
    """
    results = {
        'dates': [],
        'stock_price': [],
        'shares_owned': [],
        'portfolio_value': [],
        'total_invested': [],
        'portfolio_profit': [],
        'portfolio_profit_pct': [],
        'tbill_value': [],
        'tbill_profit': [],
        # Значення скориговані на інфляцію (в "початкових доларах")
        # total_invested_real = купівельна спроможність готівки (якби не інвестували)
        'portfolio_value_real': [],
        'total_invested_real': [],  # "готівка реальна" - втрачає вартість через інфляцію
        'portfolio_profit_real': [],
        'tbill_value_real': [],
        'tbill_profit_real': [],
    }

    shares = 0.0
    total_invested = 0.0
    tbill_balance = 0.0

    # Базовий CPI для розрахунку реальних значень (на ПОЧАТОК періоду)
    # Це дозволяє виразити все в "початкових доларах" - тобто купівельній спроможності на момент старту
    base_cpi = cpi_data.iloc[0]

    # Отримуємо ціни закриття
    prices = stock_data['Close']

    for date in prices.index:
        date_key = date.date() if hasattr(date, 'date') else date

        # Перевіряємо чи це день покупки
        is_purchase_day = any(
            inv_date.date() == date_key if hasattr(date_key, '__eq__') else inv_date == date_key
            for inv_date in investment_dates
            if inv_date <= date
        )

        # Знаходимо найближчу дату покупки що вже минула
        purchase_dates_passed = [d for d in investment_dates if d <= date]

        # Покупка акцій
        for inv_date in investment_dates:
            inv_date_normalized = pd.Timestamp(inv_date)
            if inv_date_normalized == date:
                price = prices.loc[date]
                if not pd.isna(price) and price > 0:
                    new_shares = investment_amount / price
                    shares += new_shares
                    total_invested += investment_amount

                    # T-bill інвестиція
                    tbill_balance += investment_amount

        # Поточна вартість портфеля
        current_price = prices.loc[date]
        if pd.isna(current_price):
            continue

        portfolio_value = shares * current_price
        portfolio_profit = portfolio_value - total_invested
        portfolio_profit_pct = (portfolio_profit / total_invested * 100) if total_invested > 0 else 0

        # Розрахунок T-bill з накопиченими відсотками
        if date in tbill_rates.index:
            daily_rate = tbill_rates.loc[date] / 365
            tbill_balance *= (1 + daily_rate)

        tbill_profit = tbill_balance - total_invested

        # Корекція на інфляцію (використовуємо asof для пошуку найближчого CPI)
        try:
            cpi_current = cpi_data.asof(date)
            if pd.notna(cpi_current) and cpi_current > 0:
                inflation_factor = base_cpi / cpi_current
            else:
                inflation_factor = 1.0
        except Exception:
            inflation_factor = 1.0

        # Записуємо результати
        results['dates'].append(date)
        results['stock_price'].append(current_price)
        results['shares_owned'].append(shares)
        results['portfolio_value'].append(portfolio_value)
        results['total_invested'].append(total_invested)
        results['portfolio_profit'].append(portfolio_profit)
        results['portfolio_profit_pct'].append(portfolio_profit_pct)
        results['tbill_value'].append(tbill_balance)
        results['tbill_profit'].append(tbill_profit)

        # Реальні (скориговані на інфляцію) значення
        results['portfolio_value_real'].append(portfolio_value * inflation_factor)
        results['total_invested_real'].append(total_invested * inflation_factor)
        results['portfolio_profit_real'].append(portfolio_profit * inflation_factor)
        results['tbill_value_real'].append(tbill_balance * inflation_factor)
        results['tbill_profit_real'].append(tbill_profit * inflation_factor)

    return results


def create_visualizations(
    results: dict,
    ticker: str,
    investment_amount: float,
    start_year: int,
    end_year: int,
    output_dir: Path
):
    """Створити та зберегти візуалізації."""

    dates = results['dates']

    # Стиль графіків
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_size = (14, 8)

    # 1. Ціна акції
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(dates, results['stock_price'], 'b-', linewidth=1.5, label='Ціна акції')
    ax.set_title(f'Ціна {ticker} ({start_year}-{end_year})', fontsize=14)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Ціна ($)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / '1_stock_price.png', dpi=150)
    plt.close()

    # 2. Розмір портфеля (номінальний vs реальний vs готівка)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(dates, results['portfolio_value'], 'b-', linewidth=1.5, label='Портфель (номінальний)')
    ax.plot(dates, results['portfolio_value_real'], 'g-', linewidth=1.5, label='Портфель (реальний)')
    ax.plot(dates, results['total_invested'], 'r--', linewidth=1.5, label='Інвестовано (номінально)')
    ax.plot(dates, results['total_invested_real'], 'r-', linewidth=1.5, alpha=0.7, label='Готівка (реальна купівельна спроможність)')
    ax.set_title(f'Вартість портфеля {ticker} ({start_year}-{end_year})\n(все в "початкових доларах")', fontsize=14)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Вартість ($)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / '2_portfolio_value.png', dpi=150)
    plt.close()

    # 3. Прибутки портфеля
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(dates, results['portfolio_profit'], 'b-', linewidth=1.5, label='Прибуток (номінальний)')
    ax.plot(dates, results['portfolio_profit_real'], 'g-', linewidth=1.5, label='Прибуток (реальний)')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(dates, results['portfolio_profit'], 0,
                    where=[p > 0 for p in results['portfolio_profit']],
                    alpha=0.3, color='green', label='Прибуток')
    ax.fill_between(dates, results['portfolio_profit'], 0,
                    where=[p < 0 for p in results['portfolio_profit']],
                    alpha=0.3, color='red', label='Збиток')
    ax.set_title(f'Прибуток/Збиток портфеля {ticker} ({start_year}-{end_year})', fontsize=14)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Прибуток ($)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / '3_portfolio_profit.png', dpi=150)
    plt.close()

    # 4. Порівняння з T-bills
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(dates, results['portfolio_value'], 'b-', linewidth=1.5, label=f'Портфель {ticker}')
    ax.plot(dates, results['tbill_value'], 'orange', linewidth=1.5, label='T-bills')
    ax.plot(dates, results['total_invested'], 'r--', linewidth=1.5, label='Інвестовано')
    ax.set_title(f'Порівняння: {ticker} vs T-bills ({start_year}-{end_year})', fontsize=14)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Вартість ($)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / '4_stocks_vs_tbills.png', dpi=150)
    plt.close()

    # 5. Порівняння прибутків (реальних)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(dates, results['portfolio_profit_real'], 'b-', linewidth=1.5, label=f'Прибуток {ticker} (реальний)')
    ax.plot(dates, results['tbill_profit_real'], 'orange', linewidth=1.5, label='Прибуток T-bills (реальний)')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title(f'Реальні прибутки: {ticker} vs T-bills ({start_year}-{end_year})', fontsize=14)
    ax.set_xlabel('Дата')
    ax.set_ylabel('Реальний прибуток ($)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / '5_real_profit_comparison.png', dpi=150)
    plt.close()

    # 6. Зведений дашборд
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Ціна акції
    axes[0, 0].plot(dates, results['stock_price'], 'b-', linewidth=1)
    axes[0, 0].set_title(f'Ціна {ticker}')
    axes[0, 0].set_ylabel('Ціна ($)')

    # Вартість портфеля
    axes[0, 1].plot(dates, results['portfolio_value'], 'b-', linewidth=1, label='Номінальна')
    axes[0, 1].plot(dates, results['portfolio_value_real'], 'g-', linewidth=1, label='Реальна')
    axes[0, 1].plot(dates, results['total_invested'], 'r--', linewidth=1, label='Інвестовано')
    axes[0, 1].plot(dates, results['total_invested_real'], 'r-', linewidth=1, alpha=0.7, label='Готівка (реальна)')
    axes[0, 1].set_title('Вартість портфеля')
    axes[0, 1].set_ylabel('Вартість ($)')
    axes[0, 1].legend(fontsize=7)

    # Прибуток %
    axes[1, 0].plot(dates, results['portfolio_profit_pct'], 'b-', linewidth=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Прибуток портфеля (%)')
    axes[1, 0].set_ylabel('Прибуток (%)')

    # Порівняння з T-bills
    axes[1, 1].plot(dates, results['portfolio_profit_real'], 'b-', linewidth=1, label=f'{ticker}')
    axes[1, 1].plot(dates, results['tbill_profit_real'], 'orange', linewidth=1, label='T-bills')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Реальний прибуток: Акції vs T-bills')
    axes[1, 1].set_ylabel('Прибуток ($)')
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.suptitle(
        f'Симуляція інвестицій: {ticker} | ${investment_amount}/2 тижні | {start_year}-{end_year}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(output_dir / '6_dashboard.png', dpi=150)
    plt.close()

    print(f"Графіки збережено в: {output_dir}")


def print_summary(results: dict, ticker: str, investment_amount: float):
    """Вивести підсумок симуляції."""
    final_idx = -1

    total_invested = results['total_invested'][final_idx]
    total_invested_real = results['total_invested_real'][final_idx]  # Готівка (реальна)
    portfolio_value = results['portfolio_value'][final_idx]
    portfolio_profit = results['portfolio_profit'][final_idx]
    portfolio_profit_pct = results['portfolio_profit_pct'][final_idx]

    portfolio_value_real = results['portfolio_value_real'][final_idx]
    portfolio_profit_real = results['portfolio_profit_real'][final_idx]

    tbill_value = results['tbill_value'][final_idx]
    tbill_profit = results['tbill_profit'][final_idx]
    tbill_value_real = results['tbill_value_real'][final_idx]
    tbill_profit_real = results['tbill_profit_real'][final_idx]

    shares = results['shares_owned'][final_idx]

    # Втрати від інфляції якби тримали готівку
    cash_inflation_loss = total_invested - total_invested_real
    cash_inflation_loss_pct = (cash_inflation_loss / total_invested * 100) if total_invested > 0 else 0

    print("\n" + "=" * 65)
    print("ПІДСУМОК СИМУЛЯЦІЇ")
    print("=" * 65)
    print(f"Тікер: {ticker}")
    print(f"Сума інвестиції: ${investment_amount} кожні 2 тижні")
    print(f"Кількість покупок: {int(total_invested / investment_amount)}")
    print(f"Акцій у портфелі: {shares:.4f}")
    print("-" * 65)
    print("НОМІНАЛЬНІ ЗНАЧЕННЯ:")
    print(f"  Всього інвестовано: ${total_invested:,.2f}")
    print(f"  Вартість портфеля:  ${portfolio_value:,.2f}")
    print(f"  Прибуток:           ${portfolio_profit:,.2f} ({portfolio_profit_pct:.2f}%)")
    print("-" * 65)
    print("РЕАЛЬНІ ЗНАЧЕННЯ (в 'початкових доларах'):")
    print(f"  Вартість портфеля:  ${portfolio_value_real:,.2f}")
    print(f"  Реальний прибуток:  ${portfolio_profit_real:,.2f}")
    print("-" * 65)
    print("ПОРІВНЯННЯ З ГОТІВКОЮ (якби не інвестували):")
    print(f"  Готівка номінально: ${total_invested:,.2f}")
    print(f"  Готівка реально:    ${total_invested_real:,.2f} (купівельна спроможність)")
    print(f"  Втрати від інфляції: ${cash_inflation_loss:,.2f} ({cash_inflation_loss_pct:.1f}%)")
    print(f"  Перевага портфеля:  ${portfolio_value_real - total_invested_real:,.2f} (реальна)")
    print("-" * 65)
    print("ПОРІВНЯННЯ З T-BILLS:")
    print(f"  T-bills вартість:   ${tbill_value:,.2f} (номінальна)")
    print(f"  T-bills реально:    ${tbill_value_real:,.2f}")
    print(f"  Перевага акцій:     ${portfolio_value_real - tbill_value_real:,.2f} (реальна)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description='Симуляція інвестицій в акції з урахуванням інфляції',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  python investment_simulation.py --ticker GSPC --start 2010 --end 2024 --amount 500
  python investment_simulation.py --ticker AAPL --start 2015 --end 2023 --amount 1000
  python investment_simulation.py --ticker MSFT --start 2000 --end 2024 --amount 250
        """
    )

    parser.add_argument(
        '--ticker', '-t',
        type=str,
        default='GSPC',
        help='Тікер акції (наприклад: GSPC, AAPL, MSFT). За замовчуванням: GSPC (S&P 500)'
    )

    parser.add_argument(
        '--start', '-s',
        type=int,
        default=2010,
        help='Початковий рік симуляції. За замовчуванням: 2010'
    )

    parser.add_argument(
        '--end', '-e',
        type=int,
        default=2024,
        help='Кінцевий рік симуляції. За замовчуванням: 2024'
    )

    parser.add_argument(
        '--amount', '-a',
        type=float,
        default=500.0,
        help='Сума інвестиції кожні 2 тижні ($). За замовчуванням: 500'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./simulation_results',
        help='Базова директорія для результатів. За замовчуванням: ./simulation_results'
    )

    args = parser.parse_args()

    # Валідація параметрів
    if args.start >= args.end:
        print("Помилка: Початковий рік має бути меншим за кінцевий")
        return

    if args.amount <= 0:
        print("Помилка: Сума інвестиції має бути більше 0")
        return

    # Створюємо папку з параметрами в назві
    output_dir = Path(args.output) / f"{args.ticker}_{args.start}-{args.end}_${int(args.amount)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Дати для завантаження даних
    start_date = f"{args.start}-01-01"
    end_date = f"{args.end}-12-31"

    print(f"\nСимуляція інвестицій: {args.ticker}")
    print(f"Період: {args.start} - {args.end}")
    print(f"Інвестиція: ${args.amount} кожні 2 тижні")
    print("-" * 40)

    # Завантаження даних
    stock_data = download_stock_data(args.ticker, start_date, end_date)
    cpi_data = download_inflation_data(start_date, end_date)
    tbill_rates = download_tbill_rates(start_date, end_date)

    # Отримуємо дати покупок
    start_dt = datetime(args.start, 1, 1)
    end_dt = datetime(args.end, 12, 31)
    investment_dates = get_biweekly_fridays(start_dt, end_dt)

    # Фільтруємо дати, для яких є дані
    available_dates = set(stock_data.index.date)
    investment_dates = [
        d for d in investment_dates
        if d.date() in available_dates or any(
            (d - timedelta(days=i)).date() in available_dates
            for i in range(1, 4)
        )
    ]

    # Коригуємо дати на найближчий торговий день
    adjusted_dates = []
    for d in investment_dates:
        if d.date() in available_dates:
            adjusted_dates.append(pd.Timestamp(d))
        else:
            for i in range(1, 7):
                next_day = d + timedelta(days=i)
                if next_day.date() in available_dates:
                    adjusted_dates.append(pd.Timestamp(next_day))
                    break
                prev_day = d - timedelta(days=i)
                if prev_day.date() in available_dates:
                    adjusted_dates.append(pd.Timestamp(prev_day))
                    break

    print(f"Кількість покупок: {len(adjusted_dates)}")

    # Симуляція
    results = simulate_investment(
        stock_data=stock_data,
        investment_dates=adjusted_dates,
        investment_amount=args.amount,
        cpi_data=cpi_data,
        tbill_rates=tbill_rates
    )

    # Візуалізація
    create_visualizations(
        results=results,
        ticker=args.ticker,
        investment_amount=args.amount,
        start_year=args.start,
        end_year=args.end,
        output_dir=output_dir
    )

    # Підсумок
    print_summary(results, args.ticker, args.amount)

    # Зберігаємо дані в CSV
    df = pd.DataFrame({
        'date': results['dates'],
        'stock_price': results['stock_price'],
        'shares_owned': results['shares_owned'],
        'portfolio_value': results['portfolio_value'],
        'total_invested': results['total_invested'],
        'portfolio_profit': results['portfolio_profit'],
        'portfolio_profit_pct': results['portfolio_profit_pct'],
        'portfolio_value_real': results['portfolio_value_real'],
        'portfolio_profit_real': results['portfolio_profit_real'],
        'tbill_value': results['tbill_value'],
        'tbill_profit': results['tbill_profit'],
        'tbill_value_real': results['tbill_value_real'],
        'tbill_profit_real': results['tbill_profit_real'],
    })
    df.to_csv(output_dir / 'simulation_data.csv', index=False)
    print(f"\nДані збережено в: {output_dir / 'simulation_data.csv'}")


if __name__ == '__main__':
    main()
