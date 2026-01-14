# Investment Simulation

Інструменти для симуляції довгострокових інвестицій з використанням стратегії Dollar Cost Averaging (DCA) та оптимізації портфеля.

## Можливості

- **DCA симуляція** - регулярні покупки кожної другої п'ятниці з порівнянням T-bills
- **Реальна дохідність** - врахування інфляції через CPI
- **Метрики ризику** - Sharpe, Sortino, Calmar Ratio, Max Drawdown
- **Оптимізація портфеля** - Modern Portfolio Theory (Mean-Variance Optimization)
- **HTML звіт** - інтерактивний огляд результатів

## Швидкий старт

```bash
# Клонування та налаштування
git clone <repo-url>
cd stock-bond-inv-simulation

# Запуск всіх симуляцій (автоматично створить venv)
./run_simulations.sh
```

Результати будуть у `simulation_results/report.html`

## Встановлення

```bash
# Створення віртуального оточення
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Встановлення залежностей
pip install -r requirements.txt
```

## Використання

### Основний скрипт

```bash
./run_simulations.sh [опції]
```

| Опція | Опис | За замовчуванням |
|-------|------|------------------|
| `-s YEAR` | Початковий рік | 2020 |
| `-e YEAR` | Кінцевий рік | 2025 |
| `-a AMOUNT` | Сума інвестиції ($/2 тижні) | 500 |
| `-t "TICKERS"` | Список тікерів | Див. нижче |

**Приклади:**

```bash
# За замовчуванням
./run_simulations.sh

# Власний період та сума
./run_simulations.sh -s 2015 -e 2024 -a 1000

# Власний набір тікерів
./run_simulations.sh -t "AAPL MSFT GOOGL BTC-USD"
```

### Окремі скрипти

```bash
# Симуляція для одного тікера
python investment_simulation.py -t AAPL -s 2020 -e 2025 -a 500

# Агрегація метрик
python aggregate_results.py

# Оптимізація портфеля
python optimize_portfolio.py -s 2020 -e 2025 -a 500

# Генерація HTML звіту
python generate_report.py
```

## Тікери за замовчуванням

| Категорія | Тікери | Опис |
|-----------|--------|------|
| **Індекси** | GSPC, DJI, IXIC, QQQ | S&P 500, Dow Jones, Nasdaq, QQQ ETF |
| **Регіони** | EZU, EEM | Eurozone, Emerging Markets |
| **Облігації** | TLT | 20+ Year Treasury Bond |
| **Commodities** | GLD | Gold (SPDR) |
| **Нерухомість** | VNQ | Real Estate (REITs) |
| **Крипто** | BTC-USD, ETH-USD, SOL-USD | Bitcoin, Ethereum, Solana |

## Структура результатів

```
simulation_results/
├── GSPC_2020-2025_$500/
│   ├── 1_stock_price.png
│   ├── 2_portfolio_value.png
│   ├── 3_portfolio_profit.png
│   ├── 4_stocks_vs_tbills.png
│   ├── 5_real_profit_comparison.png
│   ├── 6_dashboard.png
│   ├── metrics.csv
│   └── simulation_data.csv
├── ...
├── comparison.csv           # Порівняння всіх активів
├── portfolio_results.csv    # Результати оптимізації
├── portfolio_comparison.png # Графік порівняння портфелів
└── report.html              # HTML звіт
```

## Метрики

| Метрика | Опис |
|---------|------|
| **Total Return** | Загальна дохідність (%) |
| **CAGR** | Compound Annual Growth Rate |
| **Sharpe Ratio** | Risk-adjusted return (vs risk-free rate) |
| **Sortino Ratio** | Sharpe з урахуванням лише негативної волатильності |
| **Max Drawdown** | Максимальна просадка від піку |
| **Calmar Ratio** | CAGR / Max Drawdown |
| **Risk-Reward Score** | Комбінована метрика (див. SPECIFICATION.md) |

## Тести

```bash
# Всі тести
pytest test_investment_simulation.py -v

# З покриттям
pytest test_investment_simulation.py --cov=investment_simulation
```

## Обмеження

- Не враховує комісії брокера
- Не враховує податки
- Не враховує дивіденди (для деяких індексів)
- Дані інфляції та T-bills тільки для США

## Ліцензія

MIT
