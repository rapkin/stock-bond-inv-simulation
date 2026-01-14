# Investment Simulation - Technical Specification

Технічна специфікація для автоматичної генерації проекту з використанням AI.

## Мета проекту

### Проблема

Звичайна людина з регулярним доходом хоче інвестувати, але стикається з питаннями:
- **Куди інвестувати?** Акції, індекси, облігації, золото, крипта - що обрати?
- **Чи це вигідно?** Як порівняти з простим триманням грошей або депозитом?
- **Який реальний прибуток?** Номінальні цифри оманливі через інфляцію
- **Які ризики?** Скільки можна втратити в найгіршому випадку?
- **Як збалансувати портфель?** Яку частку виділити на кожен актив?

### Рішення

Цей проект симулює реалістичний сценарій інвестування:
- Фіксована сума кожні 2 тижні (день зарплати) - стратегія Dollar Cost Averaging
- Порівняння з безризиковими T-bills та готівкою (з урахуванням інфляції)
- Розрахунок реальної дохідності (в купівельній спроможності)
- Метрики ризику для об'єктивної оцінки
- Оптимізація портфеля для балансу прибутку та ризику

### Ключові принципи

1. **Реалістичність** - симуляція максимально наближена до реального інвестування звичайної людини
2. **Прозорість** - всі формули та припущення документовані
3. **Порівнюваність** - однакова методологія для всіх активів
4. **Врахування інфляції** - номінальний прибуток нічого не означає без реальної купівельної спроможності
5. **Баланс ризику/прибутку** - високий прибуток без врахування ризику - це азарт, не інвестування

### Що проект НЕ робить

- Не дає фінансових порад (тільки інструмент аналізу)
- Не враховує податки та комісії (залежать від юрисдикції)
- Не прогнозує майбутнє (аналізує тільки історичні дані)
- Не враховує дивіденди для деяких індексів

## Основні сценарії використання

1. **Порівняння активів** - який актив показав найкращі результати за період?
2. **Аналіз ризиків** - які були максимальні просадки? Скільки часу тривало відновлення?
3. **Реальна дохідність** - скільки заробив інвестор в реальних грошах після інфляції?
4. **Оптимізація портфеля** - як розподілити інвестиції для балансу ризику та прибутку?
5. **Порівняння з альтернативами** - чи краще було просто покласти гроші на T-bills?

## Архітектура

```
┌─────────────────────────────────────────────────────────────────────┐
│                        run_simulations.sh                           │
│                    (Orchestration Script)                           │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ investment_   │   │  aggregate_     │   │   optimize_     │
│ simulation.py │   │  results.py     │   │   portfolio.py  │
│               │   │                 │   │                 │
│ - DCA logic   │   │ - Metrics       │   │ - MPT/MVO       │
│ - T-bills     │   │   aggregation   │   │ - Portfolio     │
│ - Inflation   │   │ - Risk-Reward   │   │   simulation    │
│ - Metrics     │   │   Score         │   │                 │
└───────┬───────┘   └────────┬────────┘   └────────┬────────┘
        │                    │                     │
        └────────────────────┼─────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ generate_       │
                    │ report.py       │
                    │                 │
                    │ - HTML report   │
                    │ - Charts embed  │
                    └─────────────────┘
```

## Модулі

### 1. investment_simulation.py

**Призначення:** DCA симуляція для одного тікера

#### Функції

| Функція | Опис |
|---------|------|
| `get_biweekly_fridays(start, end)` | Генерує список других п'ятниць кожного місяця |
| `download_stock_data(ticker, start, end)` | Завантажує ціни з Yahoo Finance (з кешуванням) |
| `download_inflation_data(start, end)` | Завантажує CPI з FRED API (з кешуванням) |
| `download_tbill_rates(start, end)` | Завантажує ставки T-bills з FRED (з кешуванням) |
| `simulate_investment(...)` | Основна симуляція DCA |
| `calculate_risk_metrics(df)` | Розрахунок Sharpe, Sortino, Max DD, etc. |

#### Логіка кешування

```python
cache_dir = .cache/{year}/  # Річна структура
cache_key = f"{ticker}_{year}.pkl" | f"cpi_{year}.pkl" | f"tbill_{year}.pkl"

# При завантаженні - перевіряємо кеш по роках
# При збереженні - зберігаємо окремо для кожного року
```

#### Фільтрація даних

```python
# Видаляємо нульові та невалідні значення
data = data[data['Close'] > 0].dropna(subset=['Close'])
```

### 2. aggregate_results.py

**Призначення:** Агрегація метрик з усіх симуляцій

#### Risk-Reward Score Formula

```python
Score = Sortino × sqrt(1 - |MaxDD|/100) × sqrt(Return/100) × suspicion_penalty

де:
  - Sortino: Sortino Ratio (дохідність / downside deviation)
  - MaxDD: Maximum Drawdown у відсотках
  - Return: Total Return у відсотках
  - suspicion_penalty: штраф для аномальних результатів
    - 0.7 якщо Sortino > 7
    - 0.7 якщо MaxDD < 15% при Return > 50%
```

**Логіка:**
- `sqrt` для MaxDD: м'якший штраф за просадки (0% DD = 1.0, 50% DD = 0.71)
- `sqrt` для Return: менше згладжування ніж `log` (100% = 1.0, 400% = 2.0)
- `suspicion_penalty`: штраф за "занадто ідеальні" результати

### 3. optimize_portfolio.py

**Призначення:** Оптимізація портфеля за Modern Portfolio Theory

#### Mean-Variance Optimization

```python
def optimize_portfolio(returns, cov_matrix, risk_free_rate):
    """
    Максимізує Sharpe Ratio: (portfolio_return - risk_free_rate) / portfolio_std

    Constraints:
      - sum(weights) = 1
      - 0 <= weight_i <= 1 (long-only)

    Method: scipy.optimize.minimize (SLSQP)
    """
```

#### Портфелі для порівняння

| Портфель | Опис |
|----------|------|
| `optimal` | Максимальний Sharpe Ratio (MPT) |
| `equal_weight` | Рівні ваги всіх активів |

#### Симуляція портфеля

```python
# Біжучи по investment_dates:
for date in get_biweekly_fridays(start, end):
    for ticker, weight in portfolio.items():
        shares_to_buy = (amount * weight) / price[ticker][date]
        holdings[ticker] += shares_to_buy
```

### 4. generate_report.py

**Призначення:** Генерація HTML звіту

#### Структура звіту

```html
<html>
  <head><!-- Inline CSS styles --></head>
  <body>
    <h1>Portfolio Analysis Report</h1>

    <!-- Portfolio configurations table -->
    <section class="portfolio-config">
      <table>ticker | optimal_weight | equal_weight</table>
    </section>

    <!-- Metrics comparison -->
    <section class="metrics">
      <table>metric | optimal | equal_weight</table>
    </section>

    <!-- Embedded charts (base64) -->
    <section class="charts">
      <img src="data:image/png;base64,{chart_data}" />
    </section>
  </body>
</html>
```

## Формули

### Метрики ризику

```python
# Sharpe Ratio
sharpe = (portfolio_return - risk_free_rate) / std(portfolio_returns)

# Sortino Ratio (тільки downside volatility)
downside_returns = returns[returns < 0]
sortino = (portfolio_return - risk_free_rate) / std(downside_returns)

# Maximum Drawdown
running_max = cummax(portfolio_value)
drawdown = (portfolio_value - running_max) / running_max
max_drawdown = min(drawdown)

# Calmar Ratio
calmar = CAGR / abs(max_drawdown)

# CAGR (Compound Annual Growth Rate)
cagr = (final_value / initial_value) ^ (1/years) - 1
```

### Інфляційні корекції

```python
# Реальне значення (в "початкових доларах")
base_cpi = CPI[start_date]
inflation_factor = base_cpi / CPI[current_date]  # < 1 при інфляції
real_value = nominal_value × inflation_factor
```

## Джерела даних

| Дані | API | Endpoint/Symbol |
|------|-----|-----------------|
| Ціни акцій | Yahoo Finance (yfinance) | `ticker` |
| CPI (інфляція) | FRED | `CPIAUCSL` |
| T-bill rates | FRED | `DTB3` (3-month) |

### Fallback при недоступності FRED

```python
# Якщо FRED недоступний, використовуємо синтетичні дані:
synthetic_cpi = 3% annual inflation
synthetic_tbill = 2% annual rate
```

## Структура файлів

```
project/
├── investment_simulation.py   # DCA симуляція
├── aggregate_results.py       # Агрегація метрик
├── optimize_portfolio.py      # MPT оптимізація
├── generate_report.py         # HTML звіт
├── run_simulations.sh         # Orchestration
├── test_investment_simulation.py
├── requirements.txt
├── README.md                  # Документація для користувачів
├── SPECIFICATION.md           # Цей файл
├── .cache/                    # Кеш даних (по роках)
│   └── {year}/
│       ├── {ticker}_{year}.pkl
│       ├── cpi_{year}.pkl
│       └── tbill_{year}.pkl
└── simulation_results/        # Результати
    ├── {ticker}_{period}_{amount}/
    │   ├── 1_stock_price.png
    │   ├── 2_portfolio_value.png
    │   ├── 3_portfolio_profit.png
    │   ├── 4_stocks_vs_tbills.png
    │   ├── 5_real_profit_comparison.png
    │   ├── 6_dashboard.png
    │   ├── metrics.csv
    │   └── simulation_data.csv
    ├── comparison.csv
    ├── portfolio_results.csv
    ├── portfolio_comparison.png
    └── report.html
```

## Залежності

```
yfinance>=0.2.28        # Yahoo Finance API
pandas>=2.0.0           # Data manipulation
pandas-datareader>=0.10.0  # FRED API
numpy>=1.24.0           # Numerical operations
matplotlib>=3.7.0       # Visualization
scipy>=1.11.0           # Optimization (SLSQP)
pytest>=7.4.0           # Testing
```

## Тестування

### Мокінг

```python
# Мокаємо load_cached_data для всіх data download функцій
@patch('investment_simulation.load_cached_data')
@patch('investment_simulation.save_cached_data')
def test_download_stock_data(mock_save, mock_load):
    mock_load.return_value = None  # Force download
    # ...
```

### Синтетичні дані для тестів

```python
def create_test_data(start, end, initial_price=100, volatility=0.02):
    dates = pd.date_range(start, end, freq='B')
    prices = initial_price * np.exp(np.cumsum(np.random.randn(len(dates)) * volatility))
    return pd.DataFrame({'Close': prices}, index=dates)
```

## Обмеження

- Комісії брокера не враховуються
- Податки не враховуються
- Дивіденди не враховуються для деяких індексів
- Дані інфляції та T-bills тільки для США
- Кеш не має автоматичного очищення (ручне видалення .cache/)
