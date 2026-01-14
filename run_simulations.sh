#!/bin/bash
#
# Investment Simulation Runner
# Запускає симуляції для списку тікерів та агрегує результати
#
# Використання:
#   ./run_simulations.sh
#   ./run_simulations.sh -s 2020 -e 2024 -a 1000 -t "AAPL MSFT GOOGL"
#

# === ПАРАМЕТРИ ЗА ЗАМОВЧУВАННЯМ ===
START_YEAR=2020
END_YEAR=2025
AMOUNT=500

# Тікери за замовчуванням:
#   Індекси:
#     GSPC     - S&P 500
#     DJI      - Dow Jones Industrial Average
#     IXIC     - Nasdaq Composite
#     QQQ      - Invesco QQQ (Nasdaq 100)
#   Регіони:
#     EZU      - iShares MSCI Eurozone ETF (Євросоюз)
#     EEM      - iShares MSCI Emerging Markets (країни що розвиваються)
#   Облігації:
#     TLT      - iShares 20+ Year Treasury Bond (довгострокові облігації)
#   Commodities:
#     GLD      - SPDR Gold Shares (золото)
#   Нерухомість:
#     VNQ      - Vanguard Real Estate (REITs)
#   Криптовалюти:
#     BTC-USD  - Bitcoin
#     ETH-USD  - Ethereum
#     SOL-USD  - Solana
TICKERS="GSPC DJI IXIC QQQ EZU EEM TLT GLD VNQ BTC-USD ETH-USD SOL-USD"

# === ПАРСИНГ АРГУМЕНТІВ ===
while getopts "s:e:a:t:h" opt; do
    case $opt in
        s) START_YEAR="$OPTARG" ;;
        e) END_YEAR="$OPTARG" ;;
        a) AMOUNT="$OPTARG" ;;
        t) TICKERS="$OPTARG" ;;
        h)
            echo "Investment Simulation Runner"
            echo ""
            echo "Використання: $0 [опції]"
            echo ""
            echo "Опції:"
            echo "  -s YEAR    Початковий рік (за замовчуванням: $START_YEAR)"
            echo "  -e YEAR    Кінцевий рік (за замовчуванням: $END_YEAR)"
            echo "  -a AMOUNT  Сума інвестиції кожні 2 тижні (за замовчуванням: $AMOUNT)"
            echo "  -t TICKERS Список тікерів в лапках (за замовчуванням: \"$TICKERS\")"
            echo "  -h         Показати цю довідку"
            echo ""
            echo "Приклади:"
            echo "  $0"
            echo "  $0 -s 2020 -e 2024 -a 1000"
            echo "  $0 -t \"AAPL MSFT GOOGL BTC-USD\""
            echo "  $0 -s 2015 -e 2025 -a 500 -t \"GSPC AAPL MSFT BTC-USD ETH-USD\""
            exit 0
            ;;
        \?)
            echo "Невідома опція: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# === ВИЗНАЧЕННЯ ШЛЯХІВ ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# === ПЕРЕВІРКА ВІРТУАЛЬНОГО ОТОЧЕННЯ ===
if [ ! -d "$VENV_DIR" ]; then
    echo "Віртуальне оточення не знайдено. Створюю..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    source "$VENV_DIR/bin/activate"
fi

# === ВИВІД ПАРАМЕТРІВ ===
echo "========================================"
echo "Investment Simulation Runner"
echo "========================================"
echo "Період:     $START_YEAR - $END_YEAR"
echo "Інвестиція: \$$AMOUNT кожні 2 тижні"
echo "Тікери:     $TICKERS"
echo "========================================"
echo ""

# === ЗАПУСК СИМУЛЯЦІЙ ===
FAILED=0
SUCCESSFUL=0

for TICKER in $TICKERS; do
    echo "----------------------------------------"
    echo "Запуск симуляції для: $TICKER"
    echo "----------------------------------------"

    python "$SCRIPT_DIR/investment_simulation.py" \
        --ticker "$TICKER" \
        --start "$START_YEAR" \
        --end "$END_YEAR" \
        --amount "$AMOUNT"

    if [ $? -eq 0 ]; then
        ((SUCCESSFUL++))
        echo ""
    else
        ((FAILED++))
        echo "ПОМИЛКА: Симуляція для $TICKER завершилась з помилкою"
        echo ""
    fi
done

# === АГРЕГАЦІЯ РЕЗУЛЬТАТІВ ===
echo "========================================"
echo "Агрегація результатів..."
echo "========================================"

python "$SCRIPT_DIR/aggregate_results.py"

# === ПІДСУМОК ===
echo ""
echo "========================================"
echo "ЗАВЕРШЕНО"
echo "========================================"
echo "Успішно: $SUCCESSFUL"
echo "Помилок: $FAILED"
echo "Результати: $SCRIPT_DIR/simulation_results/"
echo "Порівняння: $SCRIPT_DIR/simulation_results/comparison.csv"
echo "========================================"

# Деактивація віртуального оточення
deactivate
