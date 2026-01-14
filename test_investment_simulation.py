#!/usr/bin/env python3
"""
Unit tests for investment_simulation.py
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from investment_simulation import (
    get_biweekly_fridays,
    download_stock_data,
    download_inflation_data,
    download_tbill_rates,
    simulate_investment,
    create_visualizations,
    print_summary,
)


class TestGetBiweeklyFridays:
    """Tests for get_biweekly_fridays function."""

    def test_returns_list_of_fridays(self):
        """All returned dates should be Fridays."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        fridays = get_biweekly_fridays(start, end)

        for friday in fridays:
            assert friday.weekday() == 4, f"{friday} is not a Friday"

    def test_returns_biweekly_interval(self):
        """Dates should be exactly 14 days apart."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        fridays = get_biweekly_fridays(start, end)

        for i in range(1, len(fridays)):
            delta = (fridays[i] - fridays[i - 1]).days
            assert delta == 14, f"Interval between {fridays[i-1]} and {fridays[i]} is {delta}, not 14"

    def test_dates_within_range(self):
        """All dates should be within the specified range."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 6, 30)
        fridays = get_biweekly_fridays(start, end)

        for friday in fridays:
            assert friday >= start
            assert friday <= end

    def test_empty_range(self):
        """Should return empty list for invalid range."""
        start = datetime(2023, 6, 1)
        end = datetime(2023, 1, 1)
        fridays = get_biweekly_fridays(start, end)
        assert len(fridays) == 0

    def test_single_day_range_not_friday(self):
        """Single day range that's not a Friday should return empty."""
        start = datetime(2023, 1, 2)  # Monday
        end = datetime(2023, 1, 2)
        fridays = get_biweekly_fridays(start, end)
        assert len(fridays) == 0

    def test_single_day_range_is_friday(self):
        """Single day range that is a Friday should return it."""
        start = datetime(2023, 1, 6)  # Friday
        end = datetime(2023, 1, 6)
        fridays = get_biweekly_fridays(start, end)
        assert len(fridays) == 1
        assert fridays[0] == datetime(2023, 1, 6)

    def test_first_friday_finding(self):
        """Should find first Friday after start date."""
        start = datetime(2023, 1, 1)  # Sunday
        end = datetime(2023, 1, 31)
        fridays = get_biweekly_fridays(start, end)

        assert len(fridays) > 0
        assert fridays[0] == datetime(2023, 1, 6)  # First Friday in Jan 2023

    def test_year_has_approximately_26_biweekly_fridays(self):
        """A full year should have approximately 26 biweekly Fridays."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        fridays = get_biweekly_fridays(start, end)

        # 52 weeks / 2 = 26 biweekly periods
        assert 25 <= len(fridays) <= 27


class TestDownloadStockData:
    """Tests for download_stock_data function."""

    @patch('investment_simulation.yf.download')
    def test_converts_gspc_ticker(self, mock_download):
        """Should convert GSPC to ^GSPC."""
        mock_download.return_value = pd.DataFrame(
            {'Close': [100, 101]},
            index=pd.date_range('2023-01-01', periods=2)
        )

        download_stock_data('GSPC', '2023-01-01', '2023-01-31')
        mock_download.assert_called_with('^GSPC', start='2023-01-01', end='2023-01-31', progress=False)

    @patch('investment_simulation.yf.download')
    def test_converts_dji_ticker(self, mock_download):
        """Should convert DJI to ^DJI."""
        mock_download.return_value = pd.DataFrame(
            {'Close': [100, 101]},
            index=pd.date_range('2023-01-01', periods=2)
        )

        download_stock_data('DJI', '2023-01-01', '2023-01-31')
        mock_download.assert_called_with('^DJI', start='2023-01-01', end='2023-01-31', progress=False)

    @patch('investment_simulation.yf.download')
    def test_raises_on_empty_data(self, mock_download):
        """Should raise ValueError when no data returned."""
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="Не вдалося завантажити дані"):
            download_stock_data('INVALID', '2023-01-01', '2023-01-31')

    @patch('investment_simulation.yf.download')
    def test_handles_multiindex_columns(self, mock_download):
        """Should handle MultiIndex columns from yfinance."""
        df = pd.DataFrame(
            {'Close': [100, 101], 'Open': [99, 100]},
            index=pd.date_range('2023-01-01', periods=2)
        )
        df.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL'), ('Open', 'AAPL')])
        mock_download.return_value = df

        result = download_stock_data('AAPL', '2023-01-01', '2023-01-31')
        assert 'Close' in result.columns


class TestDownloadInflationData:
    """Tests for download_inflation_data function."""

    @patch('investment_simulation.pdr.DataReader')
    def test_returns_interpolated_daily_data(self, mock_reader):
        """Should return daily interpolated CPI data."""
        monthly_data = pd.DataFrame(
            {'CPIAUCSL': [100, 101, 102]},
            index=pd.date_range('2023-01-01', periods=3, freq='MS')
        )
        mock_reader.return_value = monthly_data

        result = download_inflation_data('2023-01-01', '2023-03-31')

        assert isinstance(result, pd.Series)
        assert len(result) > 3  # Should be interpolated to daily

    @patch('investment_simulation.pdr.DataReader')
    def test_fallback_on_error(self, mock_reader):
        """Should use synthetic CPI when download fails."""
        mock_reader.side_effect = Exception("Connection error")

        result = download_inflation_data('2023-01-01', '2023-12-31')

        assert isinstance(result, pd.Series)
        assert len(result) > 0


class TestDownloadTbillRates:
    """Tests for download_tbill_rates function."""

    @patch('investment_simulation.pdr.DataReader')
    def test_converts_percentage_to_decimal(self, mock_reader):
        """Should convert percentage to decimal (divide by 100)."""
        monthly_data = pd.DataFrame(
            {'TB3MS': [2.0, 2.5, 3.0]},  # Percentages
            index=pd.date_range('2023-01-01', periods=3, freq='MS')
        )
        mock_reader.return_value = monthly_data

        result = download_tbill_rates('2023-01-01', '2023-03-31')

        assert result.iloc[0] == pytest.approx(0.02, rel=0.1)

    @patch('investment_simulation.pdr.DataReader')
    def test_fallback_on_error(self, mock_reader):
        """Should use default rate when download fails."""
        mock_reader.side_effect = Exception("Connection error")

        result = download_tbill_rates('2023-01-01', '2023-12-31')

        assert isinstance(result, pd.Series)
        assert all(result == 0.02)


class TestSimulateInvestment:
    """Tests for simulate_investment function."""

    def create_mock_data(self, start_date='2023-01-01', periods=100, start_price=100):
        """Create mock stock data for testing."""
        dates = pd.date_range(start_date, periods=periods, freq='B')  # Business days
        prices = start_price + np.cumsum(np.random.randn(periods))
        prices = np.maximum(prices, 10)  # Ensure positive prices

        stock_data = pd.DataFrame({'Close': prices}, index=dates)
        cpi_data = pd.Series(100 + np.arange(periods) * 0.01, index=dates)
        tbill_rates = pd.Series(0.02, index=dates)

        return stock_data, cpi_data, tbill_rates, dates

    def test_returns_expected_keys(self):
        """Result should contain all expected keys."""
        stock_data, cpi_data, tbill_rates, dates = self.create_mock_data()
        investment_dates = [dates[0], dates[14], dates[28]]

        result = simulate_investment(
            stock_data, investment_dates, 500, cpi_data, tbill_rates
        )

        expected_keys = [
            'dates', 'stock_price', 'shares_owned', 'portfolio_value',
            'total_invested', 'portfolio_profit', 'portfolio_profit_pct',
            'tbill_value', 'tbill_profit', 'portfolio_value_real',
            'total_invested_real', 'portfolio_profit_real',
            'tbill_value_real', 'tbill_profit_real'
        ]

        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_shares_increase_after_purchase(self):
        """Shares owned should increase after each purchase."""
        stock_data, cpi_data, tbill_rates, dates = self.create_mock_data()
        investment_dates = [dates[0], dates[14], dates[28]]

        result = simulate_investment(
            stock_data, investment_dates, 500, cpi_data, tbill_rates
        )

        # Find indices of purchase dates
        purchase_indices = []
        for i, d in enumerate(result['dates']):
            if d in investment_dates:
                purchase_indices.append(i)

        # Shares should be increasing at purchase points
        for i in range(1, len(purchase_indices)):
            prev_idx = purchase_indices[i - 1]
            curr_idx = purchase_indices[i]
            # After purchase, shares should be higher
            assert result['shares_owned'][curr_idx] > result['shares_owned'][prev_idx]

    def test_total_invested_matches_purchases(self):
        """Total invested should equal number of purchases times amount."""
        stock_data, cpi_data, tbill_rates, dates = self.create_mock_data()
        investment_dates = [dates[0], dates[14], dates[28]]
        amount = 500

        result = simulate_investment(
            stock_data, investment_dates, amount, cpi_data, tbill_rates
        )

        expected_total = len(investment_dates) * amount
        assert result['total_invested'][-1] == pytest.approx(expected_total, rel=0.01)

    def test_profit_calculation(self):
        """Portfolio profit should be value minus invested."""
        stock_data, cpi_data, tbill_rates, dates = self.create_mock_data()
        investment_dates = [dates[0]]

        result = simulate_investment(
            stock_data, investment_dates, 500, cpi_data, tbill_rates
        )

        for i in range(len(result['dates'])):
            expected_profit = result['portfolio_value'][i] - result['total_invested'][i]
            assert result['portfolio_profit'][i] == pytest.approx(expected_profit, rel=0.01)

    def test_no_investment_dates_returns_zeros(self):
        """Empty investment dates should return zero values."""
        stock_data, cpi_data, tbill_rates, dates = self.create_mock_data()

        result = simulate_investment(
            stock_data, [], 500, cpi_data, tbill_rates
        )

        assert all(v == 0 for v in result['shares_owned'])
        assert all(v == 0 for v in result['total_invested'])

    def test_inflation_adjustment(self):
        """Real values should differ from nominal when CPI changes."""
        dates = pd.date_range('2023-01-01', periods=100, freq='B')
        stock_data = pd.DataFrame({'Close': [100] * 100}, index=dates)

        # Increasing CPI means inflation
        cpi_data = pd.Series(100 + np.arange(100) * 0.5, index=dates)
        tbill_rates = pd.Series(0.02, index=dates)

        result = simulate_investment(
            stock_data, [dates[0]], 1000, cpi_data, tbill_rates
        )

        # Real values should be adjusted (different from nominal at the end)
        # With base_cpi at start: at end, real values should be LOWER than nominal
        # because inflation reduces purchasing power
        assert result['portfolio_value_real'][-1] < result['portfolio_value'][-1]

        # At start (first purchase), real and nominal should be approximately equal
        # (inflation_factor ≈ 1 when cpi_current ≈ base_cpi)
        first_purchase_idx = 0
        assert result['portfolio_value_real'][first_purchase_idx] == pytest.approx(
            result['portfolio_value'][first_purchase_idx], rel=0.01
        )

    def test_cash_purchasing_power_decreases_with_inflation(self):
        """Cash (total_invested_real) should decrease relative to nominal with inflation."""
        dates = pd.date_range('2023-01-01', periods=100, freq='B')
        stock_data = pd.DataFrame({'Close': [100] * 100}, index=dates)

        # 50% CPI increase over period (significant inflation)
        cpi_data = pd.Series(100 + np.arange(100) * 0.5, index=dates)
        tbill_rates = pd.Series(0.0, index=dates)

        result = simulate_investment(
            stock_data, [dates[0], dates[14], dates[28]], 1000, cpi_data, tbill_rates
        )

        # At the end, cash purchasing power should be less than nominal
        # because inflation erodes the value of money
        assert result['total_invested_real'][-1] < result['total_invested'][-1]

        # The loss should be proportional to inflation
        inflation_rate = cpi_data.iloc[-1] / cpi_data.iloc[0]
        expected_real = result['total_invested'][-1] / inflation_rate
        assert result['total_invested_real'][-1] == pytest.approx(expected_real, rel=0.1)


class TestCreateVisualizations:
    """Tests for create_visualizations function."""

    def create_mock_results(self, n=100):
        """Create mock results for visualization tests."""
        dates = pd.date_range('2023-01-01', periods=n, freq='B')
        return {
            'dates': list(dates),
            'stock_price': list(100 + np.cumsum(np.random.randn(n))),
            'shares_owned': list(np.linspace(0, 10, n)),
            'portfolio_value': list(np.linspace(0, 5000, n)),
            'total_invested': list(np.linspace(0, 4000, n)),
            'portfolio_profit': list(np.linspace(-100, 1000, n)),
            'portfolio_profit_pct': list(np.linspace(-10, 25, n)),
            'tbill_value': list(np.linspace(0, 4200, n)),
            'tbill_profit': list(np.linspace(0, 200, n)),
            'portfolio_value_real': list(np.linspace(0, 4800, n)),
            'total_invested_real': list(np.linspace(0, 3800, n)),
            'portfolio_profit_real': list(np.linspace(-100, 950, n)),
            'tbill_value_real': list(np.linspace(0, 4000, n)),
            'tbill_profit_real': list(np.linspace(0, 150, n)),
        }

    def test_creates_all_expected_files(self):
        """Should create all expected visualization files."""
        results = self.create_mock_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            create_visualizations(
                results, 'TEST', 500, 2023, 2023, output_dir
            )

            expected_files = [
                '1_stock_price.png',
                '2_portfolio_value.png',
                '3_portfolio_profit.png',
                '4_stocks_vs_tbills.png',
                '5_real_profit_comparison.png',
                '6_dashboard.png',
            ]

            for filename in expected_files:
                filepath = output_dir / filename
                assert filepath.exists(), f"Missing file: {filename}"

    def test_files_have_content(self):
        """Generated files should have non-zero size."""
        results = self.create_mock_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            create_visualizations(
                results, 'TEST', 500, 2023, 2023, output_dir
            )

            for png_file in output_dir.glob('*.png'):
                assert png_file.stat().st_size > 0, f"Empty file: {png_file.name}"


class TestPrintSummary:
    """Tests for print_summary function."""

    def create_mock_results(self):
        """Create mock results for summary tests."""
        return {
            'dates': [datetime(2023, 12, 31)],
            'stock_price': [150.0],
            'shares_owned': [10.5],
            'portfolio_value': [1575.0],
            'total_invested': [1000.0],
            'portfolio_profit': [575.0],
            'portfolio_profit_pct': [57.5],
            'tbill_value': [1050.0],
            'tbill_profit': [50.0],
            'portfolio_value_real': [1500.0],
            'total_invested_real': [950.0],
            'portfolio_profit_real': [550.0],
            'tbill_value_real': [1000.0],
            'tbill_profit_real': [45.0],
        }

    def test_prints_without_error(self, capsys):
        """Should print summary without raising errors."""
        results = self.create_mock_results()
        print_summary(results, 'TEST', 500)

        captured = capsys.readouterr()
        assert 'TEST' in captured.out
        assert '500' in captured.out or '$500' in captured.out


class TestIntegration:
    """Integration tests for the full simulation flow."""

    @patch('investment_simulation.yf.download')
    @patch('investment_simulation.pdr.DataReader')
    def test_full_simulation_flow(self, mock_pdr, mock_yf):
        """Test complete simulation from data download to visualization."""
        # Setup mock data
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='B')
        stock_df = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(len(dates))) * 2,
            'Open': 99 + np.cumsum(np.random.randn(len(dates))) * 2,
            'High': 101 + np.cumsum(np.random.randn(len(dates))) * 2,
            'Low': 98 + np.cumsum(np.random.randn(len(dates))) * 2,
        }, index=dates)
        mock_yf.return_value = stock_df

        cpi_df = pd.DataFrame(
            {'CPIAUCSL': np.linspace(100, 102, 6)},
            index=pd.date_range('2023-01-01', periods=6, freq='MS')
        )
        tbill_df = pd.DataFrame(
            {'TB3MS': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]},
            index=pd.date_range('2023-01-01', periods=6, freq='MS')
        )
        mock_pdr.side_effect = [cpi_df, tbill_df]

        # Run simulation components
        stock_data = download_stock_data('TEST', '2023-01-01', '2023-06-30')
        cpi_data = download_inflation_data('2023-01-01', '2023-06-30')
        tbill_rates = download_tbill_rates('2023-01-01', '2023-06-30')

        investment_dates = get_biweekly_fridays(
            datetime(2023, 1, 1), datetime(2023, 6, 30)
        )

        # Filter to available dates
        available_dates = set(stock_data.index)
        investment_dates_ts = [
            pd.Timestamp(d) for d in investment_dates
            if pd.Timestamp(d) in available_dates
        ]

        if investment_dates_ts:
            results = simulate_investment(
                stock_data, investment_dates_ts, 500, cpi_data, tbill_rates
            )

            assert len(results['dates']) > 0
            assert results['total_invested'][-1] > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_biweekly_fridays_leap_year(self):
        """Should handle leap year correctly."""
        start = datetime(2024, 1, 1)  # 2024 is a leap year
        end = datetime(2024, 12, 31)
        fridays = get_biweekly_fridays(start, end)

        # Verify February 29 is handled
        feb_fridays = [f for f in fridays if f.month == 2]
        assert len(feb_fridays) >= 1

    def test_biweekly_fridays_year_boundary(self):
        """Should work across year boundaries."""
        start = datetime(2022, 12, 1)
        end = datetime(2023, 2, 28)
        fridays = get_biweekly_fridays(start, end)

        # Should have fridays in both years
        years = set(f.year for f in fridays)
        assert 2022 in years
        assert 2023 in years

    def test_simulation_with_constant_price(self):
        """Simulation should work with constant stock price."""
        dates = pd.date_range('2023-01-01', periods=100, freq='B')
        stock_data = pd.DataFrame({'Close': [100.0] * 100}, index=dates)
        cpi_data = pd.Series([100.0] * 100, index=dates)
        tbill_rates = pd.Series([0.02] * 100, index=dates)

        result = simulate_investment(
            stock_data, [dates[0], dates[14]], 500, cpi_data, tbill_rates
        )

        # With constant price and no inflation adjustment, profit should be 0
        # (aside from T-bill interest)
        assert result['portfolio_profit'][-1] == pytest.approx(0, abs=1)

    def test_simulation_with_single_purchase(self):
        """Should handle single purchase correctly."""
        dates = pd.date_range('2023-01-01', periods=50, freq='B')
        prices = [100.0] + [110.0] * 49  # Jump after purchase
        stock_data = pd.DataFrame({'Close': prices}, index=dates)
        cpi_data = pd.Series([100.0] * 50, index=dates)
        tbill_rates = pd.Series([0.0] * 50, index=dates)

        result = simulate_investment(
            stock_data, [dates[0]], 1000, cpi_data, tbill_rates
        )

        # Should have 10 shares (1000/100), worth 1100 at the end (10 * 110)
        assert result['shares_owned'][-1] == pytest.approx(10.0, rel=0.01)
        assert result['portfolio_value'][-1] == pytest.approx(1100.0, rel=0.01)
        assert result['portfolio_profit'][-1] == pytest.approx(100.0, rel=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
