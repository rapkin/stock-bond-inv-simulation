#!/usr/bin/env python3
"""
Unit tests for aggregate_results.py
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from aggregate_results import (
    find_metrics_files,
    aggregate_metrics,
    format_comparison_table,
)


class TestFindMetricsFiles:
    """Tests for find_metrics_files function."""

    def test_finds_metrics_files(self):
        """Should find all metrics.csv files in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create test structure
            (results_dir / 'sim1').mkdir()
            (results_dir / 'sim2').mkdir()
            (results_dir / 'sim1' / 'metrics.csv').write_text('test')
            (results_dir / 'sim2' / 'metrics.csv').write_text('test')

            files = find_metrics_files(results_dir)

            assert len(files) == 2

    def test_returns_empty_for_no_files(self):
        """Should return empty list when no metrics files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            files = find_metrics_files(results_dir)

            assert len(files) == 0

    def test_ignores_non_metrics_files(self):
        """Should only find files named metrics.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            (results_dir / 'sim1').mkdir()
            (results_dir / 'sim1' / 'data.csv').write_text('test')
            (results_dir / 'sim1' / 'other_metrics.csv').write_text('test')

            files = find_metrics_files(results_dir)

            assert len(files) == 0


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""

    def create_metrics_csv(self, path: Path, ticker: str, sharpe: float):
        """Helper to create a metrics CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{
            'ticker': ticker,
            'start_year': 2023,
            'end_year': 2023,
            'investment_amount': 500,
            'total_invested': 10000,
            'final_value': 12000,
            'total_return_pct': 20.0,
            'cagr_pct': 20.0,
            'annual_volatility_pct': 15.0,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sharpe * 1.2,
            'max_drawdown_pct': -10.0,
            'calmar_ratio': 2.0,
            'win_rate_pct': 55.0,
            'max_underwater_days': 30,
            'risk_free_rate_pct': 2.0,
        }])
        df.to_csv(path, index=False)

    def test_aggregates_multiple_files(self):
        """Should combine metrics from multiple simulations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            self.create_metrics_csv(results_dir / 'AAPL' / 'metrics.csv', 'AAPL', 1.5)
            self.create_metrics_csv(results_dir / 'MSFT' / 'metrics.csv', 'MSFT', 2.0)

            df = aggregate_metrics(results_dir)

            assert len(df) == 2
            assert set(df['ticker'].tolist()) == {'AAPL', 'MSFT'}

    def test_returns_empty_for_no_files(self):
        """Should return empty DataFrame when no files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            df = aggregate_metrics(results_dir)

            assert df.empty

    def test_adds_source_dir_column(self):
        """Should add source_dir column to track origin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            self.create_metrics_csv(results_dir / 'TEST_2023' / 'metrics.csv', 'TEST', 1.0)

            df = aggregate_metrics(results_dir)

            assert 'source_dir' in df.columns
            assert df['source_dir'].iloc[0] == 'TEST_2023'


class TestFormatComparisonTable:
    """Tests for format_comparison_table function."""

    def test_selects_key_columns(self):
        """Should select only key columns for comparison."""
        df = pd.DataFrame([{
            'ticker': 'TEST',
            'start_year': 2023,
            'end_year': 2023,
            'sharpe_ratio': 1.5,
            'extra_column': 'should_be_removed',
            'source_dir': 'test_dir',
        }])

        result = format_comparison_table(df)

        assert 'ticker' in result.columns
        assert 'sharpe_ratio' in result.columns
        assert 'extra_column' not in result.columns
        assert 'source_dir' not in result.columns

    def test_sorts_by_sharpe_ratio(self):
        """Should sort by Sharpe ratio descending."""
        df = pd.DataFrame([
            {'ticker': 'LOW', 'sharpe_ratio': 0.5, 'start_year': 2023, 'end_year': 2023},
            {'ticker': 'HIGH', 'sharpe_ratio': 2.0, 'start_year': 2023, 'end_year': 2023},
            {'ticker': 'MID', 'sharpe_ratio': 1.0, 'start_year': 2023, 'end_year': 2023},
        ])

        result = format_comparison_table(df)

        assert result['ticker'].iloc[0] == 'HIGH'
        assert result['ticker'].iloc[2] == 'LOW'

    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame()

        result = format_comparison_table(df)

        assert result.empty


class TestIntegration:
    """Integration tests for the full aggregation flow."""

    def test_full_aggregation_flow(self):
        """Test complete aggregation from files to formatted output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)

            # Create realistic metrics files
            for ticker, sharpe in [('GSPC', 1.8), ('BTC-USD', 1.2), ('AAPL', 2.1)]:
                sim_dir = results_dir / f'{ticker}_2022-2025_$500'
                sim_dir.mkdir()
                df = pd.DataFrame([{
                    'ticker': ticker,
                    'start_year': 2022,
                    'end_year': 2025,
                    'investment_amount': 500,
                    'total_invested': 52000,
                    'final_value': 70000 + sharpe * 10000,
                    'total_return_pct': 30 + sharpe * 10,
                    'cagr_pct': 8 + sharpe * 2,
                    'annual_volatility_pct': 20 - sharpe * 2,
                    'sharpe_ratio': sharpe,
                    'sortino_ratio': sharpe * 1.5,
                    'max_drawdown_pct': -15 + sharpe * 2,
                    'calmar_ratio': sharpe * 0.5,
                    'win_rate_pct': 50 + sharpe * 5,
                    'max_underwater_days': 60 - int(sharpe * 10),
                    'risk_free_rate_pct': 2.0,
                }])
                df.to_csv(sim_dir / 'metrics.csv', index=False)

            # Run aggregation
            aggregated = aggregate_metrics(results_dir)
            formatted = format_comparison_table(aggregated)

            # Verify results
            assert len(formatted) == 3
            assert formatted['ticker'].iloc[0] == 'AAPL'  # Highest Sharpe
            assert formatted['ticker'].iloc[2] == 'BTC-USD'  # Lowest Sharpe


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
