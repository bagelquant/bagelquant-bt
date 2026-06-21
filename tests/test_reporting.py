from __future__ import annotations

import polars as pl

from bagelquant_bt import (
    BacktestConfig,
    run_factor_evaluation,
    run_weight_backtest,
    summary_report,
)
from bagelquant_bt.reporting import _dataframe_to_html
from bagelquant_bt.visualization import _label_with_mean


def test_summary_report_renders_and_writes_backtest_html(tmp_path) -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["a", "a", "a"],
            "price": [1.0, 1.1, 1.2],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-04"],
            "asset_id": ["a", "a", "a"],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=1_000, annualization=2),
    )
    output_path = tmp_path / "backtest_report.html"
    missing_keys_path = tmp_path / "backtest_report_missing_price_keys.csv"

    html = summary_report(result, output_path=output_path, annualization=2)

    assert html.startswith("<!doctype html>")
    assert "Performance" in html
    assert "Trading Summary" in html
    assert "Missing Price Keys" not in html
    assert "Portfolio Cumulative Returns" in html
    assert "Plotly.newPlot" in html
    assert "<h3>Returns</h3>" not in html
    assert "<h3>Value</h3>" not in html
    assert "<h3>Transaction Costs</h3>" not in html
    assert output_path.read_text(encoding="utf-8") == html
    assert pl.read_csv(missing_keys_path).to_dicts() == [
        {"time": "2024-01-04", "asset_id": "a"}
    ]


def test_summary_report_renders_factor_tables_and_plots() -> None:
    prices = pl.DataFrame(
        {
            "time": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            "asset_id": ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
            "price": [1.0, 1.1, 1.2, 1.3, 1.0, 0.9, 0.8, 0.7, 1.0, 1.0, 1.1, 1.2],
        }
    )
    factor = prices.select("time", "asset_id").with_columns(
        pl.when(pl.col("asset_id") == "a")
        .then(3.0)
        .when(pl.col("asset_id") == "c")
        .then(2.0)
        .otherwise(1.0)
        .alias("factor")
    )
    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(initial_capital=1_000, annualization=4, quantiles=3),
    )

    html = summary_report(result, annualization=4)

    assert "<h3>Summary</h3>" in html
    assert "Spread SR" in html
    assert "rankICIR" in html
    assert "N/A" in html
    assert "<h3>IC Summary</h3>" not in html
    assert "<h3>IC Decay</h3>" not in html
    assert "TOP N Lag Analysis" in html
    assert "TOP N Performance" in html
    assert "<h3>Spread Performance</h3>" in html
    assert "<h3>Long-Short Lag Analysis</h3>" not in html
    assert "<h3>Spread Summary</h3>" not in html
    assert "Annualized Volatility" in html
    assert "Transaction Cost" in html
    assert "<th>EAR</th>" in html
    assert "<th>SR</th>" in html
    assert "Quantile Performance" in html
    assert "Missing Price Keys" not in html
    assert "Information Coefficient" in html
    assert "IC Distribution" in html
    assert "TOP N Gross Lag Cumulative Returns" in html
    assert "Long-Short Net Lag Cumulative Returns" in html
    assert "<h3>IC Series</h3>" not in html
    assert "<h3>Quantile Returns</h3>" not in html
    assert "<h3>Top Minus Bottom</h3>" not in html
    section_order = [
        html.index("<h2>IC and ICIR</h2>"),
        html.index("<h2>Quantile Performance</h2>"),
        html.index("<h2>Spread Performance</h2>"),
        html.index("<h2>TOP N</h2>"),
    ]
    assert section_order == sorted(section_order)
    assert html.index("<h3>Summary</h3>") < html.index("<h2>IC and ICIR</h2>")
    assert html.index("<h3>TOP N Performance</h3>") < html.index(
        "TOP N Cumulative Returns"
    )
    assert html.index("<h3>Spread Performance</h3>") < html.index(
        "Long-Short Cumulative Returns"
    )


def test_report_numeric_text_uses_four_significant_digits() -> None:
    table = _dataframe_to_html(pl.DataFrame({"value": [1.23456, 12345.6]}))

    assert ">1.235<" in table
    assert ">1.235e+04<" in table
    assert _label_with_mean("Example", pl.Series([1.23456])) == "Example (avg 1.235)"


def test_summary_report_writes_missing_price_keys_to_explicit_csv(
    tmp_path,
) -> None:
    prices = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset_id": ["a", "a", "a"],
            "price": [1.0, 1.1, 1.2],
        }
    )
    weights = pl.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-04"],
            "asset_id": ["a", "b"],
            "weight": [1.0, 1.0],
        }
    )
    result = run_weight_backtest(
        weights,
        prices,
        config=BacktestConfig(initial_capital=1_000, annualization=2),
    )
    missing_keys_path = tmp_path / "missing_keys.csv"

    summary_report(result, missing_price_keys_output_path=missing_keys_path)

    assert pl.read_csv(missing_keys_path).to_dicts() == [
        {"time": "2024-01-04", "asset_id": "b"}
    ]
