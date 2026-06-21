"""HTML reporting helpers for backtest and factor results."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Protocol

import plotly.graph_objects as go
import polars as pl

from .results import BacktestResult, FactorEvaluationResult
from .visualization import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_ic,
    plot_ic_decay,
    plot_ic_distribution,
    plot_lag_cumulative_return,
    plot_lag_sharpe,
    plot_quantile_cumulative_returns,
    plot_rolling_ic,
    plot_rolling_sharpe,
    plot_rolling_volatility,
    plot_turnover_and_costs,
)


class SupportsWrite(Protocol):
    def write(self, data: str) -> object: ...


def summary_report(
    result: BacktestResult | FactorEvaluationResult,
    *,
    output_path: str | Path | None = None,
    missing_price_keys_output_path: str | Path | None = None,
    title: str | None = None,
    annualization: int = 252,
) -> str:
    """Build a self-contained HTML report for a backtest or factor result."""

    if isinstance(result, FactorEvaluationResult):
        report_title = title or "Factor Evaluation Summary Report"
        body = _factor_report(result, annualization=annualization)
    elif isinstance(result, BacktestResult):
        report_title = title or "Backtest Summary Report"
        body = _backtest_report(result, annualization=annualization)
    else:
        raise TypeError("result must be BacktestResult or FactorEvaluationResult")

    html = _document(report_title, body)
    if output_path is not None:
        html_path = Path(output_path)
        html_path.write_text(html, encoding="utf-8")
        csv_path = missing_price_keys_output_path or _default_missing_price_keys_path(
            html_path
        )
        _write_missing_price_keys_csv(
            result.missing_price_keys,
            csv_path,
        )
    elif missing_price_keys_output_path is not None:
        _write_missing_price_keys_csv(
            result.missing_price_keys,
            missing_price_keys_output_path,
        )
    return html


def _backtest_report(result: BacktestResult, *, annualization: int) -> str:
    tables = [
        _table_section("Performance", result.performance),
        _table_section("Trading Summary", _trading_summary(result)),
    ]
    figures = [
        plot_cumulative_returns(result, title="Portfolio Cumulative Returns"),
        plot_drawdown(result),
        plot_turnover_and_costs(result),
        plot_rolling_sharpe(result, annualization=annualization),
        plot_rolling_volatility(result, annualization=annualization),
    ]
    return _section("Tables", "".join(tables)) + _figures_section("Plots", figures)


def _factor_report(result: FactorEvaluationResult, *, annualization: int) -> str:
    sections = [
        _table_section("Summary", _factor_summary(result), none_display="N/A"),
        _factor_ic_section(result),
        _factor_quantile_section(result),
        _factor_spread_section(result, annualization=annualization),
        _factor_top_n_section(result, annualization=annualization),
    ]
    return "".join(sections)


def _default_missing_price_keys_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_missing_price_keys.csv")


def _write_missing_price_keys_csv(
    missing_price_keys: pl.DataFrame,
    output_path: str | Path,
) -> None:
    missing_price_keys.write_csv(Path(output_path))


def _factor_ic_section(result: FactorEvaluationResult) -> str:
    body = "".join(
        [
            _figure_to_html(plot_ic(result)),
            _figure_to_html(plot_rolling_ic(result)),
            _figure_to_html(plot_ic_distribution(result)),
            _figure_to_html(plot_ic_decay(result)),
        ]
    )
    return _section("IC and ICIR", body)


def _factor_top_n_section(
    result: FactorEvaluationResult,
    *,
    annualization: int,
) -> str:
    lag_returns = plot_lag_cumulative_return(result)
    body = "".join(
        [
            _table_section(
                "TOP N Performance",
                _standard_performance_table(result.top_n_backtest),
            ),
            _table_section("TOP N Lag Analysis", _lag_analysis(result, "top_n")),
            _figure_to_html(
                plot_cumulative_returns(
                    result.top_n_backtest,
                    title="TOP N Cumulative Returns",
                )
            ),
            _figure_to_html(plot_drawdown(result.top_n_backtest)),
            _figure_to_html(plot_turnover_and_costs(result.top_n_backtest)),
            _figure_to_html(
                plot_rolling_sharpe(
                    result.top_n_backtest,
                    annualization=annualization,
                )
            ),
            _figure_to_html(
                plot_rolling_volatility(
                    result.top_n_backtest,
                    annualization=annualization,
                )
            ),
            _figure_to_html(lag_returns[0]),
            _figure_to_html(lag_returns[1]),
        ]
    )
    return _section("TOP N", body)


def _factor_spread_section(
    result: FactorEvaluationResult,
    *,
    annualization: int,
) -> str:
    tables: list[str] = []
    if result.long_short_backtest is not None:
        tables.extend(
            [
                _table_section(
                    "Spread Performance",
                    _standard_performance_table(result.long_short_backtest),
                ),
            ]
        )
    lag_returns = plot_lag_cumulative_return(result)
    figures = [_figure_to_html(plot_lag_sharpe(result))]
    if result.long_short_backtest is not None:
        figures.extend(
            [
                _figure_to_html(
                    plot_cumulative_returns(
                        result.long_short_backtest,
                        title="Long-Short Cumulative Returns",
                    )
                ),
                _figure_to_html(plot_drawdown(result.long_short_backtest)),
                _figure_to_html(plot_turnover_and_costs(result.long_short_backtest)),
                _figure_to_html(
                    plot_rolling_sharpe(
                        result.long_short_backtest,
                        annualization=annualization,
                    )
                ),
                _figure_to_html(
                    plot_rolling_volatility(
                        result.long_short_backtest,
                        annualization=annualization,
                    )
                ),
                _figure_to_html(lag_returns[2]),
                _figure_to_html(lag_returns[3]),
            ]
        )
    return _section("Spread Performance", "".join(tables + figures))


def _factor_quantile_section(result: FactorEvaluationResult) -> str:
    quantile_summary = (
        result.quantile_returns.group_by("quantile")
        .agg(
            pl.col("return").mean().alias("mean_return"),
            pl.col("return").std().alias("std_return"),
            pl.col("cumulative_return").last().alias("final_cumulative_return"),
        )
        .sort("quantile")
    )
    body = "".join(
        [
            _table_section("Quantile Performance", quantile_summary),
            _figure_to_html(plot_quantile_cumulative_returns(result)),
        ]
    )
    return _section("Quantile Performance", body)


def _lag_analysis(result: FactorEvaluationResult, portfolio: str) -> pl.DataFrame:
    return result.lag_analysis.filter(result.lag_analysis["portfolio"] == portfolio)


def _factor_summary(result: FactorEvaluationResult) -> pl.DataFrame:
    """Return headline factor metrics in gross/net rows for the report."""

    ic = {row["method"]: row for row in result.ic_summary.iter_rows(named=True)}
    pearson = ic.get("pearson", {})
    spearman = ic.get("spearman", {})
    spread = result.long_short_backtest.summary if result.long_short_backtest else None
    top_n = result.top_n_backtest.summary
    return pl.DataFrame(
        [
            {
                "Portfolio": "Gross",
                "Spread SR": spread.gross_sharpe if spread else None,
                "IC": pearson.get("mean"),
                "rankIC": spearman.get("mean"),
                "ICIR": pearson.get("icir"),
                "rankICIR": spearman.get("icir"),
                "Spread EAR": spread.gross_annualized_return if spread else None,
                "TOP N EAR": top_n.gross_annualized_return,
                "TOP N SR": top_n.gross_sharpe,
            },
            {
                "Portfolio": "Net",
                "Spread SR": spread.net_sharpe if spread else None,
                "IC": None,
                "rankIC": None,
                "ICIR": None,
                "rankICIR": None,
                "Spread EAR": spread.net_annualized_return if spread else None,
                "TOP N EAR": top_n.net_annualized_return,
                "TOP N SR": top_n.net_sharpe,
            },
        ]
    )


def _standard_performance_table(result: BacktestResult) -> pl.DataFrame:
    """Return the common gross/net performance table used in factor reports."""

    summary = result.summary
    return pl.DataFrame(
        [
            {
                "Portfolio": "Gross",
                "Total Return": summary.gross_total_return,
                "EAR": summary.gross_annualized_return,
                "Annualized Volatility": summary.gross_annualized_volatility,
                "SR": summary.gross_sharpe,
                "Max Drawdown": summary.gross_max_drawdown,
                "Average Turnover": summary.average_turnover,
                "Transaction Cost": None,
            },
            {
                "Portfolio": "Net",
                "Total Return": summary.net_total_return,
                "EAR": summary.net_annualized_return,
                "Annualized Volatility": summary.net_annualized_volatility,
                "SR": summary.net_sharpe,
                "Max Drawdown": summary.net_max_drawdown,
                "Average Turnover": summary.average_turnover,
                "Transaction Cost": summary.total_transaction_cost,
            },
        ]
    )


def _trading_summary(result: BacktestResult) -> pl.DataFrame:
    costs = result.transaction_costs.data
    total_fee = float(costs["total_fee"].sum()) if costs.height else 0.0  # type: ignore
    total_notional = (
        float(costs["traded_notional"].sum()) if costs.height else 0.0  # type: ignore
    )
    average_turnover = (
        float(result.turnover["turnover"].mean()) if result.turnover.height else None  # type: ignore
    )
    return pl.DataFrame(
        [
            {"metric": "average_turnover", "value": average_turnover},
            {"metric": "total_traded_notional", "value": total_notional},
            {"metric": "total_transaction_cost", "value": total_fee},
        ],
        schema={"metric": pl.String, "value": pl.Float64},
    )


def _document(title: str, body: str) -> str:
    escaped_title = escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      line-height: 1.5;
      color: #172033;
      background: #f6f8fb;
    }}
    body {{
      margin: 0;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    h1, h2, h3 {{
      line-height: 1.2;
    }}
    h1 {{
      margin: 0 0 24px;
      font-size: 32px;
    }}
    h2 {{
      margin: 32px 0 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid #d9e0ea;
      font-size: 24px;
    }}
    h3 {{
      margin: 24px 0 10px;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 0 0 18px;
      background: white;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #dce3ed;
      padding: 7px 9px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      background: #edf2f7;
      font-weight: 650;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .plot {{
      margin: 18px 0 28px;
      background: white;
      border: 1px solid #dce3ed;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escaped_title}</h1>
    {body}
  </main>
</body>
</html>
"""


def _section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _table_section(
    title: str,
    frame: pl.DataFrame,
    *,
    none_display: str = "",
) -> str:
    table = _dataframe_to_html(frame, none_display=none_display)
    return f'<h3>{escape(title)}</h3><div class="table-wrap">{table}</div>'


def _figures_section(title: str, figures: list[go.Figure]) -> str:
    body = "".join(_figure_to_html(figure) for figure in figures)
    return _section(title, body)


def _dataframe_to_html(frame: pl.DataFrame, *, none_display: str = "") -> str:
    if frame.is_empty():
        return "<p>No rows.</p>"
    header = "".join(f"<th>{escape(column)}</th>" for column in frame.columns)
    rows = []
    for row in frame.iter_rows(named=True):
        cells = "".join(
            f"<td>{escape(_format_value(row[column], none_display=none_display))}</td>"
            for column in frame.columns
        )
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _figure_to_html(figure: go.Figure) -> str:
    return (
        '<div class="plot">'
        + figure.to_html(full_html=False, include_plotlyjs=False)
        + "</div>"
    )


def _format_value(value: object, *, none_display: str = "") -> str:
    if value is None:
        return none_display
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
