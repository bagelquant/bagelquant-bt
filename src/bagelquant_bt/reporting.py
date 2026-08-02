"""HTML reporting helpers for backtest and factor results."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Protocol

import plotly.graph_objects as go
import polars as pl

from .results import BacktestResult, FactorEvaluationResult
from .visualization import (
    plot_benchmark_coverage,
    plot_benchmark_cumulative_returns,
    plot_coverage,
    plot_cumulative_returns,
    plot_drawdown,
    plot_excess_returns,
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


@dataclass(frozen=True, slots=True)
class ReportFigure:
    """One ordered, named Plotly figure in a backtest report."""

    key: str
    section: str
    title: str
    figure: go.Figure


def factor_evaluation_report_figures(
    result: FactorEvaluationResult,
    *,
    annualization: int | None = None,
) -> tuple[ReportFigure, ...]:
    """Return the complete, ordered figure inventory for factor evaluation reports."""

    lag_returns = plot_lag_cumulative_return(result)
    figures = [
        ReportFigure(
            "coverage",
            "Summary & Coverage",
            "Signal Coverage",
            plot_coverage(result),
        ),
        ReportFigure("ic", "IC & ICIR", "Information Coefficient", plot_ic(result)),
        ReportFigure("rolling_ic", "IC & ICIR", "Rolling IC", plot_rolling_ic(result)),
        ReportFigure(
            "ic_distribution",
            "IC & ICIR",
            "IC Distribution",
            plot_ic_distribution(result),
        ),
        ReportFigure("ic_decay", "IC & ICIR", "IC Decay", plot_ic_decay(result)),
        ReportFigure(
            "quantile_cumulative_returns",
            "Quantiles",
            "Quantile Cumulative Returns",
            plot_quantile_cumulative_returns(result),
        ),
        ReportFigure(
            "top_n_cumulative_returns",
            "TOP N",
            "TOP N Cumulative Returns",
            plot_cumulative_returns(
                result.top_n_backtest, title="TOP N Cumulative Returns"
            ),
        ),
        ReportFigure(
            "top_n_drawdown", "TOP N", "Drawdown", plot_drawdown(result.top_n_backtest)
        ),
        ReportFigure(
            "top_n_turnover_and_costs",
            "TOP N",
            "Turnover and Transaction Costs",
            plot_turnover_and_costs(result.top_n_backtest),
        ),
        ReportFigure(
            "top_n_rolling_sharpe",
            "TOP N",
            "Rolling Sharpe",
            plot_rolling_sharpe(result.top_n_backtest, annualization=annualization),
        ),
        ReportFigure(
            "top_n_rolling_volatility",
            "TOP N",
            "Rolling Volatility",
            plot_rolling_volatility(result.top_n_backtest, annualization=annualization),
        ),
        ReportFigure(
            "top_n_gross_lag_cumulative_returns",
            "TOP N",
            "TOP N Gross Lag Cumulative Returns",
            lag_returns[0],
        ),
        ReportFigure(
            "top_n_net_lag_cumulative_returns",
            "TOP N",
            "TOP N Net Lag Cumulative Returns",
            lag_returns[1],
        ),
    ]
    figures.extend(
        [
            ReportFigure(
                "benchmark_cumulative_returns",
                "TOP N vs Benchmarks",
                "TOP N vs Benchmarks",
                plot_benchmark_cumulative_returns(result),
            ),
            ReportFigure(
                "benchmark_coverage",
                "TOP N vs Benchmarks",
                "Benchmark Coverage",
                plot_benchmark_coverage(result),
            ),
        ]
    )
    for benchmark in result.benchmark_returns.get_column("benchmark").unique(
        maintain_order=True
    ):
        key = str(benchmark).replace("-", "_").replace(" ", "_")
        figures.append(
            ReportFigure(
                f"excess_return_{key}",
                "TOP N vs Benchmarks",
                f"TOP N Excess Return vs {benchmark}",
                plot_excess_returns(result, str(benchmark)),
            )
        )
    figures.append(
        ReportFigure(
            "lag_sharpe",
            "Spread Performance",
            "Lag Analysis Sharpe",
            plot_lag_sharpe(result),
        )
    )
    if result.spread_backtest is not None:
        figures.extend(
            [
                ReportFigure(
                    "spread_cumulative_returns",
                    "Spread Performance",
                    "Spread Cumulative Returns",
                    plot_cumulative_returns(
                        result.spread_backtest, title="Spread Cumulative Returns"
                    ),
                ),
                ReportFigure(
                    "spread_drawdown",
                    "Spread Performance",
                    "Drawdown",
                    plot_drawdown(result.spread_backtest),
                ),
                ReportFigure(
                    "spread_turnover_and_costs",
                    "Spread Performance",
                    "Turnover and Transaction Costs",
                    plot_turnover_and_costs(result.spread_backtest),
                ),
                ReportFigure(
                    "spread_rolling_sharpe",
                    "Spread Performance",
                    "Rolling Sharpe",
                    plot_rolling_sharpe(
                        result.spread_backtest, annualization=annualization
                    ),
                ),
                ReportFigure(
                    "spread_rolling_volatility",
                    "Spread Performance",
                    "Rolling Volatility",
                    plot_rolling_volatility(
                        result.spread_backtest, annualization=annualization
                    ),
                ),
                ReportFigure(
                    "spread_gross_lag_cumulative_returns",
                    "Spread Performance",
                    "Spread Gross Lag Cumulative Returns",
                    lag_returns[2],
                ),
                ReportFigure(
                    "spread_net_lag_cumulative_returns",
                    "Spread Performance",
                    "Spread Net Lag Cumulative Returns",
                    lag_returns[3],
                ),
            ]
        )
    return tuple(figures)


def summary_report(
    result: BacktestResult | FactorEvaluationResult,
    *,
    output_path: str | Path | None = None,
    missing_price_keys_output_path: str | Path | None = None,
    title: str | None = None,
    annualization: int | None = None,
) -> str:
    """Build a self-contained HTML report for a backtest or factor result."""

    return _summary_report(
        result,
        output_path=output_path,
        missing_price_keys_output_path=missing_price_keys_output_path,
        title=title,
        annualization=annualization,
        factor_figures=None,
    )


def _summary_report_with_factor_figures(
    result: FactorEvaluationResult,
    figures: tuple[ReportFigure, ...],
    *,
    output_path: str | Path | None = None,
    missing_price_keys_output_path: str | Path | None = None,
    title: str | None = None,
) -> str:
    """Render a factor report while reusing an already-built figure inventory."""

    return _summary_report(
        result,
        output_path=output_path,
        missing_price_keys_output_path=missing_price_keys_output_path,
        title=title,
        annualization=None,
        factor_figures=figures,
    )


def _summary_report(
    result: BacktestResult | FactorEvaluationResult,
    *,
    output_path: str | Path | None,
    missing_price_keys_output_path: str | Path | None,
    title: str | None,
    annualization: int | None,
    factor_figures: tuple[ReportFigure, ...] | None,
) -> str:
    """Shared report renderer with an internal precomputed-figure path."""

    if isinstance(result, FactorEvaluationResult):
        report_title = title or "Signal Evaluation Summary Report"
        body = _factor_report(
            result,
            annualization=annualization,
            figures=factor_figures,
        )
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


def _backtest_report(result: BacktestResult, *, annualization: int | None) -> str:
    tables = [
        _table_section("Performance", result.performance),
        _table_section("Trading Summary", _trading_summary(result)),
        _table_section("Price Availability", _price_availability_summary(result)),
    ]
    figures = [
        plot_cumulative_returns(result, title="Portfolio Cumulative Returns"),
        plot_drawdown(result),
        plot_turnover_and_costs(result),
        plot_rolling_sharpe(result, annualization=annualization),
        plot_rolling_volatility(result, annualization=annualization),
    ]
    return _section(
        "Tables", "".join(tables) + _figure_to_html(plot_coverage(result))
    ) + _figures_section("Plots", figures)


def _factor_report(
    result: FactorEvaluationResult,
    *,
    annualization: int | None,
    figures: tuple[ReportFigure, ...] | None = None,
) -> str:
    figures_by_section = _figures_by_section(
        factor_evaluation_report_figures(result, annualization=annualization)
        if figures is None
        else figures
    )
    sections = [
        _table_section("Summary", _factor_summary(result), none_display="N/A")
        + _figure_to_html(figures_by_section["Summary & Coverage"][0]),
        _factor_ic_section(figures_by_section["IC & ICIR"]),
        _factor_quantile_section(result, figures_by_section["Quantiles"]),
        _factor_top_n_section(result, figures_by_section["TOP N"]),
        _factor_benchmark_section(
            result, figures_by_section["TOP N vs Benchmarks"]
        ),
        _factor_spread_section(result, figures_by_section["Spread Performance"]),
    ]
    return "".join(sections)


def _figures_by_section(
    figures: tuple[ReportFigure, ...],
) -> dict[str, list[go.Figure]]:
    grouped: dict[str, list[go.Figure]] = {}
    for item in figures:
        grouped.setdefault(item.section, []).append(item.figure)
    return grouped


def _default_missing_price_keys_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_missing_price_keys.csv")


def _write_missing_price_keys_csv(
    missing_price_keys: pl.DataFrame,
    output_path: str | Path,
) -> None:
    missing_price_keys.write_csv(Path(output_path))


def _factor_ic_section(figures: list[go.Figure]) -> str:
    body = "".join(_figure_to_html(figure) for figure in figures)
    return _section("IC and ICIR", body)


def _factor_top_n_section(
    result: FactorEvaluationResult,
    figures: list[go.Figure],
) -> str:
    body = "".join(
        [
            _table_section(
                "TOP N Performance",
                _standard_performance_table(result.top_n_backtest),
            ),
            _table_section(
                "TOP N Price Availability",
                _price_availability_summary(result.top_n_backtest),
            ),
            _table_section("TOP N Lag Analysis", _lag_analysis(result, "top_n")),
            *(_figure_to_html(figure) for figure in figures),
        ]
    )
    return _section("TOP N", body)


def _factor_spread_section(
    result: FactorEvaluationResult,
    figures: list[go.Figure],
) -> str:
    tables: list[str] = []
    if result.spread_backtest is not None:
        tables.extend(
            [
                _table_section(
                    "Spread Performance",
                    _standard_performance_table(result.spread_backtest),
                ),
                _table_section(
                    "Spread Price Availability",
                    _price_availability_summary(result.spread_backtest),
                ),
            ]
        )
    return _section(
        "Spread Performance",
        "".join([*tables, *(_figure_to_html(figure) for figure in figures)]),
    )


def _factor_quantile_section(
    result: FactorEvaluationResult,
    figures: list[go.Figure],
) -> str:
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
            *(_figure_to_html(figure) for figure in figures),
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
    spread = result.spread_backtest.summary if result.spread_backtest else None
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


def _factor_benchmark_section(
    result: FactorEvaluationResult,
    figures: list[go.Figure],
) -> str:
    coverage = (
        result.benchmark_coverage.group_by("benchmark")
        .agg(
            pl.col("coverage_ratio").mean().alias("mean_coverage"),
            pl.col("coverage_ratio").min().alias("minimum_coverage"),
        )
        .sort("benchmark")
    )
    incomplete = coverage.filter(pl.col("minimum_coverage") < 1.0)
    warning = ""
    if incomplete.height:
        names = ", ".join(
            escape(str(value))
            for value in incomplete.get_column("benchmark").to_list()
        )
        warning = (
            "<p><strong>Coverage warning:</strong> available samples were "
            f"renormalized for {names}; index returns were not forward-filled.</p>"
        )
    return _section(
        "TOP N vs Benchmarks",
        "".join(
            [
                warning,
                _table_section(
                    "Benchmark Performance", result.benchmark_performance
                ),
                _table_section("Benchmark Coverage", coverage),
                *(_figure_to_html(figure) for figure in figures),
            ]
        ),
    )


def _price_availability_summary(result: BacktestResult) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"metric": "frozen_price_gap_rows", "value": result.price_gaps.height},
            {
                "metric": "unexecuted_target_weight_rows",
                "value": result.unexecuted_weight_keys.height,
            },
            {
                "metric": "raw_missing_price_key_rows",
                "value": result.missing_price_keys.height,
            },
            {
                "metric": "market_rule_blocked_trade_rows",
                "value": result.execution_blocks.height,
            },
        ],
        schema={"metric": pl.String, "value": pl.Int64},
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


signal_evaluation_report_figures = factor_evaluation_report_figures
