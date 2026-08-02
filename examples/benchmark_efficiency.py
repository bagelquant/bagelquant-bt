"""Isolated performance benchmarks for factor evaluation and portfolio paths.

Examples:

    uv run python examples/benchmark_efficiency.py --case dense-factor
    uv run python examples/benchmark_efficiency.py --case all --runs 3
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from bagelquant_bt import (
    BacktestConfig,
    PortfolioPathIdentity,
    materialize_portfolio_path,
    prepare_factor_market_data,
    resume_portfolio_path,
)
from bagelquant_bt.factor import run_factor_evaluation, top_n_equal_weights

CASES = (
    "dense-factor",
    "monthly-factor",
    "constrained-factor",
    "portfolio-path",
)


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    case: str
    data_seconds: float
    data_peak_rss_mb: float
    compute_seconds: float
    peak_rss_mb: float
    rows: int
    segments: int = 1
    single_seconds: float = 0.0
    segmented_seconds: float = 0.0
    materialization_seconds: float = 0.0
    materialized_peak_rss_mb: float = 0.0


def _peak_rss_mb() -> float:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not success:
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / (1024.0 * 1024.0)

    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _market(
    *,
    asset_count: int,
    session_count: int,
    signal_stride: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    assets = pl.DataFrame(
        {
            "asset_id": [f"a{index:04d}" for index in range(asset_count)],
            "_asset_index": range(asset_count),
        }
    )
    sessions = pl.DataFrame(
        {
            "time": pl.date_range(
                date(2010, 1, 1),
                date(2010, 1, 1) + timedelta(days=session_count - 1),
                eager=True,
            ),
            "_session_index": range(session_count),
        }
    )
    prices = (
        assets.join(sessions, how="cross")
        .with_columns(
            (
                100.0
                + pl.col("_asset_index") * 0.01
                + pl.col("_session_index") * 0.002
                + (
                    (pl.col("_asset_index") * 17 + pl.col("_session_index") * 11)
                    % 31
                )
                * 0.0005
            ).alias("price")
        )
        .select("time", "asset_id", "price")
    )
    factor = (
        assets.join(
            sessions.filter(pl.col("_session_index") % signal_stride == 0),
            how="cross",
        )
        .with_columns(
            (
                (
                    pl.col("_asset_index") * 13
                    + pl.col("_session_index") * 7
                )
                % 10_003
            )
            .cast(pl.Float64)
            .alias("factor")
        )
        .select("time", "asset_id", "factor")
    )
    return prices, factor


def _factor_case(
    *,
    case: str,
    asset_count: int,
    session_count: int,
    signal_stride: int,
    constrained: bool,
) -> BenchmarkMeasurement:
    data_started = time.perf_counter()
    prices, factor = _market(
        asset_count=asset_count,
        session_count=session_count,
        signal_stride=signal_stride,
    )
    market = prepare_factor_market_data(prices)
    availability = None
    if constrained:
        availability = (
            prices.with_row_index("_row")
            .filter(
                (pl.col("_row") % 997 == 0)
                & (pl.col("time") > prices.get_column("time").min())
            )
            .select("time", "asset_id")
            .with_columns(
                pl.lit(False).alias("can_buy"),
                pl.lit(True).alias("can_sell"),
                pl.lit("synthetic_limit").alias("reason"),
            )
        )
    data_seconds = time.perf_counter() - data_started
    data_peak_rss_mb = _peak_rss_mb()

    compute_started = time.perf_counter()
    result = run_factor_evaluation(
        factor,
        prices,
        config=BacktestConfig(
            initial_capital=1_000_000,
            quantiles=5,
            top_n=min(50, asset_count),
        ),
        market_data=market,
        execution_availability=availability,
    )
    compute_seconds = time.perf_counter() - compute_started
    compute_peak_rss = _peak_rss_mb()
    materialize_started = time.perf_counter()
    _ = (
        result.top_n_backtest.weights.height,
        result.top_n_backtest.target_weights.height,
        (
            0
            if result.spread_backtest is None
            else result.spread_backtest.weights.height
        ),
        (
            0
            if result.spread_backtest is None
            else result.spread_backtest.target_weights.height
        ),
    )
    materialization_seconds = time.perf_counter() - materialize_started
    return BenchmarkMeasurement(
        case=case,
        data_seconds=data_seconds,
        data_peak_rss_mb=data_peak_rss_mb,
        compute_seconds=compute_seconds,
        peak_rss_mb=compute_peak_rss,
        rows=result.factor.height,
        materialization_seconds=materialization_seconds,
        materialized_peak_rss_mb=_peak_rss_mb(),
    )


def _portfolio_path_case() -> BenchmarkMeasurement:
    data_started = time.perf_counter()
    prices, factor = _market(
        asset_count=450,
        session_count=2_520,
        signal_stride=20,
    )
    weights = top_n_equal_weights(factor, top_n=50)
    market = prepare_factor_market_data(prices)
    rebalance_times = weights.get_column("time").unique().sort().to_list()
    identity = PortfolioPathIdentity(
        alpha_revision="benchmark",
        universe="synthetic",
        policy_combo="top-50",
    )
    config = BacktestConfig(initial_capital=1_000_000, top_n=50)
    data_seconds = time.perf_counter() - data_started
    data_peak_rss_mb = _peak_rss_mb()

    compute_started = time.perf_counter()
    materialize_portfolio_path(
        weights,
        prices,
        identity=identity,
        config=config,
        prepared_forward_returns=market.forward_returns,
        prepared_price_gaps=market.price_data.price_gaps if market.price_data else None,
    )
    single_seconds = time.perf_counter() - compute_started

    segmented_started = time.perf_counter()
    checkpoint = None
    for index, start in enumerate(rebalance_times):
        end = (
            rebalance_times[index + 1]
            if index + 1 < len(rebalance_times)
            else prices.get_column("time").max()
        )
        segment_prices = prices.filter(
            (pl.col("time") >= start) & (pl.col("time") <= end)
        )
        segment_returns = market.forward_returns.filter(
            (pl.col("time") >= start) & (pl.col("time") < end)
        )
        segment_weights = weights.filter(pl.col("time") == start)
        if checkpoint is None:
            chunk = materialize_portfolio_path(
                segment_weights,
                segment_prices,
                identity=identity,
                config=config,
                prepared_forward_returns=segment_returns,
            )
        else:
            chunk = resume_portfolio_path(
                segment_weights,
                segment_prices,
                identity=identity,
                checkpoint=checkpoint,
                config=config,
                prepared_forward_returns=segment_returns,
            )
        checkpoint = chunk.checkpoint
    segmented_seconds = time.perf_counter() - segmented_started
    return BenchmarkMeasurement(
        case="portfolio-path",
        data_seconds=data_seconds,
        data_peak_rss_mb=data_peak_rss_mb,
        compute_seconds=single_seconds + segmented_seconds,
        peak_rss_mb=_peak_rss_mb(),
        rows=prices.height,
        segments=len(rebalance_times),
        single_seconds=single_seconds,
        segmented_seconds=segmented_seconds,
        materialized_peak_rss_mb=_peak_rss_mb(),
    )


def _run_child(case: str) -> BenchmarkMeasurement:
    if case == "dense-factor":
        return _factor_case(
            case=case,
            asset_count=250,
            session_count=300,
            signal_stride=1,
            constrained=False,
        )
    if case == "monthly-factor":
        return _factor_case(
            case=case,
            asset_count=2_000,
            session_count=2_520,
            signal_stride=20,
            constrained=False,
        )
    if case == "constrained-factor":
        return _factor_case(
            case=case,
            asset_count=2_000,
            session_count=2_520,
            signal_stride=20,
            constrained=True,
        )
    return _portfolio_path_case()


def _run_isolated(case: str) -> BenchmarkMeasurement:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        case,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return BenchmarkMeasurement(**json.loads(completed.stdout.strip().splitlines()[-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "all"), default="dense-factor")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--_child", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._child:
        print(json.dumps(asdict(_run_child(args._child)), sort_keys=True))
        return
    if args.runs <= 0:
        parser.error("--runs must be positive")

    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        measurements = [_run_isolated(case) for _ in range(args.runs)]
        summary = {
            "case": case,
            "runs": args.runs,
            "median_data_seconds": statistics.median(
                item.data_seconds for item in measurements
            ),
            "max_data_peak_rss_mb": max(
                item.data_peak_rss_mb for item in measurements
            ),
            "median_compute_seconds": statistics.median(
                item.compute_seconds for item in measurements
            ),
            "max_peak_rss_mb": max(
                item.peak_rss_mb for item in measurements
            ),
            "rows": measurements[0].rows,
            "segments": measurements[0].segments,
        }
        if case != "portfolio-path":
            summary["median_materialization_seconds"] = statistics.median(
                item.materialization_seconds for item in measurements
            )
            summary["max_materialized_peak_rss_mb"] = max(
                item.materialized_peak_rss_mb for item in measurements
            )
        if case == "portfolio-path":
            summary["median_single_seconds"] = statistics.median(
                item.single_seconds for item in measurements
            )
            summary["median_segmented_seconds"] = statistics.median(
                item.segmented_seconds for item in measurements
            )
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
