from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from bagelquant_bt import (
    benjamini_hochberg,
    centered_rank_book_weights,
    gross_one_tail_weights,
    hac_mean_test,
    non_overlapping_cohort_statistics,
    quantile_curve_structure,
    rolling_window_information_coefficients,
    window_book_returns,
    window_factor_returns,
    window_quantile_forward_returns,
    window_tail_returns,
)


def _factor(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "evaluation_date": [date(2024, 1, 1)] * len(values),
            "execution_date": [date(2024, 1, 2)] * len(values),
            "asset_id": [f"a{index}" for index in range(len(values))],
            "factor": values,
        }
    )


def test_centered_rank_book_weights_match_hand_calculation() -> None:
    weights = centered_rank_book_weights(_factor([1.0, 2.0, 3.0, 4.0]))

    assert weights.get_column("book_weight").to_list() == pytest.approx(
        [-0.375, -0.125, 0.125, 0.375]
    )
    assert weights.get_column("book_weight").sum() == pytest.approx(0.0)
    assert weights.filter(pl.col("book_weight") > 0).get_column(
        "book_weight"
    ).sum() == pytest.approx(0.5)
    assert weights.filter(pl.col("book_weight") < 0).get_column(
        "book_weight"
    ).sum() == pytest.approx(-0.5)
    assert weights.get_column("book_weight").abs().sum() == pytest.approx(1.0)


def test_centered_rank_book_uses_average_ties_and_rejects_constants() -> None:
    tied = centered_rank_book_weights(_factor([1.0, 2.0, 2.0, 3.0, 4.0]))
    tied_weights = tied.get_column("book_weight").to_list()

    assert tied_weights[1] == pytest.approx(tied_weights[2])
    assert sum(tied_weights) == pytest.approx(0.0)
    assert sum(abs(value) for value in tied_weights) == pytest.approx(1.0)

    constant = centered_rank_book_weights(_factor([2.0, 2.0, 2.0]))
    assert constant.get_column("book_weight").null_count() == 3
    assert constant.get_column("unavailable_reason").null_count() == 0


def test_centered_rank_book_handles_odd_and_sparse_dynamic_cross_sections() -> None:
    odd = centered_rank_book_weights(_factor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert odd.filter(pl.col("asset_id") == "a2").get_column(
        "book_weight"
    ).item() == pytest.approx(0.0)
    assert odd.get_column("book_weight").abs().sum() == pytest.approx(1.0)

    sparse = _factor([1.0, 2.0, float("nan"), 4.0, 5.0]).filter(
        pl.col("asset_id") != "a1"
    )
    weights = centered_rank_book_weights(sparse)
    assert weights.get_column("asset_id").to_list() == ["a0", "a3", "a4"]
    assert weights.get_column("book_weight").abs().sum() == pytest.approx(1.0)


def test_gross_one_tail_weights_use_half_books_and_zero_middle() -> None:
    weights = gross_one_tail_weights(_factor([float(value) for value in range(10)]))

    assert weights.get_column("tail_weight").abs().sum() == pytest.approx(1.0)
    assert weights.filter(pl.col("tail_weight") > 0).get_column(
        "tail_weight"
    ).sum() == pytest.approx(0.5)
    assert weights.filter(pl.col("tail_weight") < 0).get_column(
        "tail_weight"
    ).sum() == pytest.approx(-0.5)
    assert weights.filter(pl.col("tail_weight") == 0.0).height == 8


def test_book_return_never_reweights_around_missing_forward_labels() -> None:
    factor = _factor([1.0, 2.0, 3.0, 4.0])
    weights = centered_rank_book_weights(factor)
    forward = pl.DataFrame(
        {
            "evaluation_date": [date(2024, 1, 1)] * 4,
            "execution_date": [date(2024, 1, 2)] * 4,
            "target_end_date": [date(2024, 1, 3)] * 4,
            "asset_id": ["a0", "a1", "a2", "a3"],
            "window_kind": ["cumulative"] * 4,
            "window_id": ["cumulative_1d"] * 4,
            "start_session": [1] * 4,
            "end_session": [1] * 4,
            "start_offset": [0] * 4,
            "end_offset": [1] * 4,
            "forward_return": [-0.03, -0.01, 0.01, 0.03],
        }
    )

    complete = window_book_returns(weights, forward).row(0, named=True)
    assert complete["book_return"] == pytest.approx(0.025)
    assert complete["coverage_ratio"] == pytest.approx(1.0)

    missing = window_book_returns(
        weights,
        forward.with_columns(
            pl.when(pl.col("asset_id") == "a3")
            .then(None)
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        ),
    ).row(0, named=True)
    assert missing["book_return"] == pytest.approx(0.01375)
    assert missing["expected_count"] == 4
    assert missing["observed_count"] == 3
    assert missing["coverage_ratio"] == pytest.approx(0.75)
    assert missing["unavailable_reason"] is None


def test_tail_return_keeps_original_weights_when_one_label_is_missing() -> None:
    factor = _factor([float(value) for value in range(10)])
    weights = gross_one_tail_weights(factor)
    forward = pl.DataFrame(
        {
            "evaluation_date": [date(2024, 1, 1)] * 10,
            "execution_date": [date(2024, 1, 2)] * 10,
            "target_end_date": [date(2024, 1, 3)] * 10,
            "asset_id": [f"a{index}" for index in range(10)],
            "window_kind": ["cumulative"] * 10,
            "window_id": ["cumulative_1d"] * 10,
            "start_session": [1] * 10,
            "end_session": [1] * 10,
            "start_offset": [0] * 10,
            "end_offset": [1] * 10,
            "forward_return": [-0.02, *([0.0] * 8), None],
        }
    )

    result = window_tail_returns(weights, forward).row(0, named=True)

    assert result["tail_return"] == pytest.approx(0.01)
    assert result["expected_count"] == 2
    assert result["observed_count"] == 1
    assert result["coverage_ratio"] == pytest.approx(0.5)
    assert result["unavailable_reason"] is None


def test_book_tail_and_quantiles_reveal_different_cross_section_structure() -> None:
    factor = _factor([float(value) for value in range(10)])
    metadata = {
        "evaluation_date": [date(2024, 1, 1)] * 10,
        "execution_date": [date(2024, 1, 2)] * 10,
        "target_end_date": [date(2024, 1, 3)] * 10,
        "asset_id": [f"a{index}" for index in range(10)],
        "window_kind": ["cumulative"] * 10,
        "window_id": ["cumulative_1d"] * 10,
        "start_session": [1] * 10,
        "end_session": [1] * 10,
        "start_offset": [0] * 10,
        "end_offset": [1] * 10,
    }
    book_weights = centered_rank_book_weights(factor)
    tail_weights = gross_one_tail_weights(factor)

    middle_only = pl.DataFrame(
        {
            **metadata,
            "forward_return": [0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
        }
    )
    assert (
        window_book_returns(book_weights, middle_only).get_column("book_return").item()
        > 0.0
    )
    assert window_tail_returns(tail_weights, middle_only).get_column(
        "tail_return"
    ).item() == pytest.approx(0.0)

    extreme_only = pl.DataFrame(
        {**metadata, "forward_return": [-0.1] + [0.0] * 8 + [0.1]}
    )
    book = (
        window_book_returns(book_weights, extreme_only).get_column("book_return").item()
    )
    tail = (
        window_tail_returns(tail_weights, extreme_only).get_column("tail_return").item()
    )
    assert 0.0 < book < tail

    u_shape = pl.DataFrame({**metadata, "forward_return": [0.1] + [0.0] * 8 + [0.1]})
    assert window_book_returns(book_weights, u_shape).get_column(
        "book_return"
    ).item() == pytest.approx(0.0)
    quantiles = window_quantile_forward_returns(factor, u_shape)
    endpoints = quantiles.filter(pl.col("quantile").is_in(["q1", "q10"]))
    assert endpoints.get_column("quantile_return").to_list() == pytest.approx(
        [0.1, 0.1]
    )

    flat = window_quantile_forward_returns(
        factor,
        pl.DataFrame({**metadata, "forward_return": [0.0] * 10}),
    )
    flat_structure = quantile_curve_structure(flat)
    assert flat_structure.get_column("quantile_rank_ic").item() is None
    assert flat_structure.get_column(
        "quantile_linearity_slope"
    ).item() == pytest.approx(0.0)
    assert flat_structure.get_column("quantile_linearity_r_squared").item() is None
    assert flat_structure.get_column("unavailable_reason").item() == (
        "quantile-rank IC requires non-constant returns"
    )


def test_quantile_structure_reports_positive_signal_order_linearity() -> None:
    factor = _factor([float(value) for value in range(10)])
    metadata = {
        "evaluation_date": [date(2024, 1, 1)] * 10,
        "execution_date": [date(2024, 1, 2)] * 10,
        "target_end_date": [date(2024, 1, 3)] * 10,
        "asset_id": [f"a{index}" for index in range(10)],
        "window_kind": ["cumulative"] * 10,
        "window_id": ["cumulative_1d"] * 10,
        "start_session": [1] * 10,
        "end_session": [1] * 10,
        "start_offset": [0] * 10,
        "end_offset": [1] * 10,
        "forward_return": [float(value) / 100.0 for value in range(10)],
    }

    structure = quantile_curve_structure(
        window_quantile_forward_returns(factor, pl.DataFrame(metadata))
    ).row(0, named=True)

    assert structure["quantile_rank_ic"] == pytest.approx(1.0)
    assert structure["monotonicity"] == pytest.approx(1.0)
    assert structure["quantile_linearity_slope"] > 0.0
    assert structure["quantile_linearity_r_squared"] == pytest.approx(1.0)

    reversed_curve = quantile_curve_structure(
        window_quantile_forward_returns(
            factor,
            pl.DataFrame(
                {
                    **metadata,
                    "forward_return": list(reversed(metadata["forward_return"])),
                }
            ),
        )
    ).row(0, named=True)
    assert reversed_curve["quantile_linearity_slope"] < 0.0
    assert reversed_curve["quantile_linearity_r_squared"] == pytest.approx(1.0)

    incomplete = quantile_curve_structure(
        window_quantile_forward_returns(factor, pl.DataFrame(metadata)).filter(
            pl.col("quantile") != "q10"
        )
    ).row(0, named=True)
    assert incomplete["quantile_linearity_slope"] is None
    assert incomplete["quantile_linearity_r_squared"] is None
    assert incomplete["unavailable_reason"] == "complete q1-to-q10 curve required"


def test_standardized_factor_return_is_per_cross_sectional_standard_deviation() -> None:
    factor = _factor([1.0, 2.0, 3.0, 4.0])
    metadata = {
        "evaluation_date": [date(2024, 1, 1)] * 4,
        "execution_date": [date(2024, 1, 2)] * 4,
        "target_end_date": [date(2024, 1, 3)] * 4,
        "asset_id": [f"a{index}" for index in range(4)],
        "window_kind": ["cumulative"] * 4,
        "window_id": ["cumulative_1d"] * 4,
        "start_session": [1] * 4,
        "end_session": [1] * 4,
        "start_offset": [0] * 4,
        "end_offset": [1] * 4,
        "forward_return": [0.01, 0.02, 0.03, 0.04],
    }
    forward = pl.DataFrame(metadata)

    standardized = (
        window_factor_returns(
            factor,
            forward,
            standardization="cross_sectional_zscore",
        )
        .get_column("factor_return")
        .item()
    )
    shifted_scaled = (
        window_factor_returns(
            factor.with_columns((pl.col("factor") * 7.0 + 100.0).alias("factor")),
            forward,
            standardization="cross_sectional_zscore",
        )
        .get_column("factor_return")
        .item()
    )

    assert standardized == pytest.approx(0.01 * (5.0**0.5) / 2.0)
    assert shifted_scaled == pytest.approx(standardized)

    constant = window_factor_returns(
        factor.with_columns(pl.lit(1.0).alias("factor")),
        forward,
        standardization="cross_sectional_zscore",
    )
    too_sparse = window_factor_returns(
        factor.head(2),
        forward,
        standardization="cross_sectional_zscore",
    )
    three_finite = window_factor_returns(
        factor.with_columns(
            pl.when(pl.col("asset_id") == "a3")
            .then(float("nan"))
            .otherwise(pl.col("factor"))
            .alias("factor")
        ),
        forward,
        standardization="cross_sectional_zscore",
    )
    assert constant.get_column("factor_return").item() is None
    assert too_sparse.get_column("factor_return").item() is None
    assert three_finite.get_column("factor_return").item() is not None


def test_hac_bh_and_all_staggered_cohorts_are_deterministic() -> None:
    values = [0.01, 0.02, -0.01, 0.03, 0.02, 0.04]
    result = hac_mean_test(values, window_width=2)

    assert result.sample_size == 6
    assert result.lag == 2
    assert result.mean == pytest.approx(sum(values) / len(values))
    assert result.standard_error is not None
    assert result.confidence_low < result.mean < result.confidence_high

    q_values = benjamini_hochberg([0.01, 0.04, 0.03, None])
    assert q_values == pytest.approx([0.03, 0.04, 0.04, None])

    dates = [date(2024, 1, 1) + timedelta(days=value) for value in range(6)]
    cohorts = non_overlapping_cohort_statistics(dates, values, window_width=2)
    assert cohorts["cohort_count"] == 2
    assert cohorts["cohort_same_sign_ratio"] == pytest.approx(1.0)


def test_rolling_ic_uses_240_valid_causal_observations() -> None:
    sessions = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(242)]
    pearson: list[float | None] = [float(index) for index in range(242)]
    pearson[100] = None
    ic = pl.DataFrame(
        {
            "evaluation_date": sessions,
            "window_kind": ["cumulative"] * len(sessions),
            "window_id": ["cumulative_1d"] * len(sessions),
            "start_session": [1] * len(sessions),
            "end_session": [1] * len(sessions),
            "pearson_ic": pearson,
            "spearman_ic": [0.25] * len(sessions),
        }
    )

    rolling = rolling_window_information_coefficients(ic)
    pearson_rolling = rolling.filter(pl.col("method") == "pearson").sort(
        "evaluation_date"
    )

    non_null = pearson_rolling.filter(pl.col("rolling_ic").is_not_null())
    assert non_null.height == 2
    first = non_null.row(0, named=True)
    assert first["evaluation_date"] == sessions[240]
    expected = [float(index) for index in range(241) if index != 100]
    assert first["rolling_ic"] == pytest.approx(sum(expected) / 240.0)
    assert first["rolling_observations"] == 240
