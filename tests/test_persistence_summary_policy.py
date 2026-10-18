"""Canonical persistence policy: exact values, typed boundaries and layout reuse."""

from __future__ import annotations

import itertools
import json
import math
import os
import struct
import subprocess
import sys
from datetime import date
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_bt import PERSISTENCE_SUMMARY_POLICY, summarize_signal_persistence

SCHEMA = {
    "horizon_sessions": pl.Int64,
    "mean_rank_autocorrelation": pl.Float64,
    "sample_size": pl.Int64,
    "signal_half_life_band": pl.String,
}


def _series(values, *, horizon=1):
    return pl.DataFrame(
        {"horizon_sessions": [horizon] * len(values), "rank_autocorrelation": values},
        schema={"horizon_sessions": pl.Int64, "rank_autocorrelation": pl.Float64},
    )


def _bits(value):
    return None if value is None else struct.pack(">d", value).hex()


def _signature(frame):
    assert frame.schema == SCHEMA
    return [
        (row[0], _bits(row[1]), row[2], row[3]) for row in frame.iter_rows()
    ]


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 4.0],
        [1e100, 1.0, -1e100],
        [sys.float_info.max, sys.float_info.max, -sys.float_info.max],
        [-sys.float_info.max, -sys.float_info.max, sys.float_info.max],
        [1.0, 2.0**-53, -1.0] * 17,
        [0.0, -0.0, -0.0],
        [1.0, -1.0],
        [-math.ulp(0.0), 0.0, 0.0],
        [math.ulp(0.0), 0.0, 0.0],
    ],
)
def test_finite_mean_matches_independent_exact_ratio_reference(values):
    expected = float(sum(map(Fraction.from_float, values), Fraction()) / len(values))
    output = summarize_signal_persistence(_series(values), [1])
    assert output.schema == SCHEMA
    assert _bits(output.item(0, "mean_rank_autocorrelation")) == _bits(expected)
    assert output.item(0, "sample_size") == len(values)


@pytest.mark.parametrize(
    "values,expected_bits,count",
    [
        ([None, None], None, 0),
        ([None, 0.25, 0.75], _bits(0.5), 2),
        ([math.inf, 0.25, None], _bits(math.inf), 2),
        ([-math.inf, 0.25], _bits(-math.inf), 2),
        ([math.inf, -math.inf, None], "7ff8000000000000", 2),
        ([struct.unpack(">d", bytes.fromhex("fff8000000000123"))[0], 0.25],
         "7ff8000000000000", 2),
        ([struct.unpack(">d", bytes.fromhex("7ff0000000000001"))[0], math.inf],
         "7ff8000000000000", 2),
    ],
)
def test_nulls_and_nonfinites_have_explicit_counts_and_order_independent_bits(
    values, expected_bits, count
):
    expected = None
    for permutation in itertools.permutations(values):
        output = summarize_signal_persistence(_series(list(permutation)), [1, 5])
        assert _bits(output.item(0, "mean_rank_autocorrelation")) == expected_bits
        assert output.item(0, "sample_size") == count
        assert output.item(1, "sample_size") == 0
        observed = _signature(output)
        if expected is None:
            expected = observed
        assert observed == expected


def test_grid_duplicates_missing_horizons_and_observations_keep_original_weights():
    series = pl.DataFrame(
        {
            "horizon_sessions": [1, 1, 5, 5, 5, 60, None],
            "rank_autocorrelation": [0.75, 0.75, 0.25, 0.75, None, None, -1.0],
            # Daily asset counts must not become observation weights.
            "sample_size": [1, 100, 1, 10_000, 200, 1, 1],
        },
        schema={
            "horizon_sessions": pl.Int64,
            "rank_autocorrelation": pl.Float64,
            "sample_size": pl.UInt32,
        },
    )
    result = summarize_signal_persistence(series, [120, 5, 1, 5, 60])
    assert result.schema == SCHEMA
    assert result["horizon_sessions"].to_list() == [1, 5, 5, 60, 120]
    assert result["mean_rank_autocorrelation"].to_list() == [0.75, 0.5, 0.5, None, None]
    assert result["sample_size"].to_list() == [2, 2, 2, 0, 0]
    assert result["signal_half_life_band"].to_list() == ["1-5D"] * 5


@pytest.mark.parametrize(
    "rho,expected",
    [(math.nextafter(0.5, -math.inf), "0-1D"), (0.5, "0-1D"),
     (math.nextafter(0.5, math.inf), ">120D"), (math.inf, ">120D"),
     (-math.inf, "0-1D"), (math.nan, ">120D")],
)
def test_band_keeps_original_threshold_and_nonfinite_comparison(rho, expected):
    result = summarize_signal_persistence(_series([rho]), [1])
    assert result.item(0, "signal_half_life_band") == expected


def test_band_retains_unavailable_for_missing_last_requested_lag():
    result = summarize_signal_persistence(_series([0.75]), [1, 120])
    assert result["signal_half_life_band"].to_list() == ["unavailable"] * 2


@pytest.mark.parametrize(
    "empty",
    [pl.DataFrame(), _series([]),
     _series([]).with_columns(pl.col("rank_autocorrelation").cast(pl.String)),
     _series([]).with_columns(pl.col("rank_autocorrelation").cast(pl.Float32))],
)
def test_original_empty_frame_bypass_and_empty_requests_are_typed(empty):
    assert _signature(summarize_signal_persistence(empty, [5, 1])) == [
        (1, None, 0, "unavailable"), (5, None, 0, "unavailable")
    ]
    assert _signature(summarize_signal_persistence(empty, [])) == []
    assert _signature(summarize_signal_persistence(_series([0.5]), [])) == []


def test_requested_integer_coercion_is_unchanged():
    assert summarize_signal_persistence(_series([0.5]), ["1", 5.9])[
        "horizon_sessions"
    ].to_list() == [1, 5]


@pytest.mark.parametrize("missing", ["horizon_sessions", "rank_autocorrelation"])
def test_nonempty_missing_columns_remain_rejected(missing):
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        summarize_signal_persistence(_series([0.5]).drop(missing), [1])


def test_incompatible_horizon_join_is_not_silently_coerced():
    with pytest.raises(pl.exceptions.SchemaError):
        summarize_signal_persistence(
            _series([0.5]).with_columns(pl.col("horizon_sessions").cast(pl.String)),
            [1],
        )


@pytest.mark.parametrize(
    "label,dtype,values",
    [
        ("float64", pl.Float64, [0.25, 0.75]),
        ("float32", pl.Float32, [0.25, 0.75]),
        ("int64", pl.Int64, [0, 1]),
        ("uint32", pl.UInt32, [0, 1]),
        ("boolean", pl.Boolean, [False, True]),
        ("string", pl.String, ["0.25", "0.75"]),
        ("null", pl.Null, [None, None]),
        ("decimal", pl.Decimal(20, 2), [Decimal("0.25"), Decimal("0.75")]),
        ("date", pl.Date, [date(2024, 1, 1), date(2024, 1, 3)]),
    ],
)
def test_original_observed_dtype_admission_schema_counts_grid_and_band(
    label, dtype, values
):
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "legacy_persistence_admission.json").read_text())
    expected = next(case for case in fixture["cases"] if case["case"] == label)
    frame = pl.DataFrame(
        {"horizon_sessions": [1, 1], "rank_autocorrelation": values},
        schema={"horizon_sessions": pl.Int64, "rank_autocorrelation": dtype},
    )
    assert str(frame.schema) == expected["input_schema"]
    if "error_type" in expected["original"]:
        assert expected["original"]["error_type"] == "TypeError"
        with pytest.raises(TypeError):
            summarize_signal_persistence(frame, [1, 5])
        return
    result = summarize_signal_persistence(frame, [1, 5])
    assert str(result.schema) == expected["original"]["schema"]
    assert [[_bits(value) if isinstance(value, float) else repr(value)
             for value in row] for row in result.iter_rows()] == (
        expected["original"]["rows"]
    )


@pytest.mark.parametrize(
    "dtype,values",
    [
        (pl.Int64, [2**63 - 1, -(2**63), 1, None]),
        (pl.UInt64, [2**64 - 1, 2**64 - 3, 0, None]),
        (pl.Int32, [2**31 - 1, -(2**31), 1, None]),
        (pl.Boolean, [True, False, True, None]),
        (pl.Decimal(38, 1), [Decimal("10000000000000000000000000001.0"),
                            Decimal("-10000000000000000000000000000.0"),
                            Decimal("0.0"), None]),
    ],
)
def test_numeric_means_preserve_exact_input_before_final_float64(dtype, values):
    finite = [value for value in values if value is not None]
    expected = float(sum(map(Fraction, finite), Fraction()) / len(finite))
    frame = pl.DataFrame(
        {"horizon_sessions": [1] * len(values), "rank_autocorrelation": values},
        schema={"horizon_sessions": pl.Int64, "rank_autocorrelation": dtype},
    )
    for precision in (1, 50):
        with localcontext() as context:
            context.prec = precision
            for layout in (frame, frame.reverse(), frame.rechunk()):
                result = summarize_signal_persistence(layout, [1, 5])
                assert result.schema == SCHEMA
                assert _bits(result.item(0, "mean_rank_autocorrelation")) == (
                    _bits(expected)
                )
                assert result["sample_size"].to_list() == [len(finite), 0]


@pytest.mark.parametrize(
    "values",
    [[1.0, 2**-24, -1.0], [0.0, -0.0], [-2**-149, 0.0, 0.0]],
)
def test_float32_keeps_established_dtype_and_canonical_final_cast(values):
    frame = pl.DataFrame(
        {"horizon_sessions": [1] * len(values), "rank_autocorrelation": values},
        schema={"horizon_sessions": pl.Int64, "rank_autocorrelation": pl.Float32},
    )
    represented = frame["rank_autocorrelation"].to_list()
    exact64 = float(sum(map(Fraction.from_float, represented), Fraction()) /
                    len(represented))
    expected = struct.pack(">f", exact64)
    for layout in (frame, frame.reverse(),
                   pl.concat([frame.head(1), frame.slice(1)], rechunk=False)):
        result = summarize_signal_persistence(layout, [1, 5])
        assert result.schema == {**SCHEMA, "mean_rank_autocorrelation": pl.Float32}
        assert struct.pack(">f", result.item(0, "mean_rank_autocorrelation")) == (
            expected
        )
        assert result["sample_size"].to_list() == [len(values), 0]


@pytest.mark.parametrize(
    "values,bits",
    [([math.nan, 0.25], "7fc00000"),
     ([math.inf, -math.inf], "7fc00000"),
     ([math.inf, 0.25], "7f800000"),
     ([-math.inf, 0.25], "ff800000")],
)
def test_float32_nonfinite_policy_preserves_established_output_type(values, bits):
    frame = pl.DataFrame(
        {"horizon_sessions": [1, 1], "rank_autocorrelation": values},
        schema={"horizon_sessions": pl.Int64, "rank_autocorrelation": pl.Float32},
    )
    for layout in (frame, frame.reverse()):
        output = summarize_signal_persistence(layout, [1])
        assert output.schema == {**SCHEMA, "mean_rank_autocorrelation": pl.Float32}
        assert struct.pack(">f", output.item(0, "mean_rank_autocorrelation")).hex() == (
            bits
        )
        assert output.item(0, "sample_size") == 2


def test_string_and_null_counts_stay_unavailable_without_numeric_coercion():
    for dtype, values, count in (
        (pl.String, ["0.25", "nan", None], 2),
        (pl.Null, [None, None, None], 0),
    ):
        frame = pl.DataFrame(
            {"horizon_sessions": [1, 1, 1], "rank_autocorrelation": values},
            schema={"horizon_sessions": pl.Int64, "rank_autocorrelation": dtype},
        )
        output = summarize_signal_persistence(frame, [5, 1, 1])
        assert output.schema == {**SCHEMA, "mean_rank_autocorrelation": dtype}
        assert output["mean_rank_autocorrelation"].to_list() == [None, None, None]
        assert output["sample_size"].to_list() == [count, count, 0]
        assert output["signal_half_life_band"].to_list() == ["unavailable"] * 3


@pytest.mark.parametrize(
    "values,horizons,counts",
    [([None, None], [1, 5], [0, 0]),
     ([date(2024, 1, 1), date(2024, 1, 3)], [], []),
     ([date(2024, 1, 1), date(2024, 1, 3)], [5], [0])],
)
def test_temporal_null_empty_and_unrequested_groups_keep_unavailable_visibility(
    values, horizons, counts
):
    frame = pl.DataFrame(
        {"horizon_sessions": [1, 1], "rank_autocorrelation": values},
        schema={"horizon_sessions": pl.Int64, "rank_autocorrelation": pl.Date},
    )
    # This is the old public Polars output-type contract, with no data reduction.
    expected_type = frame.lazy().select(
        pl.col("rank_autocorrelation").mean()
    ).collect_schema()["rank_autocorrelation"]
    result = summarize_signal_persistence(frame, horizons)
    assert result.schema == {**SCHEMA, "mean_rank_autocorrelation": expected_type}
    assert result["mean_rank_autocorrelation"].to_list() == [None] * len(horizons)
    assert result["sample_size"].to_list() == counts
    assert result["signal_half_life_band"].to_list() == ["unavailable"] * len(horizons)


def test_temporal_error_does_not_preempt_original_horizon_join_error():
    frame = pl.DataFrame(
        {"horizon_sessions": ["1"], "rank_autocorrelation": [date(2024, 1, 1)]},
        schema={"horizon_sessions": pl.String, "rank_autocorrelation": pl.Date},
    )
    with pytest.raises(pl.exceptions.SchemaError):
        summarize_signal_persistence(frame, [1])


def test_row_chunk_and_split_layouts_preserve_every_output_bit():
    rows = [(h, value) for h in (1, 5, 120) for value in [1.0, 2**-53, -1.0] * 37]
    frame = pl.DataFrame(
        rows,
        schema={"horizon_sessions": pl.Int64, "rank_autocorrelation": pl.Float64},
        orient="row",
    )
    expected = _signature(summarize_signal_persistence(frame, [120, 1, 5]))
    layouts = (
        frame.reverse(),
        frame.sample(fraction=1.0, shuffle=True, seed=37),
        frame.rechunk(),
        pl.concat([frame.head(111), frame.slice(111)], rechunk=False),
        pl.concat([frame.slice(i, 1) for i in range(frame.height)], rechunk=False),
    )
    for layout in layouts:
        assert _signature(summarize_signal_persistence(layout, [120, 1, 5])) == expected


def test_fresh_native_one_and_two_have_identical_full_typed_outputs(tmp_path):
    # These are later test-owned synthetic children, never real-root workers.
    program = """
import json, struct, sys
import polars as pl
from bagelquant_bt import PERSISTENCE_SUMMARY_POLICY, summarize_signal_persistence
rows=[(h,v) for h,values in (
 (1,[1.0,2.0**-53,-1.0]*401),
 (5,[sys.float_info.max,sys.float_info.max,-sys.float_info.max]),
 (120,[float('inf'),-float('inf'),None])) for v in values]
frame=pl.DataFrame(rows,orient='row',
 schema={'horizon_sessions':pl.Int64,'rank_autocorrelation':pl.Float64})
out=summarize_signal_persistence(frame,[120,5,1])
print(json.dumps({'policy':PERSISTENCE_SUMMARY_POLICY,
 'schema':[(k,str(v)) for k,v in out.schema.items()],
 'rows':[(h,None if v is None else struct.pack('>d',v).hex(),n,b)
         for h,v,n,b in out.iter_rows()],
 'threads':pl.thread_pool_size()}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1] / "src"), *sys.path]
    )
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    outputs = []
    for threads in (1, 2):
        environment["POLARS_MAX_THREADS"] = str(threads)
        run = subprocess.run(
            [sys.executable, "-B", "-c", program], cwd=tmp_path, env=environment,
            check=True, capture_output=True, text=True, timeout=20,
        )
        output = json.loads(run.stdout)
        assert output.pop("threads") == threads
        outputs.append(output)
    assert outputs[0] == outputs[1]
    assert outputs[0]["policy"] == PERSISTENCE_SUMMARY_POLICY


def test_default_grid_uses_same_authoritative_function():
    from bagelquant_bt.horizon import SIGNAL_PERSISTENCE_HORIZONS

    result = summarize_signal_persistence(_series([0.5]), SIGNAL_PERSISTENCE_HORIZONS)
    assert_frame_equal(
        result,
        summarize_signal_persistence(_series([0.5]).rechunk(),
                                     SIGNAL_PERSISTENCE_HORIZONS),
        check_exact=True,
    )
