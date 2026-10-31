"""Bounded ingestion preserves the saved scalar formulas and historical work."""

import builtins
import copy
import weakref
from datetime import date, timedelta
from statistics import correlation

import polars as pl
import pytest

import bagelquant_bt.correlation as owner
from bagelquant_bt import pair_correlation, pair_correlation_batches

START = date(2020, 1, 1)


def returns(count, *, offset=0, right=False):
    return pl.DataFrame(
        {
            "time": [START + timedelta(days=offset + i) for i in range(count)],
            "value": [float((offset + i) % 4 if right else offset + i)
                      for i in range(count)],
        },
        schema={"time": pl.Date, "value": pl.Float64},
    )


def cross_section(offset, *, right=False):
    return pl.DataFrame(
        {"time": [START + timedelta(days=offset)] * 5,
         "asset_id": ["a", "b", "c", "d", "e"],
         "value": [3, 2, 2, 1, float("inf")] if right else [1, 1, 2, 3, None]},
        schema={"time": pl.Date, "asset_id": pl.String, "value": pl.Float64},
    )


@pytest.mark.parametrize("batch_size", [1, 4, 20])
def test_return_batches_equal_single_and_independent_cold_reference(batch_size):
    left, right = returns(20), returns(20, right=True)
    expected = pair_correlation(
        left, right, changed_dates=left["time"].to_list(), start=START,
        end=START + timedelta(days=19),
    )
    actual = pair_correlation_batches(
        ((left.slice(i, batch_size), right.slice(i, batch_size),
          left["time"].slice(i, batch_size).to_list())
         for i in range(0, 20, batch_size)),
        start=START, end=START + timedelta(days=19),
    )
    assert actual == expected
    assert actual["correlation"] == pytest.approx(
        correlation(left["value"].to_list(), right["value"].to_list())
    )


@pytest.mark.parametrize("method", ["pearson", "spearman"])
@pytest.mark.parametrize("zero_missing", [False, True])
def test_cross_section_batches_preserve_ties_finite_matching_and_repairs(
    method, zero_missing
):
    kwargs = {"kind": "cross_section", "method": method,
              "zero_missing": zero_missing, "start": START,
              "end": START + timedelta(days=4)}
    left = [cross_section(i) for i in range(5)]
    right = [cross_section(i, right=True) for i in range(5)]
    dates = [START + timedelta(days=i) for i in range(5)]
    cold = pair_correlation_batches(
        zip(left, right, [[day] for day in dates], strict=True), **kwargs
    )
    assert cold == pair_correlation(
        pl.concat(left), pl.concat(right), changed_dates=dates, **kwargs
    )
    if method == "spearman":
        assert cold["correlation"] == pytest.approx(-5 / 6)
    updated_right = right[1].with_columns(pl.col("value") * -1)
    expected = pair_correlation(
        pl.concat([left[0], left[1], left[3], left[4]]),
        pl.concat([right[0], updated_right, right[3], right[4]]),
        changed_dates=[dates[0], dates[1], dates[3], dates[4]], **kwargs,
    )
    repaired = pair_correlation_batches(
        [(left[1], updated_right, [dates[1]]),
         (left[2].head(0), right[2].head(0), [dates[2]])],
        previous=cold["state"], **kwargs,
    )
    assert repaired["processed_dates"] == 2
    assert {k: v for k, v in repaired.items() if k != "processed_dates"} == {
        k: v for k, v in expected.items() if k != "processed_dates"
    }


def test_return_batches_append_repair_delete_and_noop_keep_original_state():
    end = START + timedelta(days=19)
    left, right = returns(20), returns(20, right=True)
    dates = left["time"].to_list()
    first = pair_correlation(
        left.head(12), right.head(12), changed_dates=dates[:12], start=START, end=end,
    )
    original = copy.deepcopy(first["state"])
    replacement = right.slice(4, 1).with_columns(pl.lit(15.0).alias("value"))
    appended = pair_correlation_batches(
        [(left.slice(12, 4), right.slice(12, 4), dates[12:16]),
         (left.slice(16), right.slice(16), dates[16:]),
         (left.slice(4, 1), replacement, dates[4:5]),
         (left.head(0), right.head(0), dates[5:6])],
        previous=first["state"], start=dates[2], end=dates[18],
    )
    x = [float(i) for i in range(2, 19) if i != 5]
    y = [15.0 if i == 4 else float(i % 4) for i in range(2, 19) if i != 5]
    assert appended["correlation"] == pytest.approx(correlation(x, y))
    assert appended["processed_dates"] == 10
    assert first["state"] == original
    noop = pair_correlation_batches(
        iter(()), previous=appended["state"], start=dates[2], end=dates[18],
    )
    assert noop == {**appended, "processed_dates": 0}
    empty = pair_correlation_batches(iter(()), start=START, end=end)
    assert empty["correlation"] is None
    assert empty["state"]["days"] == []
    assert empty["processed_dates"] == 0


@pytest.mark.parametrize("history", [32, 256])
@pytest.mark.parametrize("batch_count", [1, 8])
def test_history_is_visited_once_and_finalized_once(history, batch_count, monkeypatch):
    left, right = returns(history), returns(history, right=True)
    prior = pair_correlation(
        left, right, changed_dates=left["time"].to_list(), start=START,
        end=START + timedelta(days=history + batch_count),
    )["state"]

    class MeasuredHistory(list):
        iterations = 0
        visits = 0

        def __iter__(self):
            self.iterations += 1
            for row in super().__iter__():
                self.visits += 1
                yield row

    rows = MeasuredHistory(prior["days"])
    prior = {**prior, "days": rows}
    finalizations, kernel_dates = [], []
    kernel = owner._update_pair_days

    def measured_kernel(days, left, right, **kwargs):
        kernel_dates.extend(left["time"].to_list())
        return kernel(days, left, right, **kwargs)

    def measured_sort(items):
        finalizations.append(len(items))
        return builtins.sorted(items)

    monkeypatch.setattr(owner, "_update_pair_days", measured_kernel)
    monkeypatch.setattr(owner, "sorted", measured_sort, raising=False)
    result = pair_correlation_batches(
        ((returns(1, offset=history + i), returns(1, offset=history + i, right=True),
          [START + timedelta(days=history + i)]) for i in range(batch_count)),
        previous=prior, start=START, end=START + timedelta(days=history + batch_count),
    )
    assert rows.iterations == 1
    assert rows.visits == history
    assert finalizations == [history + batch_count]
    assert kernel_dates == [START + timedelta(days=history + i)
                            for i in range(batch_count)]
    assert result["processed_dates"] == batch_count


def test_batches_release_consumed_panels_before_requesting_next_batch():
    refs = []

    def batches():
        for i in range(3):
            assert all(ref() is None for ref in refs)
            left, right = returns(1, offset=i), returns(1, offset=i, right=True)
            refs.extend((weakref.ref(left), weakref.ref(right)))
            yield left, right, [START + timedelta(days=i)]
            del left, right

    pair_correlation_batches(batches(), start=START, end=START + timedelta(days=3))
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("failure", ["duplicate", "invalid", "producer", "cancel"])
def test_failed_batches_preserve_previous_and_do_not_finalize(failure, monkeypatch):
    left, right = returns(12), returns(12, right=True)
    previous = pair_correlation(
        left, right, changed_dates=left["time"].to_list(), start=START,
        end=START + timedelta(days=20),
    )["state"]
    original = copy.deepcopy(previous)
    terminal = RuntimeError("producer or cancellation failure")
    canceled = False

    def batches():
        nonlocal canceled
        yield returns(1, offset=12), returns(1, offset=12), [START + timedelta(days=12)]
        if failure == "producer":
            raise terminal
        if failure == "cancel":
            canceled = True
            yield (
                returns(1, offset=13), returns(1, offset=13),
                [START + timedelta(days=13)],
            )
        elif failure == "duplicate":
            yield returns(0), returns(0), [START + timedelta(days=12)]
        else:
            yield returns(1), returns(1), [START + timedelta(days=13)]

    def check_canceled():
        if canceled:
            raise terminal

    def fail_finalize(*args, **kwargs):
        pytest.fail("a failed batch must not finalize a partial result")

    monkeypatch.setattr(owner, "_summarize_pair_days", fail_finalize)
    error_type = RuntimeError if failure in {"producer", "cancel"} else ValueError
    with pytest.raises(error_type) as caught:
        pair_correlation_batches(
            batches(), previous=previous, start=START,
            end=START + timedelta(days=20), check_canceled=check_canceled,
        )
    if error_type is RuntimeError:
        assert caught.value is terminal
    assert previous == original


def test_cancellation_precedes_lazy_producer_reads():
    consumed = []
    failure = RuntimeError("canceled")

    def batches():
        consumed.append(True)
        yield returns(0), returns(0), []

    def cancel():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        pair_correlation_batches(batches(), start=START, end=START,
                                 check_canceled=cancel)
    assert caught.value is failure
    assert consumed == []
