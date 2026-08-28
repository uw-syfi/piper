import math
import statistics

import pytest

from src.profile_statistics import (
    predicted_reduction,
    prediction_error,
    student_t_critical_95,
    summarize_mean_samples,
    summarize_paired_reductions,
    summarize_paired_savings,
    summarize_timing_samples,
)


def test_timing_summary_is_deterministic_and_reports_relative_ci_error() -> None:
    samples = [8.0, 9.0, 10.0, 11.0, 12.0]
    first = summarize_timing_samples(samples, bootstrap_repetitions=500, seed=7)
    second = summarize_timing_samples(samples, bootstrap_repetitions=500, seed=7)

    assert first == second
    assert first.samples == 5
    assert first.mean == pytest.approx(10.0)
    assert first.median == pytest.approx(10.0)
    assert first.std == pytest.approx(math.sqrt(2.5))
    assert first.ci95_low <= first.estimate <= first.ci95_high
    assert first.error_pct == pytest.approx(
        (first.ci95_high - first.ci95_low) / 2.0 / first.estimate * 100.0
    )


def test_prediction_and_reduction_percentages_have_explicit_semantics() -> None:
    error = prediction_error(90.0, 100.0)
    assert error.signed_pct == pytest.approx(-10.0)
    assert error.absolute_pct == pytest.approx(10.0)
    assert predicted_reduction(75.0, 100.0) == pytest.approx(25.0)


def test_paired_reduction_bootstraps_after_computing_each_pair() -> None:
    reductions, summary = summarize_paired_reductions(
        (10.0, 100.0, 1000.0),
        (9.0, 50.0, 990.0),
        bootstrap_repetitions=2_000,
        seed=7,
    )

    # Per-pair savings are 10%, 50%, and 1%, whose median is 10%.  Computing
    # the reduction from the two separate medians would incorrectly give 50%.
    assert reductions == pytest.approx((10.0, 50.0, 1.0))
    assert summary.estimator == "paired_median"
    assert summary.estimate == pytest.approx(10.0)
    assert summary.ci95_low <= summary.estimate <= summary.ci95_high


def test_paired_reduction_requires_aligned_sample_counts() -> None:
    with pytest.raises(ValueError, match="aligned pairs"):
        summarize_paired_reductions((10.0, 20.0), (9.0,))


def test_mean_summary_uses_sample_sd_and_student_t_interval() -> None:
    summary = summarize_mean_samples((8.0, 9.0, 10.0, 11.0, 12.0))

    assert student_t_critical_95(4) == pytest.approx(2.7764451052)
    assert summary.estimator == "mean"
    assert summary.estimate == pytest.approx(10.0)
    assert summary.mean == pytest.approx(10.0)
    assert summary.median == pytest.approx(10.0)
    assert summary.std == pytest.approx(math.sqrt(2.5))
    assert summary.ci95_low == pytest.approx(8.0367568385)
    assert summary.ci95_high == pytest.approx(11.9632431615)


def test_paired_savings_uses_ratio_of_means_and_covariance() -> None:
    summary = summarize_paired_savings(
        (10.0, 20.0, 30.0),
        (8.0, 15.0, 27.0),
    )

    assert summary.samples == 3
    assert summary.absolute_mean == pytest.approx(10.0 / 3.0)
    assert summary.percent == pytest.approx(100.0 / 6.0)
    assert summary.percent_standard_error == pytest.approx(5.3575837561)
    assert summary.percent_ci95_low == pytest.approx(-6.3851557064)
    assert summary.percent_ci95_high == pytest.approx(39.7184890397)


def test_paired_savings_does_not_average_per_iteration_percentages() -> None:
    normal = (10.0, 100.0, 1000.0)
    isolated = (9.0, 50.0, 990.0)
    summary = summarize_paired_savings(normal, isolated)
    per_iteration_percentages = tuple(
        100.0 * (normal_value - isolated_value) / normal_value
        for normal_value, isolated_value in zip(
            normal, isolated
        )
    )

    assert summary.percent == pytest.approx(5.4954954955)
    assert summary.percent != pytest.approx(statistics.fmean(per_iteration_percentages))
    assert summary.percent != pytest.approx(
        statistics.median(per_iteration_percentages)
    )


def test_mean_and_paired_savings_single_sample_have_no_interval() -> None:
    endpoint = summarize_mean_samples((10.0,))
    savings = summarize_paired_savings((10.0,), (8.0,))

    assert endpoint.std is None
    assert endpoint.ci95_low is None
    assert endpoint.ci95_high is None
    assert endpoint.median == pytest.approx(10.0)
    assert savings.absolute_mean == pytest.approx(2.0)
    assert savings.percent == pytest.approx(20.0)
    assert savings.percent_standard_error is None
    assert savings.percent_ci95_low is None
    assert savings.percent_ci95_high is None


def test_mean_and_paired_savings_handle_degenerate_and_negative_data() -> None:
    endpoint = summarize_mean_samples((10.0, 10.0, 10.0))
    negative = summarize_paired_savings(
        (10.0, 20.0, 30.0),
        (12.0, 21.0, 33.0),
    )

    assert endpoint.std == pytest.approx(0.0)
    assert endpoint.ci95_low == pytest.approx(10.0)
    assert endpoint.ci95_high == pytest.approx(10.0)
    assert negative.absolute_mean == pytest.approx(-2.0)
    assert negative.percent == pytest.approx(-10.0)
    assert negative.percent_ci95_low < negative.percent < negative.percent_ci95_high


def test_paired_savings_rejects_unaligned_or_zero_normal_samples() -> None:
    with pytest.raises(ValueError, match="aligned pairs"):
        summarize_paired_savings((10.0, 20.0), (9.0,))
    with pytest.raises(ValueError, match="normal sample mean must be positive"):
        summarize_paired_savings((0.0, 0.0), (0.0, 0.0))
