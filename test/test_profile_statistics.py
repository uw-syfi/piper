import math

import pytest

from src.profile_statistics import (
    predicted_reduction,
    prediction_error,
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
