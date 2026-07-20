"""Statistical summaries for profiler and simulator timing samples.

The helpers in this module are deliberately dependency-free and deterministic.
They are shared by the example harness and offline simulator so CSV and JSON
artifacts use identical definitions of confidence and prediction error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import statistics
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class TimingStatistics:
    """One timing distribution and the uncertainty of its center estimate."""

    samples: int
    estimator: str
    estimate: float
    mean: float
    median: float
    std: float
    ci95_low: float
    ci95_high: float
    error_pct: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionError:
    """Signed and absolute error of a prediction against measured truth."""

    signed_pct: float
    absolute_pct: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _validated_samples(samples: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(sample) for sample in samples)
    if not values:
        raise ValueError("at least one sample is required")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"samples must be finite and non-negative: {values!r}")
    return values


def _estimator(name: str) -> Callable[[Sequence[float]], float]:
    if name == "mean":
        return lambda values: float(statistics.fmean(values))
    if name == "median":
        return lambda values: float(statistics.median(values))
    raise ValueError(f"unsupported statistics estimator: {name!r}")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Linearly interpolated percentile without a NumPy dependency."""

    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = min(1.0, max(0.0, float(probability))) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_timing_samples(
    samples: Iterable[float],
    *,
    estimator: str = "mean",
    bootstrap_repetitions: int = 1_000,
    seed: int = 0,
) -> TimingStatistics:
    """Summarize samples with a deterministic non-parametric 95% CI.

    ``error_pct`` is the 95% confidence-interval half-width divided by the
    selected center estimate.  It is sampling uncertainty, not prediction
    error.  A zero-valued estimate has zero error only when its interval is
    also exactly zero; otherwise its relative error is infinite.
    """

    values = _validated_samples(samples)
    statistic = _estimator(estimator)
    estimate = statistic(values)
    mean = float(statistics.fmean(values))
    median = float(statistics.median(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0

    if len(values) == 1 or bootstrap_repetitions <= 0:
        ci_low = ci_high = estimate
    else:
        rng = random.Random(int(seed))
        sample_count = len(values)
        bootstrapped = sorted(
            statistic(tuple(values[rng.randrange(sample_count)] for _ in range(sample_count)))
            for _ in range(int(bootstrap_repetitions))
        )
        ci_low = _percentile(bootstrapped, 0.025)
        ci_high = _percentile(bootstrapped, 0.975)

    half_width = max(0.0, ci_high - ci_low) / 2.0
    if estimate == 0.0:
        error_pct = 0.0 if half_width == 0.0 else math.inf
    else:
        error_pct = half_width / abs(estimate) * 100.0
    return TimingStatistics(
        samples=len(values),
        estimator=estimator,
        estimate=estimate,
        mean=mean,
        median=median,
        std=std,
        ci95_low=ci_low,
        ci95_high=ci_high,
        error_pct=error_pct,
    )


def prediction_error(predicted: float, actual: float) -> PredictionError:
    """Return percentage error of ``predicted`` relative to ``actual``."""

    predicted_value = float(predicted)
    actual_value = float(actual)
    if not math.isfinite(predicted_value) or not math.isfinite(actual_value):
        raise ValueError("prediction and actual values must be finite")
    if actual_value <= 0.0:
        raise ValueError("actual value must be positive for percentage error")
    signed = (predicted_value - actual_value) / actual_value * 100.0
    return PredictionError(signed_pct=signed, absolute_pct=abs(signed))


def predicted_reduction(candidate: float, baseline: float) -> float:
    """Percentage reduction of ``candidate`` relative to ``baseline``."""

    candidate_value = float(candidate)
    baseline_value = float(baseline)
    if not math.isfinite(candidate_value) or not math.isfinite(baseline_value):
        raise ValueError("candidate and baseline values must be finite")
    if baseline_value <= 0.0:
        raise ValueError("baseline must be positive for percentage reduction")
    return (baseline_value - candidate_value) / baseline_value * 100.0


__all__ = [
    "PredictionError",
    "TimingStatistics",
    "predicted_reduction",
    "prediction_error",
    "summarize_timing_samples",
]
