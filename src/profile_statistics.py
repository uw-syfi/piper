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
class MeanStatistics:
    """Arithmetic-mean estimate with a Student-t confidence interval."""

    samples: int
    estimator: str
    estimate: float
    mean: float
    median: float
    std: float | None
    ci95_low: float | None
    ci95_high: float | None
    error_pct: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class PairedSavingsStatistics:
    """Mean absolute and ratio-of-means savings for aligned observations."""

    samples: int
    absolute_mean: float
    percent: float
    percent_standard_error: float | None
    percent_ci95_low: float | None
    percent_ci95_high: float | None

    def to_dict(self) -> dict[str, float | int | None]:
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


def _continued_fraction_beta(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by regularized incomplete beta."""

    max_iterations = 200
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    scale = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return scale * _continued_fraction_beta(a, b, x) / a
    return 1.0 - scale * _continued_fraction_beta(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if value == 0.0:
        return 0.5
    ratio = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        ratio, degrees_of_freedom / 2.0, 0.5
    )
    return 1.0 - tail if value > 0.0 else tail


def student_t_critical_95(degrees_of_freedom: int) -> float:
    """Return the positive two-sided 95% Student-t critical value."""

    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    target = 0.975
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < target:
        upper *= 2.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def summarize_mean_samples(samples: Iterable[float]) -> MeanStatistics:
    """Summarize samples without resampling, using a Student-t mean CI."""

    values = _validated_samples(samples)
    sample_count = len(values)
    mean = float(statistics.fmean(values))
    median = float(statistics.median(values))
    if sample_count == 1:
        std = None
        ci_low = None
        ci_high = None
        error_pct = None
    else:
        std = float(statistics.stdev(values))
        half_width = (
            student_t_critical_95(sample_count - 1)
            * std
            / math.sqrt(sample_count)
        )
        ci_low = mean - half_width
        ci_high = mean + half_width
        error_pct = (
            (0.0 if half_width == 0.0 else math.inf)
            if mean == 0.0
            else half_width / abs(mean) * 100.0
        )
    return MeanStatistics(
        samples=sample_count,
        estimator="mean",
        estimate=mean,
        mean=mean,
        median=median,
        std=std,
        ci95_low=ci_low,
        ci95_high=ci_high,
        error_pct=error_pct,
    )


def summarize_paired_savings(
    normal_samples: Iterable[float],
    interference_free_samples: Iterable[float],
) -> PairedSavingsStatistics:
    """Summarize paired absolute savings and ratio-of-means percent savings."""

    normal = _validated_samples(normal_samples)
    interference_free = _validated_samples(interference_free_samples)
    if len(normal) != len(interference_free):
        raise ValueError(
            "normal and interference-free samples must form aligned pairs: "
            f"normal={len(normal)}, interference_free={len(interference_free)}"
        )
    sample_count = len(normal)
    normal_mean = float(statistics.fmean(normal))
    interference_free_mean = float(statistics.fmean(interference_free))
    if normal_mean <= 0.0:
        raise ValueError("normal sample mean must be positive for percent savings")
    absolute_mean = float(
        statistics.fmean(
            normal_value - interference_free_value
            for normal_value, interference_free_value in zip(
                normal, interference_free
            )
        )
    )
    percent = 100.0 * absolute_mean / normal_mean
    if sample_count == 1:
        standard_error = None
        ci_low = None
        ci_high = None
    else:
        normal_variance = float(statistics.variance(normal))
        interference_free_variance = float(statistics.variance(interference_free))
        covariance = sum(
            (normal_value - normal_mean)
            * (interference_free_value - interference_free_mean)
            for normal_value, interference_free_value in zip(
                normal, interference_free
            )
        ) / (sample_count - 1)
        normal_gradient = 100.0 * interference_free_mean / (normal_mean**2)
        interference_free_gradient = -100.0 / normal_mean
        estimated_variance = (
            normal_gradient**2 * normal_variance
            + interference_free_gradient**2 * interference_free_variance
            + 2.0
            * normal_gradient
            * interference_free_gradient
            * covariance
        ) / sample_count
        # Roundoff can make an analytically non-negative quadratic form a tiny
        # negative number for perfectly correlated samples.
        tolerance = 1.0e-12 * max(
            1.0,
            normal_gradient**2 * normal_variance,
            interference_free_gradient**2 * interference_free_variance,
        )
        if estimated_variance < -tolerance:
            raise ArithmeticError(
                "paired delta-method variance is unexpectedly negative: "
                f"{estimated_variance}"
            )
        standard_error = math.sqrt(max(0.0, estimated_variance))
        half_width = student_t_critical_95(sample_count - 1) * standard_error
        ci_low = percent - half_width
        ci_high = percent + half_width
    return PairedSavingsStatistics(
        samples=sample_count,
        absolute_mean=absolute_mean,
        percent=percent,
        percent_standard_error=standard_error,
        percent_ci95_low=ci_low,
        percent_ci95_high=ci_high,
    )


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
            statistic(
                tuple(
                    values[rng.randrange(sample_count)]
                    for _ in range(sample_count)
                )
            )
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


def summarize_paired_reductions(
    baseline_samples: Iterable[float],
    candidate_samples: Iterable[float],
    *,
    bootstrap_repetitions: int = 1_000,
    seed: int = 0,
) -> tuple[tuple[float, ...], TimingStatistics]:
    """Summarize aligned per-pair percentage reductions with a median CI.

    The percentage reduction is computed for every aligned
    ``(baseline, candidate)`` pair *before* bootstrap resampling.  The returned
    point estimate is the median of those paired reductions, and the 95%
    interval is the percentile interval obtained by resampling that one list
    with replacement.  Pairing is therefore never broken by the bootstrap.
    """

    baseline = _validated_samples(baseline_samples)
    candidate = _validated_samples(candidate_samples)
    if len(baseline) != len(candidate):
        raise ValueError(
            "baseline and candidate samples must form aligned pairs: "
            f"baseline={len(baseline)}, candidate={len(candidate)}"
        )
    reductions = tuple(
        predicted_reduction(candidate_value, baseline_value)
        for baseline_value, candidate_value in zip(baseline, candidate)
    )
    # Percentage reductions may be negative, so summarize them directly
    # instead of using the non-negative timing-sample validator.
    statistic = _estimator("median")
    estimate = statistic(reductions)
    mean = float(statistics.fmean(reductions))
    median = float(statistics.median(reductions))
    std = float(statistics.stdev(reductions)) if len(reductions) > 1 else 0.0
    if len(reductions) == 1 or bootstrap_repetitions <= 0:
        ci_low = ci_high = estimate
    else:
        rng = random.Random(int(seed))
        pair_count = len(reductions)
        bootstrapped = sorted(
            statistic(
                tuple(
                    reductions[rng.randrange(pair_count)]
                    for _ in range(pair_count)
                )
            )
            for _ in range(int(bootstrap_repetitions))
        )
        ci_low = _percentile(bootstrapped, 0.025)
        ci_high = _percentile(bootstrapped, 0.975)
    half_width = max(0.0, ci_high - ci_low) / 2.0
    error_pct = (
        (0.0 if half_width == 0.0 else math.inf)
        if estimate == 0.0
        else half_width / abs(estimate) * 100.0
    )
    return reductions, TimingStatistics(
        samples=len(reductions),
        estimator="paired_median",
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
    "MeanStatistics",
    "PairedSavingsStatistics",
    "PredictionError",
    "TimingStatistics",
    "predicted_reduction",
    "prediction_error",
    "student_t_critical_95",
    "summarize_mean_samples",
    "summarize_paired_reductions",
    "summarize_paired_savings",
    "summarize_timing_samples",
]
