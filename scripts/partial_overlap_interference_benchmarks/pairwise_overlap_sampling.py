"""Adaptive sampling primitives for pairwise-overlap benchmark cells.

The GPU runner owns launch ordering and sample validity.  This module owns only
the CPU-side stopping rule, so it is dependency-free and straightforward to
unit test.  A *cell* is one independently sampled combination such as
``(case, victim, coverage target, dispatch order)``.

Invalid trials are recorded for diagnostics but never contribute to the
minimum sample count, bootstrap distribution, or maximum-valid-sample cap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
import statistics
from typing import Iterable, Sequence


DEFAULT_BOOTSTRAP_REPETITIONS = 2_000
DEFAULT_TARGET_CI_HALF_WIDTH_PP = 1.5
DEFAULT_MAX_VALID_SAMPLES = 500


def minimum_valid_samples_for_duration(duration_ms: float) -> int:
    """Return the minimum valid sample count for an isolated operation.

    Operations longer than 10 ms use 50 valid samples.  Operations in the
    1--10 ms band use 100.  Sub-millisecond operations conservatively retain
    the 100-sample requirement because launch jitter is not smaller merely
    because the measured kernel is shorter.
    """

    value = float(duration_ms)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("duration_ms must be finite and positive")
    return 50 if value > 10.0 else 100


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
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


@dataclass(frozen=True)
class MedianConfidenceInterval:
    """Median and deterministic non-parametric bootstrap 95% interval.

    ``ci95_half_width_pp`` remains half of the total interval width for
    descriptive compatibility. ``max_endpoint_distance_pp`` is the actual
    convergence statistic, so asymmetric intervals must place both endpoints
    within the configured target distance of the median.
    """

    samples: int
    median_pp: float
    ci95_low_pp: float
    ci95_high_pp: float
    ci95_half_width_pp: float
    max_endpoint_distance_pp: float
    bootstrap_repetitions: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def bootstrap_median_ci95(
    samples_pp: Iterable[float],
    *,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = 0,
) -> MedianConfidenceInterval:
    """Return a deterministic bootstrap 95% CI for a median in percentage points."""

    values = tuple(float(sample) for sample in samples_pp)
    if not values:
        raise ValueError("at least one valid sample is required")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("samples must be finite")
    repetitions = int(bootstrap_repetitions)
    if repetitions < 0:
        raise ValueError("bootstrap_repetitions must be non-negative")

    median = float(statistics.median(values))
    if len(values) == 1 or repetitions == 0:
        low = high = median
    else:
        rng = random.Random(int(seed))
        sample_count = len(values)
        bootstrap_medians = sorted(
            float(
                statistics.median(
                    values[rng.randrange(sample_count)] for _ in range(sample_count)
                )
            )
            for _ in range(repetitions)
        )
        low = _percentile(bootstrap_medians, 0.025)
        high = _percentile(bootstrap_medians, 0.975)

    return MedianConfidenceInterval(
        samples=len(values),
        median_pp=median,
        ci95_low_pp=low,
        ci95_high_pp=high,
        ci95_half_width_pp=max(0.0, high - low) / 2.0,
        max_endpoint_distance_pp=max(abs(median - low), abs(high - median)),
        bootstrap_repetitions=repetitions,
        seed=int(seed),
    )


def deterministic_cell_seed(cell_key: str, base_seed: int = 0) -> int:
    """Derive a process-independent bootstrap seed for one cell."""

    digest = hashlib.sha256(f"{int(base_seed)}:{cell_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class AdaptiveSamplingConfig:
    """Endpoint precision target and cap for independent benchmark cells."""

    target_ci_half_width_pp: float = DEFAULT_TARGET_CI_HALF_WIDTH_PP
    max_valid_samples: int = DEFAULT_MAX_VALID_SAMPLES
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        target = float(self.target_ci_half_width_pp)
        if not 1.0 <= target <= 2.0:
            raise ValueError("target_ci_half_width_pp must be between 1 and 2")
        if int(self.max_valid_samples) <= 0:
            raise ValueError("max_valid_samples must be positive")
        if int(self.bootstrap_repetitions) <= 0:
            raise ValueError("bootstrap_repetitions must be positive")


@dataclass(frozen=True)
class AdaptiveSamplingDecision:
    """Current state and stopping decision for one benchmark cell."""

    cell_key: str
    isolated_duration_ms: float
    valid_samples: int
    invalid_samples: int
    minimum_valid_samples: int
    max_valid_samples: int
    target_ci_half_width_pp: float
    median_pp: float | None
    ci95_low_pp: float | None
    ci95_high_pp: float | None
    ci95_half_width_pp: float | None
    max_endpoint_distance_pp: float | None
    precision_met: bool
    converged: bool
    should_continue: bool
    stop_reason: str | None

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return asdict(self)


class AdaptiveSamplingCell:
    """Mutable sample accumulator with a deterministic stopping rule."""

    def __init__(
        self,
        cell_key: str,
        isolated_duration_ms: float,
        config: AdaptiveSamplingConfig | None = None,
    ) -> None:
        if not str(cell_key):
            raise ValueError("cell_key must be non-empty")
        self.cell_key = str(cell_key)
        self.isolated_duration_ms = float(isolated_duration_ms)
        self.config = config or AdaptiveSamplingConfig()
        self.minimum_valid_samples = minimum_valid_samples_for_duration(
            self.isolated_duration_ms
        )
        if self.config.max_valid_samples < self.minimum_valid_samples:
            raise ValueError(
                "max_valid_samples must be at least the duration-based minimum "
                f"({self.minimum_valid_samples})"
            )
        self._valid_samples_pp: list[float] = []
        self._invalid_samples = 0

    @property
    def valid_samples_pp(self) -> tuple[float, ...]:
        return tuple(self._valid_samples_pp)

    @property
    def invalid_samples(self) -> int:
        return self._invalid_samples

    def add_sample(self, slowdown_pp: float | None, *, valid: bool) -> None:
        """Record one trial; invalid values are deliberately never inspected."""

        if not valid:
            self._invalid_samples += 1
            return
        if len(self._valid_samples_pp) >= self.config.max_valid_samples:
            raise RuntimeError("cell already reached max_valid_samples")
        if slowdown_pp is None:
            raise ValueError("a valid sample requires slowdown_pp")
        value = float(slowdown_pp)
        if not math.isfinite(value):
            raise ValueError("a valid sample must be finite")
        self._valid_samples_pp.append(value)

    def extend(self, samples: Iterable[tuple[float | None, bool]]) -> None:
        for slowdown_pp, valid in samples:
            self.add_sample(slowdown_pp, valid=valid)

    def decision(self) -> AdaptiveSamplingDecision:
        valid_count = len(self._valid_samples_pp)
        summary = None
        if valid_count:
            summary = bootstrap_median_ci95(
                self._valid_samples_pp,
                bootstrap_repetitions=self.config.bootstrap_repetitions,
                seed=deterministic_cell_seed(
                    self.cell_key, self.config.bootstrap_seed
                ),
            )

        enough_samples = valid_count >= self.minimum_valid_samples
        # Half of total interval width can hide an endpoint farther than the
        # target when a bootstrap interval is asymmetric. Bound both sides.
        precision_met = bool(
            enough_samples
            and summary is not None
            and summary.max_endpoint_distance_pp
            <= self.config.target_ci_half_width_pp
        )
        reached_cap = valid_count >= self.config.max_valid_samples
        converged = precision_met
        should_continue = not converged and not reached_cap

        if converged:
            stop_reason = "ci_precision_met"
        elif reached_cap:
            stop_reason = "max_valid_samples"
        else:
            stop_reason = None

        return AdaptiveSamplingDecision(
            cell_key=self.cell_key,
            isolated_duration_ms=self.isolated_duration_ms,
            valid_samples=valid_count,
            invalid_samples=self._invalid_samples,
            minimum_valid_samples=self.minimum_valid_samples,
            max_valid_samples=self.config.max_valid_samples,
            target_ci_half_width_pp=self.config.target_ci_half_width_pp,
            median_pp=None if summary is None else summary.median_pp,
            ci95_low_pp=None if summary is None else summary.ci95_low_pp,
            ci95_high_pp=None if summary is None else summary.ci95_high_pp,
            ci95_half_width_pp=None if summary is None else summary.ci95_half_width_pp,
            max_endpoint_distance_pp=(
                None if summary is None else summary.max_endpoint_distance_pp
            ),
            precision_met=precision_met,
            converged=converged,
            should_continue=should_continue,
            stop_reason=stop_reason,
        )


__all__ = [
    "AdaptiveSamplingCell",
    "AdaptiveSamplingConfig",
    "AdaptiveSamplingDecision",
    "DEFAULT_BOOTSTRAP_REPETITIONS",
    "DEFAULT_MAX_VALID_SAMPLES",
    "DEFAULT_TARGET_CI_HALF_WIDTH_PP",
    "MedianConfidenceInterval",
    "bootstrap_median_ci95",
    "deterministic_cell_seed",
    "minimum_valid_samples_for_duration",
]
