"""Counterfactual attribution for normal-to-interference-free replay savings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import statistics
from typing import Any, Mapping, Sequence

from .profiling import (
    INTERFERENCE_FREE_PROFILE_MODE,
    NORMAL_PROFILE_MODE,
    get_node_profile_times,
    get_prologue_profile_times,
)
from .simulator import (
    SimulationValidationError,
    _comm_id,
    profile_iteration_ids,
    simulate_training_step,
)


_P2P_KINDS = frozenset({"SEND_COMM", "RECV_COMM"})
_COLLECTIVE_KINDS = frozenset(
    {"ALL_GATHER_COMM", "REDUCE_COMM", "REDUCE_SCATTER_COMM"}
)


class AttributionValidationError(ValueError):
    """Raised when a hybrid replay cannot reproduce the two endpoint modes."""


@dataclass(frozen=True)
class AttributionFeature:
    """One profile unit switched atomically from normal to IF timing."""

    feature_id: str
    pp_indices: tuple[int, ...]
    uids: tuple[str, ...]
    node_kinds: tuple[str, ...]
    streams: tuple[str, ...]
    ranks: tuple[int, ...]
    binding_count: int
    raw_profile_savings_us: float


@dataclass(frozen=True)
class FeatureSavingsAttribution:
    """Realized makespan contribution of one attribution feature."""

    feature: AttributionFeature
    attributed_savings_us: float
    contribution_std_us: float
    contribution_ci95_us: float
    share_of_step_savings_pct: float | None
    positive_exposure_probability: float
    hidden_probability: float
    negative_probability: float
    marginal_samples_us: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["feature"] = asdict(self.feature)
        value["marginal_samples_us"] = list(self.marginal_samples_us)
        return value


@dataclass(frozen=True)
class SavingsAttributionResult:
    """Additive attribution of one point normal/IF simulation comparison."""

    normal_step_time_us: float
    interference_free_step_time_us: float
    total_step_savings_us: float
    attributed_step_savings_us: float
    additivity_residual_us: float
    permutations: int
    seed: int
    exposure_threshold_us: float
    step_time_estimator: str
    sample_iteration_ids: tuple[int, ...]
    features: tuple[FeatureSavingsAttribution, ...]

    @property
    def exposed_features(self) -> tuple[FeatureSavingsAttribution, ...]:
        return tuple(
            item
            for item in self.features
            if item.attributed_savings_us > self.exposure_threshold_us
        )

    @property
    def positive_exposed_step_savings_us(self) -> float:
        """Sum of positive feature-level makespan marginals.

        This deliberately uses each logical feature exactly once.  A feature
        may match several physical rank events in the rendered trace, and
        those repeated events must not duplicate its contribution.
        """

        return sum(
            max(0.0, item.attributed_savings_us) for item in self.features
        )

    @property
    def negative_exposed_step_savings_us(self) -> float:
        """Sum of negative feature-level makespan marginals."""

        return sum(
            min(0.0, item.attributed_savings_us) for item in self.features
        )

    @property
    def net_exposed_step_savings_us(self) -> float:
        """Signed normal-to-IF step saving from all unique feature deltas."""

        return (
            self.positive_exposed_step_savings_us
            + self.negative_exposed_step_savings_us
        )

    @property
    def net_exposed_step_savings_pct(self) -> float | None:
        if self.normal_step_time_us == 0.0:
            return None
        return 100.0 * self.net_exposed_step_savings_us / self.normal_step_time_us

    @property
    def exposed_savings_validation_residual_us(self) -> float:
        """Checksum residual against the simulator's normal-to-IF delta."""

        return self.net_exposed_step_savings_us - self.total_step_savings_us

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal_step_time_us": self.normal_step_time_us,
            "interference_free_step_time_us": (
                self.interference_free_step_time_us
            ),
            "total_step_savings_us": self.total_step_savings_us,
            "attributed_step_savings_us": self.attributed_step_savings_us,
            "additivity_residual_us": self.additivity_residual_us,
            "positive_exposed_step_savings_us": (
                self.positive_exposed_step_savings_us
            ),
            "negative_exposed_step_savings_us": (
                self.negative_exposed_step_savings_us
            ),
            "net_exposed_step_savings_us": self.net_exposed_step_savings_us,
            "net_exposed_step_savings_pct": self.net_exposed_step_savings_pct,
            "exposed_savings_validation_residual_us": (
                self.exposed_savings_validation_residual_us
            ),
            "permutations": self.permutations,
            "seed": self.seed,
            "exposure_threshold_us": self.exposure_threshold_us,
            "step_time_estimator": self.step_time_estimator,
            "sample_iteration_ids": list(self.sample_iteration_ids),
            "exposed_feature_ids": [
                item.feature.feature_id for item in self.exposed_features
            ],
            "features": [item.to_dict() for item in self.features],
        }


@dataclass
class _ProfileBinding:
    profiles: dict[int, Any]
    rank: int
    normal_profile: Any
    interference_free_profile: Any
    pp_index: int
    uid: str
    node_kind: str
    stream: str

    def select_normal(self) -> None:
        self.profiles[self.rank] = self.normal_profile

    def select_interference_free(self) -> None:
        self.profiles[self.rank] = self.interference_free_profile


@dataclass
class _MutableFeature:
    descriptor: AttributionFeature
    bindings: tuple[_ProfileBinding, ...]

    def select_normal(self) -> None:
        for binding in self.bindings:
            binding.select_normal()

    def select_interference_free(self) -> None:
        for binding in self.bindings:
            binding.select_interference_free()


def _per_pp_dags(plan_or_dags: Any) -> list[Any]:
    if hasattr(plan_or_dags, "per_pp_dags"):
        return list(plan_or_dags.per_pp_dags)
    return list(plan_or_dags)


def _feature_id(pp_index: int, node: Any) -> str:
    if node.node_kind in _P2P_KINDS:
        return f"comm:p2p:{_comm_id(node)}"
    if node.node_kind in _COLLECTIVE_KINDS:
        return f"comm:{node.node_kind.lower()}:{_comm_id(node)}"
    return f"node:pp{pp_index}:{node.uid}"


def _profile_estimate_us(profile: Any) -> float:
    return float(getattr(profile, "estimate_us", 0.0))


def _build_features(plan_or_dags: Any) -> tuple[_MutableFeature, ...]:
    grouped: dict[str, list[_ProfileBinding]] = {}
    for pp_index, dag in enumerate(_per_pp_dags(plan_or_dags)):
        for node in dag.nodes.values():
            normal = get_node_profile_times(node, NORMAL_PROFILE_MODE)
            interference_free = get_node_profile_times(
                node, INTERFERENCE_FREE_PROFILE_MODE
            )
            if set(normal) != set(interference_free):
                raise AttributionValidationError(
                    f"node {node.uid!r} PP index {pp_index} has different "
                    "normal and IF profile ranks"
                )
            feature_id = _feature_id(pp_index, node)
            bindings = grouped.setdefault(feature_id, [])
            for rank in sorted(normal):
                bindings.append(
                    _ProfileBinding(
                        profiles=normal,
                        rank=int(rank),
                        normal_profile=normal[rank],
                        interference_free_profile=interference_free[rank],
                        pp_index=pp_index,
                        uid=str(node.uid),
                        node_kind=str(node.node_kind),
                        stream=str(node.stream),
                    )
                )

        normal_prologue = get_prologue_profile_times(
            dag, NORMAL_PROFILE_MODE
        )
        interference_free_prologue = get_prologue_profile_times(
            dag, INTERFERENCE_FREE_PROFILE_MODE
        )
        if set(normal_prologue) != set(interference_free_prologue):
            raise AttributionValidationError(
                f"PP index {pp_index} has different normal and IF prologue ranks"
            )
        if normal_prologue:
            bindings = grouped.setdefault(f"prologue:pp{pp_index}", [])
            for rank in sorted(normal_prologue):
                bindings.append(
                    _ProfileBinding(
                        profiles=normal_prologue,
                        rank=int(rank),
                        normal_profile=normal_prologue[rank],
                        interference_free_profile=(
                            interference_free_prologue[rank]
                        ),
                        pp_index=pp_index,
                        uid="__prologue__",
                        node_kind="STEP_PROLOGUE",
                        stream="__prologue__",
                    )
                )

    features: list[_MutableFeature] = []
    for feature_id, bindings in sorted(grouped.items()):
        descriptor = AttributionFeature(
            feature_id=feature_id,
            pp_indices=tuple(sorted({item.pp_index for item in bindings})),
            uids=tuple(sorted({item.uid for item in bindings})),
            node_kinds=tuple(sorted({item.node_kind for item in bindings})),
            streams=tuple(sorted({item.stream for item in bindings})),
            ranks=tuple(sorted({item.rank for item in bindings})),
            binding_count=len(bindings),
            raw_profile_savings_us=sum(
                _profile_estimate_us(item.normal_profile)
                - _profile_estimate_us(item.interference_free_profile)
                for item in bindings
            ),
        )
        features.append(
            _MutableFeature(descriptor=descriptor, bindings=tuple(bindings))
        )
    if not features:
        raise AttributionValidationError("profile contains no attributable nodes")
    return tuple(features)


def _orders(
    feature_count: int,
    *,
    permutations: int,
    seed: int,
    antithetic: bool,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    result: list[tuple[int, ...]] = []
    while len(result) < permutations:
        order = list(range(feature_count))
        rng.shuffle(order)
        result.append(tuple(order))
        if antithetic and len(result) < permutations:
            result.append(tuple(reversed(order)))
    return tuple(result)


def _shared_profile_iteration_ids(plan_or_dags: Any) -> tuple[int, ...]:
    """Return paired normal/IF raw sample IDs, or empty for legacy profiles."""

    errors: dict[str, SimulationValidationError] = {}
    ids_by_mode: dict[str, tuple[int, ...]] = {}
    for mode in (NORMAL_PROFILE_MODE, INTERFERENCE_FREE_PROFILE_MODE):
        try:
            ids_by_mode[mode] = profile_iteration_ids(
                plan_or_dags, profile_mode=mode
            )
        except SimulationValidationError as exc:
            errors[mode] = exc
    if len(errors) == 2:
        has_raw_samples = False
        for dag in _per_pp_dags(plan_or_dags):
            for node in dag.nodes.values():
                for mode in (
                    NORMAL_PROFILE_MODE,
                    INTERFERENCE_FREE_PROFILE_MODE,
                ):
                    has_raw_samples = has_raw_samples or any(
                        bool(getattr(profile, "samples_us", ()))
                        for profile in get_node_profile_times(node, mode).values()
                    )
            for mode in (
                NORMAL_PROFILE_MODE,
                INTERFERENCE_FREE_PROFILE_MODE,
            ):
                has_raw_samples = has_raw_samples or any(
                    bool(getattr(profile, "samples_us", ()))
                    for profile in get_prologue_profile_times(dag, mode).values()
                )
        if not has_raw_samples:
            # Unit-test and legacy duration-only plans have only point estimates.
            return ()
        mode, error = next(iter(errors.items()))
        raise AttributionValidationError(
            f"cannot attribute median replay: {mode} raw samples are invalid: "
            f"{error}"
        ) from error
    if errors:
        mode, error = next(iter(errors.items()))
        raise AttributionValidationError(
            f"cannot attribute median replay: {mode} raw samples are invalid: "
            f"{error}"
        ) from error
    normal_ids = ids_by_mode[NORMAL_PROFILE_MODE]
    interference_free_ids = ids_by_mode[INTERFERENCE_FREE_PROFILE_MODE]
    if normal_ids != interference_free_ids:
        raise AttributionValidationError(
            "normal and IF raw profile iteration IDs differ: "
            f"normal={normal_ids!r}, IF={interference_free_ids!r}"
        )
    return normal_ids


def _simulate_step_objective(
    plan_or_dags: Any,
    *,
    profile_mode: str,
    sample_iteration_ids: tuple[int, ...],
    strict: bool,
) -> float:
    """Evaluate the same median step estimator reported by the comparison."""

    if not sample_iteration_ids:
        return simulate_training_step(
            plan_or_dags,
            profile_mode=profile_mode,
            strict=strict,
        ).step_time_us
    return float(
        statistics.median(
            simulate_training_step(
                plan_or_dags,
                profile_mode=profile_mode,
                sample_iteration_id=iteration_id,
                strict=strict,
            ).step_time_us
            for iteration_id in sample_iteration_ids
        )
    )


def attribute_profile_mode_savings(
    plan_or_dags: Any,
    *,
    permutations: int = 8,
    seed: int = 0,
    antithetic: bool = True,
    exposure_threshold_us: float = 1.0,
    strict: bool = True,
) -> SavingsAttributionResult:
    """Attribute point normal-to-IF makespan savings with replay marginals.

    Every permutation starts from the normal profile and switches one logical
    node or atomic communication transaction at a time.  Its marginal
    makespan reductions telescope exactly to the endpoint difference; their
    average is a Monte Carlo Shapley estimate.  Reversed (antithetic) orders
    reduce path-order bias without changing additivity.
    """

    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if not math.isfinite(exposure_threshold_us) or exposure_threshold_us < 0:
        raise ValueError("exposure_threshold_us must be finite and non-negative")

    features = _build_features(plan_or_dags)
    sample_iteration_ids = _shared_profile_iteration_ids(plan_or_dags)
    normal = _simulate_step_objective(
        plan_or_dags,
        profile_mode=NORMAL_PROFILE_MODE,
        sample_iteration_ids=sample_iteration_ids,
        strict=strict,
    )
    canonical_if = _simulate_step_objective(
        plan_or_dags,
        profile_mode=INTERFERENCE_FREE_PROFILE_MODE,
        sample_iteration_ids=sample_iteration_ids,
        strict=strict,
    )

    try:
        for feature in features:
            feature.select_interference_free()
        hybrid_if = _simulate_step_objective(
            plan_or_dags,
            profile_mode=NORMAL_PROFILE_MODE,
            sample_iteration_ids=sample_iteration_ids,
            strict=strict,
        )
    finally:
        for feature in features:
            feature.select_normal()

    endpoint_tolerance_us = max(1e-6, 1e-9 * max(normal, canonical_if, 1.0))
    if not math.isclose(
        hybrid_if,
        canonical_if,
        rel_tol=0.0,
        abs_tol=endpoint_tolerance_us,
    ):
        raise AttributionValidationError(
            "hybrid replay with every feature set to IF does not reproduce "
            f"the IF endpoint: hybrid={hybrid_if:.6f} us, "
            f"canonical={canonical_if:.6f} us"
        )

    marginal_samples: list[list[float]] = [list() for _ in features]
    permutation_orders = _orders(
        len(features),
        permutations=permutations,
        seed=seed,
        antithetic=antithetic,
    )
    total_savings = normal - canonical_if
    try:
        for permutation_index, order in enumerate(permutation_orders):
            current = normal
            for feature_index in order:
                features[feature_index].select_interference_free()
                updated = _simulate_step_objective(
                    plan_or_dags,
                    profile_mode=NORMAL_PROFILE_MODE,
                    sample_iteration_ids=sample_iteration_ids,
                    strict=strict,
                )
                marginal_samples[feature_index].append(current - updated)
                current = updated
            permutation_residual = (normal - current) - total_savings
            if abs(permutation_residual) > endpoint_tolerance_us:
                raise AttributionValidationError(
                    f"permutation {permutation_index} does not telescope: "
                    f"residual={permutation_residual:.6f} us"
                )
            for feature in features:
                feature.select_normal()
    finally:
        for feature in features:
            feature.select_normal()

    rows: list[FeatureSavingsAttribution] = []
    for feature, samples in zip(features, marginal_samples):
        mean = statistics.fmean(samples)
        std = statistics.stdev(samples) if len(samples) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(samples))
        positive = sum(value > exposure_threshold_us for value in samples)
        negative = sum(value < -exposure_threshold_us for value in samples)
        hidden = len(samples) - positive - negative
        rows.append(
            FeatureSavingsAttribution(
                feature=feature.descriptor,
                attributed_savings_us=mean,
                contribution_std_us=std,
                contribution_ci95_us=ci95,
                share_of_step_savings_pct=(
                    None if total_savings == 0.0 else 100.0 * mean / total_savings
                ),
                positive_exposure_probability=positive / len(samples),
                hidden_probability=hidden / len(samples),
                negative_probability=negative / len(samples),
                marginal_samples_us=tuple(samples),
            )
        )
    rows.sort(key=lambda item: (-item.attributed_savings_us, item.feature.feature_id))
    attributed = sum(item.attributed_savings_us for item in rows)
    return SavingsAttributionResult(
        normal_step_time_us=normal,
        interference_free_step_time_us=canonical_if,
        total_step_savings_us=total_savings,
        attributed_step_savings_us=attributed,
        additivity_residual_us=attributed - total_savings,
        permutations=permutations,
        seed=seed,
        exposure_threshold_us=exposure_threshold_us,
        step_time_estimator=("median" if sample_iteration_ids else "point"),
        sample_iteration_ids=sample_iteration_ids,
        features=tuple(rows),
    )


__all__ = [
    "AttributionFeature",
    "AttributionValidationError",
    "FeatureSavingsAttribution",
    "SavingsAttributionResult",
    "attribute_profile_mode_savings",
]
