import time
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator, Mapping, Sequence

import ray
import torch.fx as fx
from torch._dynamo.backends.registry import register_backend

from .fx import PIPER_ANNOTATIONS_META_KEY, split_gm_by_annotations
from .dag import (
    TrainingDAG,
    TrainingDAGEdge,
    TrainingDAGNode,
    build_training_dag,
)
from .directives import (
    _apply_order_directive,
    _apply_split_backward_stencil,
    _apply_split_directive,
    _bucket_matched_fwd_nodes,
    _parse_order_directive,
    _validate_schedule_tags_exist,
    _validate_split_backward_order_stencil,
    apply_schedule_directives,
)
from .ordering import (
    _serial_topological_order,
    resolve_total_order_per_stream,
)
from .profile_statistics import summarize_timing_samples
from .profiling import (
    NORMAL_PROFILE_MODE,
    DEFAULT_NORMAL_SCOPE,
    INTERFERENCE_FREE_PROFILE_MODE,
    ProfileConfig,
    ProfiledTrainingPlan,
    merge_actor_profiles,
    normalize_profile_mode,
)
from .state import LOG_LEVEL, create_logger, piper_metadata
from .visualization import (
    log_training_dag_dependencies,
    print_training_dag_order,
    render_training_dag,
)
from .zero import _add_inter_chain_temporal_edges, _prune_zero_lifetime_metadata

logger = create_logger("piper_backend", LOG_LEVEL)
_ANNOTATION_STACK: list[dict[str, int]] = []
_ANNOTATION_COUNTS: dict[str, int] = {}
_ANNOTATION_UID = 0


@dataclass(frozen=True)
class TimingResult:
    """A fixed-size set of hardware step and peak-memory samples."""

    step_samples_s: tuple[float, ...]
    peak_memory_samples_by_rank: dict[int, tuple[int, ...]]


def _reset_annotation_state() -> None:
    global _ANNOTATION_UID
    _ANNOTATION_STACK.clear()
    _ANNOTATION_COUNTS.clear()
    _ANNOTATION_UID = 0


@contextmanager
def annotate(name: str) -> Iterator[dict[str, int]]:
    """Annotate traced model code with a Piper schedule tag.

    Piper assigns the integer index for each tag name automatically in the
    order annotation scopes are entered during tracing.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"piper.annotate requires a non-empty string tag name, got {name!r}")

    global _ANNOTATION_UID
    index = _ANNOTATION_COUNTS.get(name, 0)
    _ANNOTATION_COUNTS[name] = index + 1
    uid = _ANNOTATION_UID
    _ANNOTATION_UID += 1

    annotation = {"name": name, "index": int(index), "uid": int(uid)}
    _ANNOTATION_STACK.append(annotation)
    fx_metadata_stack = tuple(dict(item) for item in _ANNOTATION_STACK)
    try:
        with fx.traceback.annotate({
            PIPER_ANNOTATIONS_META_KEY: fx_metadata_stack,
            "name": name,
            "index": int(index),
        }):
            yield annotation
    finally:
        popped = _ANNOTATION_STACK.pop()
        if popped is not annotation:
            raise RuntimeError("piper.annotate stack corrupted during tracing")



def _split_global_training_dag_by_pp_rank(
    training_dag: TrainingDAG,
    *,
    merge_same_device_components: bool = False,
) -> list[TrainingDAG]:
    """Split the global DAG into per-device-set disconnected DAGs."""
    # SEND/RECV pairs intentionally have no edge between them, so cross-rank
    # placement dependencies separate into disconnected local components here.
    undirected: dict[str, set[str]] = {uid: set() for uid in training_dag.nodes}
    for e in training_dag.edges:
        if e.src_uid in undirected and e.dst_uid in undirected:
            undirected[e.src_uid].add(e.dst_uid)
            undirected[e.dst_uid].add(e.src_uid)

    components: list[set[str]] = []
    seen: set[str] = set()
    for uid in training_dag.nodes:
        if uid in seen:
            continue
        comp: set[str] = set()
        stack = [uid]
        seen.add(uid)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in undirected.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)

    # Validate each component is device-homogeneous and component device-sets are distinct.
    comp_device_keys: list[tuple[int, ...]] = []
    for ci, comp in enumerate(components):
        device_keys = {
            tuple(sorted(node.device)) for uid in comp for node in [training_dag.nodes[uid]] if node.device is not None
        }
        if not device_keys:
            raise ValueError(f"component[{ci}] has no device assignment after P2P split")
        if len(device_keys) != 1:
            raise ValueError(
                f"component[{ci}] is not device-homogeneous; device sets present: {sorted(device_keys)}"
            )
        comp_device_keys.append(next(iter(device_keys)))
    if (
        len(set(comp_device_keys)) != len(comp_device_keys)
        and merge_same_device_components
    ):
        merged_by_device: dict[tuple[int, ...], set[str]] = {}
        for comp, device_key in zip(components, comp_device_keys):
            merged_by_device.setdefault(device_key, set()).update(comp)
        components = list(merged_by_device.values())
        comp_device_keys = list(merged_by_device)
    elif len(set(comp_device_keys)) != len(comp_device_keys):
        raise ValueError(
            f"expected distinct device sets across split components, got {comp_device_keys}"
        )

    # Materialize each component as a standalone TrainingDAG.
    subdags: list[TrainingDAG] = []
    for comp in components:
        sub = TrainingDAG()
        sub.prologue_profile_times_by_mode = {
            mode: dict(profiles)
            for mode, profiles in training_dag.prologue_profile_times_by_mode.items()
        }
        for uid in training_dag.nodes:
            if uid not in comp:
                continue
            n = training_dag.nodes[uid]
            sub.add_node(
                TrainingDAGNode(
                    uid=n.uid,
                    node_kind=n.node_kind,
                    compute_subkind=n.compute_subkind,
                    tag=dict(n.tag),
                    device=(None if n.device is None else list(n.device)),
                    stream=n.stream,
                    node_meta=dict(n.node_meta),
                    profile_times_by_mode={
                        mode: dict(profiles)
                        for mode, profiles in n.profile_times_by_mode.items()
                    },
                )
            )
        for e in training_dag.edges:
            if e.src_uid in comp and e.dst_uid in comp:
                sub.add_edge(
                    TrainingDAGEdge(
                        src_uid=e.src_uid,
                        dst_uid=e.dst_uid,
                        dep_kind=e.dep_kind,
                        tensor_name=e.tensor_name,
                    )
                )
        subdags.append(sub)

    def _dag_device_key(d: TrainingDAG) -> tuple[int, ...]:
        keys = {tuple(sorted(n.device)) for n in d.nodes.values() if n.device is not None}
        if len(keys) != 1:
            raise ValueError(f"sub-DAG should have exactly one device key, got {keys}")
        return next(iter(keys))

    subdags.sort(key=_dag_device_key)
    return subdags


@register_backend
def piper(gm, example_inputs, **kwargs):
    """TrainingDAG backend: split by Piper annotations and lower schedule directives."""
    del example_inputs, kwargs

    schedule_info = getattr(piper_metadata, "schedule_info", {}) or {}
    schedule_directives = getattr(piper_metadata, "schedule_directives", None)
    _top_level_gm, annotation_segments = split_gm_by_annotations(gm)

    if not annotation_segments:
        raise ValueError(
            "No Piper annotations found in the traced graph. Wrap model compute "
            "with src.piper.annotate(...) before compiling with Piper."
        )

    # Build and store the new directed DAG representation for later scheduling transforms.
    training_dag = build_training_dag(annotation_segments)
    _validate_schedule_tags_exist(training_dag, schedule_directives)
    apply_schedule_directives(
        training_dag,
        schedule_directives,
    )
    piper_metadata.training_dag = training_dag
    artifact_dir = getattr(piper_metadata, "artifact_dir", "out")
    if getattr(piper_metadata, "visualize_dag", False):
        print_training_dag_order(
            training_dag,
            label="global_training_dag_order",
            out_dir=artifact_dir,
            output_name="global_training_dag_order",
        )
    per_pp_training_dags = _split_global_training_dag_by_pp_rank(training_dag)
    for i, subdag in enumerate(per_pp_training_dags):
        zero_chains = _prune_zero_lifetime_metadata(subdag)
        resolve_total_order_per_stream(subdag)
        _add_inter_chain_temporal_edges(subdag, zero_chains)
        if getattr(piper_metadata, "visualize_dag", False):
            log_training_dag_dependencies(subdag)
            print_training_dag_order(subdag, label=f"pp{i}", rank=i, out_dir=artifact_dir)
            # render_training_dag(subdag, output_path=f"{artifact_dir}/training_dag_pp{i}")
    piper_metadata.per_pp_training_dags = per_pp_training_dags

    logger.debug(
        "built TrainingDAG with %d nodes and %d edges, split into %d local PP-rank DAG(s)",
        len(training_dag.nodes),
        len(training_dag.edges),
        len(per_pp_training_dags),
    )

    def callback(*args, _gm=gm):
        logger.debug(
            "piper compiled callback invoked directly; running local graph execution"
        )
        return _gm(*args)

    return callback


def piper_exec_dag(loss_fn, log_stats: bool = False) -> list:
    """Execute one training step using the loaded per-rank TrainingDAG."""
    actors = piper_metadata.actors
    run_refs = [
        actor.run_dag.remote(loss_fn=loss_fn)
        for actor in actors.values()
    ]
    t0 = time.perf_counter()
    results = ray.get(run_refs)
    step_time = time.perf_counter() - t0

    if log_stats:
        _log_step_stats(step_time, log_stats, actors)

    losses = []
    for result in results:
        if isinstance(result, dict):
            losses.extend(result.get("losses", []))
        elif result:
            losses.extend(result)
    return losses


def piper_measure_dags(
    loss_fn,
    *,
    iterations: int = 30,
    iteration_sleep: float = 0.0,
    log_stats: bool = True,
    collect_peak_memory: bool = True,
) -> TimingResult:
    """Measure exactly ``iterations`` normal training steps.

    Every data-parallel worker receives the same iteration count from the
    distributed program, preserving collective operation alignment.
    """

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    actors = piper_metadata.actors
    if not actors:
        raise RuntimeError("piper_setup() must create actors before timing")

    step_samples_s: list[float] = []
    peak_samples: dict[int, list[int]] = {}
    for _ in range(iterations):
        if collect_peak_memory:
            ray.get([
                actor.reset_peak_memory.remote()
                for actor in actors.values()
            ])
        started = time.perf_counter()
        piper_exec_dag(loss_fn, log_stats=log_stats)
        step_samples_s.append(time.perf_counter() - started)
        if collect_peak_memory:
            for rank, peak_bytes in ray.get([
                actor.get_and_reset_peak_memory_stats.remote()
                for actor in actors.values()
            ]):
                peak_samples.setdefault(int(rank), []).append(int(peak_bytes))
        if iteration_sleep > 0.0:
            time.sleep(iteration_sleep)

    summary = summarize_timing_samples(step_samples_s, estimator="median")
    logger.info(
        "hardware timing samples=%d, local 95%% CI error=%.3f%%",
        len(step_samples_s),
        summary.error_pct,
    )
    return TimingResult(
        step_samples_s=tuple(step_samples_s),
        peak_memory_samples_by_rank={
            rank: tuple(samples) for rank, samples in sorted(peak_samples.items())
        },
    )


def piper_profile_dags(
    loss_fn,
    config: ProfileConfig | None = None,
    *,
    profile_modes: Sequence[str] | None = None,
    profile_configs: Mapping[str, ProfileConfig] | None = None,
    output_path: str | None = None,
) -> ProfiledTrainingPlan:
    """Collect and merge one or more timing modes across every actor.

    Every DP worker must call this function as part of the same distributed
    program. Each worker collects the exact configured iteration count.
    """
    base_config = config or ProfileConfig()
    requested_modes = tuple(
        normalize_profile_mode(mode)
        for mode in (
            profile_modes
            if profile_modes is not None
            else tuple(profile_configs or (INTERFERENCE_FREE_PROFILE_MODE,))
        )
    )
    if not requested_modes:
        raise ValueError("piper_profile_dags requires at least one profile mode")
    if len(set(requested_modes)) != len(requested_modes):
        raise ValueError(f"duplicate profile modes requested: {requested_modes!r}")
    configs: dict[str, ProfileConfig] = {}
    for mode in requested_modes:
        selected = (profile_configs or {}).get(mode, base_config)
        if mode == NORMAL_PROFILE_MODE and (
            selected.isolation_scope != DEFAULT_NORMAL_SCOPE
        ):
            selected = replace(selected, isolation_scope=DEFAULT_NORMAL_SCOPE)
        configs[mode] = selected

    actors = piper_metadata.actors
    if not actors:
        raise RuntimeError("piper_setup() must create actors before profiling")
    if not piper_metadata.per_pp_training_dags:
        raise RuntimeError("piper_setup() must finalize TrainingDAGs before profiling")

    store = piper_metadata.compiled_data_store
    if store is None:
        from .compile import _COMPILED_DATA_ACTOR

        store = ray.get_actor(_COMPILED_DATA_ACTOR)
        piper_metadata.compiled_data_store = store

    dp_rank = int(os.environ["PIPER_DP_RANK"])
    dp_degree = int(os.environ["PIPER_DP_DEGREE"])
    if dp_rank == 0:
        generation = ray.get(store.begin_profile_generation.remote())
    else:
        previous_generation = int(
            getattr(piper_metadata, "profile_generation", 0)
        )
        deadline = time.monotonic() + 600.0
        generation = ray.get(store.current_profile_generation.remote())
        while generation <= previous_generation:
            if time.monotonic() > deadline:
                raise TimeoutError("timed out waiting for profile generation to begin")
            time.sleep(0.05)
            generation = ray.get(store.current_profile_generation.remote())
    piper_metadata.profile_generation = generation

    local_results = []
    local_step_samples_us: dict[str, list[float]] = {}
    for mode in requested_modes:
        mode_config = configs[mode]
        ray.get([
            actor.reset_profile_samples.remote(profile_mode=mode)
            for actor in actors.values()
        ])
        for _ in range(mode_config.warmup_iters):
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    record=False,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])

        step_samples_us: list[float] = []
        for _ in range(mode_config.repeat_iters):
            started = time.perf_counter()
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    record=True,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])
            step_samples_us.append((time.perf_counter() - started) * 1_000_000.0)

        local_step_samples_us[mode] = step_samples_us
        local_results.extend(
            ray.get([
                actor.get_profile_result.remote(
                    estimator=mode_config.estimator,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])
        )

    ray.get(
        store.publish_profile_shard.remote(
            generation,
            dp_rank,
            {
                "results": local_results,
                "step_samples_us": local_step_samples_us,
            },
        )
    )
    deadline = time.monotonic() + 600.0
    while not ray.get(
        store.profile_shards_ready.remote(generation, dp_degree)
    ):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"timed out waiting for all DP profile shards for generation {generation}"
            )
        time.sleep(0.05)

    shards = ray.get(store.get_profile_shards.remote(generation))
    all_results = [
        result
        for shard_rank in sorted(shards)
        for result in shards[shard_rank]["results"]
    ]

    # A later profile generation replaces, rather than mixes with, old samples.
    for dag in piper_metadata.per_pp_training_dags:
        dag.prologue_profile_times_by_mode.clear()
        for node in dag.nodes.values():
            node.profile_times_by_mode.clear()
    plan = merge_actor_profiles(
        piper_metadata.per_pp_training_dags,
        all_results,
        profile_configs=configs,
        topology=dict(getattr(piper_metadata, "schedule_info", {}) or {}),
    )
    step_samples_by_mode_by_dp = {
        mode: {
            str(shard_rank): list(shards[shard_rank]["step_samples_us"][mode])
            for shard_rank in sorted(shards)
        }
        for mode in requested_modes
    }
    plan.profile_metadata["instrumented_step_samples_us_by_mode_by_dp"] = (
        step_samples_by_mode_by_dp
    )
    plan.profile_metadata["instrumented_global_step_samples_us_by_mode"] = {
        mode: [
            max(float(per_dp[str(dp)][sample_index]) for dp in sorted(shards))
            for sample_index in range(
                min(len(per_dp[str(dp)]) for dp in sorted(shards))
            )
        ]
        for mode, per_dp in step_samples_by_mode_by_dp.items()
    }
    piper_metadata.profiled_training_plan = plan
    if output_path is not None and dp_rank == 0:
        plan.save_json(output_path)
    return plan


def simulate_profiled_dags(
    plan: ProfiledTrainingPlan | None = None,
    *,
    profile_mode: str = INTERFERENCE_FREE_PROFILE_MODE,
    **simulation_kwargs,
):
    """Run the stream simulator using one named duration dataset."""
    if plan is None:
        plan = getattr(piper_metadata, "profiled_training_plan", None)
    if plan is None:
        raise RuntimeError("piper_profile_dags() must run before simulation")
    from .simulator import simulate_training_step

    return simulate_training_step(
        plan, profile_mode=profile_mode, **simulation_kwargs
    )


def simulate_profiled_modes(
    plan: ProfiledTrainingPlan | None = None,
    *,
    actual_step_samples_s: Sequence[float] | None = None,
    **simulation_kwargs,
):
    """Replay interference-free/normal timings and validate normal replay."""

    if plan is None:
        plan = getattr(piper_metadata, "profiled_training_plan", None)
    if plan is None:
        raise RuntimeError("piper_profile_dags() must run before simulation")
    from .simulator import simulate_profile_modes

    actual_us = (
        None
        if actual_step_samples_s is None
        else tuple(float(sample) * 1_000_000.0 for sample in actual_step_samples_s)
    )
    return simulate_profile_modes(
        plan,
        actual_step_samples_us=actual_us,
        **simulation_kwargs,
    )


def _log_step_stats(step_time: float, log_memory: bool, actors: dict) -> None:
    stats = [f"step_time={step_time:.3f}s"]
    tokens = getattr(piper_metadata, "tokens_per_step", None)
    if tokens is not None:
        stats.append(f"throughput={tokens / step_time:.1f} tok/s")
    logger.info("  ".join(stats))
