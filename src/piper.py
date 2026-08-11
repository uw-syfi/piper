import time
import os
import json
from pathlib import Path
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator, Mapping, Sequence

import ray
import torch
import torch.fx as fx
from torch._dynamo.backends.registry import register_backend

from .fx import (
    EP_COMBINE_PAYLOAD,
    EP_DISPATCH_COUNTS,
    EP_DISPATCH_PAYLOAD,
    PIPER_ANNOTATIONS_META_KEY,
    PIPER_EP_BOUNDARY_META_KEY,
    split_gm_by_annotations,
)
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
    estimate_profile_time,
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


def _mark_ep_boundary(tensor: torch.Tensor, role: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Piper EP boundary markers require a torch.Tensor, got {type(tensor)!r}"
        )
    with fx.traceback.annotate({PIPER_EP_BOUNDARY_META_KEY: role}):
        return torch.ops.aten.alias.default(tensor)


def mark_dispatch_payload(tensor: torch.Tensor) -> torch.Tensor:
    """Mark the routed activation entering a Piper EP region."""
    return _mark_ep_boundary(tensor, EP_DISPATCH_PAYLOAD)


def mark_dispatch_counts(tensor: torch.Tensor) -> torch.Tensor:
    """Mark the per-expert token counts entering a Piper EP region."""
    return _mark_ep_boundary(tensor, EP_DISPATCH_COUNTS)


def mark_combine_payload(tensor: torch.Tensor) -> torch.Tensor:
    """Mark the expert activation leaving a Piper EP region."""
    return _mark_ep_boundary(tensor, EP_COMBINE_PAYLOAD)


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


def piper_exec_dag(
    loss_fn, log_stats: bool = False, *, return_losses: bool = True
) -> list:
    """Execute one training step using the loaded per-rank TrainingDAG."""
    actors = piper_metadata.actors
    run_refs = [
        actor.run_dag.remote(loss_fn=loss_fn, return_losses=return_losses)
        for actor in actors.values()
    ]
    t0 = time.perf_counter()
    results = ray.get(run_refs)
    step_time = time.perf_counter() - t0

    if log_stats:
        _log_step_stats(step_time, log_stats, actors)

    if not return_losses:
        return []
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
        piper_exec_dag(
            loss_fn,
            log_stats=log_stats,
            return_losses=False,
        )
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


def _piper_profile_cuda_event_dags(
    loss_fn,
    config: ProfileConfig | None = None,
    *,
    profile_modes: Sequence[str] | None = None,
    profile_configs: Mapping[str, ProfileConfig] | None = None,
    output_path: str | None = None,
    measure_uninstrumented_modes: bool = False,
) -> ProfiledTrainingPlan:
    """Legacy CUDA-event implementation of multi-mode profiling.

    Every DP worker must call this function as part of the same distributed
    program.  ``repeat_iters`` is exact unless a CI target is configured, in
    which case it is the minimum and each worker may sample up to its cap.
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

    # Build exact payload specs from the forward receivers, then reuse the
    # same activation shape for the reverse backward transfer.  A P2P edge is
    # measured outside the training DAG with both endpoints ready, so its
    # duration contains transfer work rather than pipeline rendezvous delay.
    receive_spec_shards = ray.get([
        actor.p2p_receive_specs.remote() for actor in actors.values()
    ])
    forward_specs_by_edge: dict[tuple[str, str], list[tuple[tuple[int, ...], str]]] = {}
    forward_specs_by_comm_id: dict[str, list[tuple[tuple[int, ...], str]]] = {}
    for shard in receive_spec_shards:
        for comm_id, specs in shard.items():
            forward_specs_by_comm_id.setdefault(comm_id, specs)
    for dag in piper_metadata.per_pp_training_dags:
        for node in dag.nodes.values():
            if (
                node.node_kind == "RECV_COMM"
                and node.tag.get("PASS") == "F"
                and str(node.node_meta.get("comm_id")) in forward_specs_by_comm_id
            ):
                forward_specs_by_edge[(
                    str(node.node_meta["from_uid"]), str(node.node_meta["to_uid"])
                )] = forward_specs_by_comm_id[str(node.node_meta["comm_id"])]
    p2p_specs_by_comm_id: dict[str, list[tuple[tuple[int, ...], str]]] = {}
    for dag in piper_metadata.per_pp_training_dags:
        for node in dag.nodes.values():
            if node.node_kind not in {"SEND_COMM", "RECV_COMM"}:
                continue
            comm_id = str(node.node_meta["comm_id"])
            if node.tag.get("PASS") == "F":
                specs = forward_specs_by_comm_id.get(comm_id)
            else:
                source = str(node.node_meta["from_uid"]).removesuffix(".bwd")
                target = str(node.node_meta["to_uid"]).removesuffix(".bwd")
                specs = forward_specs_by_edge.get((target, source))
            if specs is None:
                raise RuntimeError(f"missing receiver payload metadata for P2P {comm_id}")
            p2p_specs_by_comm_id[comm_id] = specs
    # Bucketing can leave large unused cached blocks after the measured
    # training steps.  Release them before allocating the standalone P2P
    # payloads so the microprofile does not fail from allocator fragmentation.
    # This happens outside every measured interval and does not affect either
    # policy's recorded node times.
    ray.get([
        actor.clear_profile_allocator_cache.remote()
        for actor in actors.values()
    ])
    local_p2p_baseline_shards = ray.get([
        actor.profile_p2p_transfer_baselines.remote(p2p_specs_by_comm_id)
        for actor in actors.values()
    ])
    local_p2p_baselines = {
        comm_id: max(
            values[comm_id] for values in local_p2p_baseline_shards
            if comm_id in values
        )
        for comm_id in p2p_specs_by_comm_id
    }

    local_results = []
    local_step_samples_us: dict[str, list[float]] = {}
    for mode in requested_modes:
        mode_config = configs[mode]
        # Give every actor a mapping to the same coordinator-owned monotonic
        # clock before collecting CUDA timeline origins for this mode.
        ray.get([
            actor.calibrate_profile_clock.remote(store)
            for actor in actors.values()
        ])
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

        # Memory-tight workloads can leave large, fragmented cached regions
        # after warmup.  Release only unused blocks before starting the timed
        # series; this call is outside every measured interval.
        ray.get([
            actor.clear_profile_allocator_cache.remote()
            for actor in actors.values()
        ])

        # The first event-instrumented iteration after clock synchronization
        # and allocator-cache release consistently includes one-time CUDA/NCCL
        # prologue work.  Execute it so the steady-state series starts from the
        # same state, then explicitly discard both node and wall-clock samples.
        ray.get([
            actor.profile_run_dag.remote(
                loss_fn=loss_fn,
                record=True,
                profile_mode=mode,
            )
            for actor in actors.values()
        ])
        ray.get([
            actor.reset_profile_samples.remote(profile_mode=mode)
            for actor in actors.values()
        ])
        ray.get([
            actor.calibrate_profile_clock.remote(store)
            for actor in actors.values()
        ])

        step_samples_us: list[float] = []
        max_repeat_iters = (
            mode_config.repeat_iters
            if mode_config.max_repeat_iters is None
            else mode_config.max_repeat_iters
        )
        for _ in range(max_repeat_iters):
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
            if (
                mode_config.target_ci95_error_pct is not None
                and len(step_samples_us) >= mode_config.repeat_iters
            ):
                statistics = summarize_timing_samples(
                    step_samples_us, estimator=mode_config.estimator
                )
                converged = statistics.error_pct <= mode_config.target_ci95_error_pct
                globally_converged = ray.get(
                    store.submit_profile_convergence.remote(
                        generation,
                        mode,
                        len(step_samples_us),
                        dp_rank,
                        converged,
                        dp_degree,
                    )
                )
                while globally_converged is None:
                    time.sleep(0.01)
                    globally_converged = ray.get(
                        store.get_profile_convergence.remote(
                            generation, mode, len(step_samples_us)
                        )
                    )
                if globally_converged:
                    logger.info(
                        "profile mode=%s converged across DP after %d samples: local 95%% CI error=%.3f%% (target %.3f%%)",
                        mode,
                        len(step_samples_us),
                        statistics.error_pct,
                        mode_config.target_ci95_error_pct,
                    )
                    break

        if mode_config.target_ci95_error_pct is not None:
            statistics = summarize_timing_samples(
                step_samples_us, estimator=mode_config.estimator
            )
            logger.info(
                "profile mode=%s retained %d samples: 95%% CI error=%.3f%% (target %.3f%%)",
                mode,
                len(step_samples_us),
                statistics.error_pct,
                mode_config.target_ci95_error_pct,
            )

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

    # CUDA events around every node perturb launch timing and can materially
    # change overlap.  When requested, measure the two stream policies again
    # without per-node events.  Alternate their order on every sample so
    # allocator/cache drift is balanced instead of being attributed to one
    # mode.  Actor-side synchronization makes each driver wall sample cover
    # the work just launched rather than the preceding asynchronous step.
    local_uninstrumented_step_samples_us: dict[str, list[float]] = {}
    if measure_uninstrumented_modes:
        local_uninstrumented_step_samples_us = {
            mode: [] for mode in requested_modes
        }
        max_warmups = max(configs[mode].warmup_iters for mode in requested_modes)
        for warmup_index in range(max_warmups):
            order = (
                requested_modes
                if warmup_index % 2 == 0
                else tuple(reversed(requested_modes))
            )
            for mode in order:
                if warmup_index >= configs[mode].warmup_iters:
                    continue
                ray.get([
                    actor.profile_run_dag.remote(
                        loss_fn=loss_fn,
                        record=False,
                        synchronize=True,
                        profile_mode=mode,
                    )
                    for actor in actors.values()
                ])

        max_repeats = max(configs[mode].repeat_iters for mode in requested_modes)
        for sample_index in range(max_repeats):
            order = (
                requested_modes
                if sample_index % 2 == 0
                else tuple(reversed(requested_modes))
            )
            for mode in order:
                if sample_index >= configs[mode].repeat_iters:
                    continue
                started = time.perf_counter()
                ray.get([
                    actor.profile_run_dag.remote(
                        loss_fn=loss_fn,
                        record=False,
                        synchronize=True,
                        profile_mode=mode,
                    )
                    for actor in actors.values()
                ])
                local_uninstrumented_step_samples_us[mode].append(
                    (time.perf_counter() - started) * 1_000_000.0
                )

    ray.get(
        store.publish_profile_shard.remote(
            generation,
            dp_rank,
            {
                "results": local_results,
                "step_samples_us": local_step_samples_us,
                "uninstrumented_step_samples_us": (
                    local_uninstrumented_step_samples_us
                ),
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
    if measure_uninstrumented_modes:
        uninstrumented_by_mode_by_dp = {
            mode: {
                str(shard_rank): list(
                    shards[shard_rank]["uninstrumented_step_samples_us"][mode]
                )
                for shard_rank in sorted(shards)
            }
            for mode in requested_modes
        }
        plan.profile_metadata[
            "uninstrumented_step_samples_us_by_mode_by_dp"
        ] = uninstrumented_by_mode_by_dp
        plan.profile_metadata[
            "uninstrumented_global_step_samples_us_by_mode"
        ] = {
            mode: [
                max(
                    float(per_dp[str(dp)][sample_index])
                    for dp in sorted(shards)
                )
                for sample_index in range(
                    min(len(per_dp[str(dp)]) for dp in sorted(shards))
                )
            ]
            for mode, per_dp in uninstrumented_by_mode_by_dp.items()
        }
    piper_metadata.profiled_training_plan = plan
    if output_path is not None and dp_rank == 0:
        plan.save_json(output_path)
    return plan


def _compare_kineto_activity_signatures(results: Sequence[Any]) -> dict[str, Any]:
    """Compare exact per-node-range GPU activity sequences across IF/normal."""

    by_mode_rank = {
        (normalize_profile_mode(result.profile_mode), int(result.global_rank)): result
        for result in results
    }
    if_ranks = {
        rank for mode, rank in by_mode_rank
        if mode == INTERFERENCE_FREE_PROFILE_MODE
    }
    normal_ranks = {
        rank for mode, rank in by_mode_rank
        if mode == NORMAL_PROFILE_MODE
    }
    common_ranks = sorted(if_ranks & normal_ranks)
    summary: dict[str, Any] = {
        "comparison": (
            "exact ordered (GPU activity category, Kineto name) sequence and "
            "multiplicity per node-range occurrence"
        ),
        "actors_compared": len(common_ranks),
        "nodes_compared": 0,
        "node_ranges_compared": 0,
        "gpu_activities_compared": 0,
        "mismatched_nodes": 0,
        "mismatched_node_ranges": 0,
        "missing_mode_ranks": {
            INTERFERENCE_FREE_PROFILE_MODE: sorted(normal_ranks - if_ranks),
            NORMAL_PROFILE_MODE: sorted(if_ranks - normal_ranks),
        },
        "mismatches": [],
    }

    def activity_counts(occurrence: Mapping[str, Any]) -> dict[tuple[str, str], int]:
        return {
            (str(item["category"]), str(item["name"])): int(item["count"])
            for item in occurrence.get("activities", ())
        }

    for rank in common_ranks:
        mode_audits = {}
        for mode in (INTERFERENCE_FREE_PROFILE_MODE, NORMAL_PROFILE_MODE):
            result = by_mode_rank[(mode, rank)]
            mode_audits[mode] = result.device_metadata.get(
                "kineto_attribution_audit", {}
            )
        node_uids = sorted(set(mode_audits[INTERFERENCE_FREE_PROFILE_MODE]) | set(
            mode_audits[NORMAL_PROFILE_MODE]
        ))
        for uid in node_uids:
            if_audit = mode_audits[INTERFERENCE_FREE_PROFILE_MODE].get(uid)
            normal_audit = mode_audits[NORMAL_PROFILE_MODE].get(uid)
            summary["nodes_compared"] += 1
            if if_audit is None or normal_audit is None:
                summary["mismatched_nodes"] += 1
                summary["mismatches"].append({
                    "global_rank": rank,
                    "uid": uid,
                    "reason": "node audit missing from one profile mode",
                    "if_present": if_audit is not None,
                    "normal_present": normal_audit is not None,
                })
                continue
            if_ranges = list(if_audit.get("range_activity_signatures", ()))
            normal_ranges = list(normal_audit.get("range_activity_signatures", ()))
            range_mismatches = []
            for occurrence in range(max(len(if_ranges), len(normal_ranges))):
                if_range = if_ranges[occurrence] if occurrence < len(if_ranges) else None
                normal_range = (
                    normal_ranges[occurrence]
                    if occurrence < len(normal_ranges)
                    else None
                )
                if if_range is not None and normal_range is not None:
                    summary["node_ranges_compared"] += 1
                    summary["gpu_activities_compared"] += int(
                        if_range.get("activity_count", 0)
                    )
                    if (
                        if_range.get("signature_sha256")
                        == normal_range.get("signature_sha256")
                        and if_range.get("activity_sequence_sha256")
                        == normal_range.get("activity_sequence_sha256")
                    ):
                        continue
                summary["mismatched_node_ranges"] += 1
                if_counts = {} if if_range is None else activity_counts(if_range)
                normal_counts = (
                    {} if normal_range is None else activity_counts(normal_range)
                )
                deltas = []
                for category, name in sorted(set(if_counts) | set(normal_counts)):
                    if_count = if_counts.get((category, name), 0)
                    normal_count = normal_counts.get((category, name), 0)
                    if if_count != normal_count:
                        deltas.append({
                            "category": category,
                            "name": name,
                            "if_count": if_count,
                            "normal_count": normal_count,
                        })
                range_mismatches.append({
                    "occurrence": occurrence,
                    "if_activity_count": (
                        None if if_range is None
                        else int(if_range.get("activity_count", 0))
                    ),
                    "normal_activity_count": (
                        None if normal_range is None
                        else int(normal_range.get("activity_count", 0))
                    ),
                    "if_signature_sha256": (
                        None if if_range is None
                        else if_range.get("signature_sha256")
                    ),
                    "normal_signature_sha256": (
                        None if normal_range is None
                        else normal_range.get("signature_sha256")
                    ),
                    "if_activity_sequence_sha256": (
                        None if if_range is None
                        else if_range.get("activity_sequence_sha256")
                    ),
                    "normal_activity_sequence_sha256": (
                        None if normal_range is None
                        else normal_range.get("activity_sequence_sha256")
                    ),
                    "activity_order_match": (
                        False
                        if if_range is None or normal_range is None
                        else if_range.get("activity_sequence_sha256")
                        == normal_range.get("activity_sequence_sha256")
                    ),
                    "different_activity_entries": len(deltas),
                    "activity_deltas": deltas[:25],
                })
            if range_mismatches:
                summary["mismatched_nodes"] += 1
                summary["mismatches"].append({
                    "global_rank": rank,
                    "uid": uid,
                    "if_range_count": len(if_ranges),
                    "normal_range_count": len(normal_ranges),
                    "ranges": range_mismatches,
                })

    summary["exact_match"] = (
        not any(summary["missing_mode_ranks"].values())
        and summary["mismatched_nodes"] == 0
    )
    # The exact activity lists can be large.  Retain deterministic per-range
    # hashes in the saved plan after the central comparison has consumed them.
    for result in results:
        for audit in result.device_metadata.get(
            "kineto_attribution_audit", {}
        ).values():
            for occurrence in audit.get("range_activity_signatures", ()):
                occurrence.pop("activities", None)
                occurrence.pop("activity_sequence", None)
    return summary


def _filter_kineto_profile_iterations(
    results: Sequence[Any], iteration_indices: Sequence[int]
) -> list[Any]:
    """Retain the same fully paired Kineto iterations in every actor shard."""

    selected = tuple(int(index) for index in iteration_indices)
    filtered_results = []
    for result in results:
        filtered_profiles = {}
        for uid, profile in result.node_profiles.items():
            if not profile.executed:
                filtered_profiles[uid] = profile
                continue
            samples = tuple(profile.samples_us[index] for index in selected)
            offsets = (
                tuple(profile.start_offset_samples_us[index] for index in selected)
                if profile.start_offset_samples_us
                else ()
            )
            filtered_profiles[uid] = replace(
                profile,
                samples_us=samples,
                sample_iteration_ids=tuple(range(len(samples))),
                estimate_us=estimate_profile_time(samples, profile.estimator),
                start_offset_samples_us=offsets,
                start_offset_estimate_us=(
                    estimate_profile_time(offsets, profile.estimator)
                    if offsets
                    else None
                ),
            )

        metadata = dict(result.device_metadata)
        filtered_audits = {}
        for uid, raw_audit in metadata.get(
            "kineto_attribution_audit", {}
        ).items():
            audit = dict(raw_audit)
            signatures = list(audit.get("range_activity_signatures", ()))
            durations = list(audit.get("simulator_duration_samples_us", ()))
            audit["range_activity_signatures"] = [
                signatures[index] for index in selected
            ]
            audit["simulator_duration_samples_us"] = [
                durations[index] for index in selected
            ]
            audit["range_count"] = len(selected)
            filtered_audits[uid] = audit
        metadata["kineto_attribution_audit"] = filtered_audits
        allocator_stats = list(metadata.get("allocator_iteration_stats", ()))
        if allocator_stats and max(selected, default=-1) < len(allocator_stats):
            metadata["allocator_iteration_stats"] = [
                allocator_stats[index] for index in selected
            ]
        filtered_results.append(
            replace(
                result,
                node_profiles=filtered_profiles,
                device_metadata=metadata,
            )
        )
    return filtered_results


def piper_profile_dags(
    loss_fn,
    config: ProfileConfig | None = None,
    *,
    profile_modes: Sequence[str] | None = None,
    profile_configs: Mapping[str, ProfileConfig] | None = None,
    output_path: str | None = None,
    measure_uninstrumented_modes: bool = False,
    trace_output_dir: str | None = None,
    trace_dp_ranks: Sequence[int] | None = None,
) -> ProfiledTrainingPlan:
    """Profile Piper nodes with Kineto/CUPTI and merge a replay plan.

    Piper intentionally does *not* insert CUDA timing events around individual
    nodes here: those events alter the concurrent-stream schedule being
    measured.  Kineto correlates GPU activity to the existing node
    ``record_function`` ranges.  The simulator subsequently derives every
    start from its DAG dependencies and stream FIFO ordering; recorded launch
    offsets are neither collected nor replayed.
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
    if not requested_modes or len(set(requested_modes)) != len(requested_modes):
        raise ValueError(f"invalid requested profile modes: {requested_modes!r}")
    configs: dict[str, ProfileConfig] = {}
    for mode in requested_modes:
        selected = (profile_configs or {}).get(mode, base_config)
        if mode == NORMAL_PROFILE_MODE and selected.isolation_scope != DEFAULT_NORMAL_SCOPE:
            selected = replace(selected, isolation_scope=DEFAULT_NORMAL_SCOPE)
        configs[mode] = selected

    actors = piper_metadata.actors
    if not actors or not piper_metadata.per_pp_training_dags:
        raise RuntimeError("piper_setup() must finalize actors and TrainingDAGs before profiling")
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
        previous = int(getattr(piper_metadata, "profile_generation", 0))
        deadline = time.monotonic() + 600.0
        generation = ray.get(store.current_profile_generation.remote())
        while generation <= previous:
            if time.monotonic() > deadline:
                raise TimeoutError("timed out waiting for Kineto profile generation")
            time.sleep(0.05)
            generation = ray.get(store.current_profile_generation.remote())
    piper_metadata.profile_generation = generation

    receive_spec_shards = ray.get([
        actor.p2p_receive_specs.remote() for actor in actors.values()
    ])
    forward_specs_by_comm_id: dict[str, list[tuple[tuple[int, ...], str]]] = {}
    for shard in receive_spec_shards:
        for comm_id, specs in shard.items():
            forward_specs_by_comm_id.setdefault(comm_id, specs)
    forward_specs_by_edge: dict[tuple[str, str], list[tuple[tuple[int, ...], str]]] = {}
    forward_uid_by_backward_uid: dict[str, str] = {}
    for dag in piper_metadata.per_pp_training_dags:
        for node in dag.nodes.values():
            if node.node_kind == "COMPUTE" and node.compute_subkind in {"BWD", "BWD_I", "BWD_W"}:
                fwd_uid = node.node_meta.get("fwd_uid")
                if fwd_uid is not None:
                    forward_uid_by_backward_uid[node.uid] = str(fwd_uid)
            comm_id = str(node.node_meta.get("comm_id", ""))
            if node.node_kind == "RECV_COMM" and node.tag.get("PASS") == "F" and comm_id in forward_specs_by_comm_id:
                forward_specs_by_edge[(str(node.node_meta["from_uid"]), str(node.node_meta["to_uid"]))] = forward_specs_by_comm_id[comm_id]
    p2p_specs_by_comm_id: dict[str, list[tuple[tuple[int, ...], str]]] = {}
    for dag in piper_metadata.per_pp_training_dags:
        for node in dag.nodes.values():
            if node.node_kind not in {"SEND_COMM", "RECV_COMM"}:
                continue
            comm_id = str(node.node_meta["comm_id"])
            if node.tag.get("PASS") == "F":
                specs = forward_specs_by_comm_id.get(comm_id)
            else:
                source = forward_uid_by_backward_uid.get(str(node.node_meta["from_uid"]))
                target = forward_uid_by_backward_uid.get(str(node.node_meta["to_uid"]))
                specs = (
                    None if source is None or target is None
                    else forward_specs_by_edge.get((target, source))
                )
            if specs is None:
                raise RuntimeError(f"missing receiver payload metadata for P2P {comm_id}")
            p2p_specs_by_comm_id[comm_id] = specs
    # Bucketing can leave large unused cached blocks after the measured
    # training steps.  Release them before allocating the standalone P2P
    # payloads so the microprofile does not fail from allocator fragmentation.
    # This happens outside every measured interval and does not affect either
    # policy's recorded node times.
    ray.get([
        actor.clear_profile_allocator_cache.remote()
        for actor in actors.values()
    ])
    local_p2p_baseline_shards = ray.get([
        actor.profile_p2p_transfer_baselines.remote(p2p_specs_by_comm_id)
        for actor in actors.values()
    ])
    local_p2p_baselines = {
        comm_id: max(shard[comm_id] for shard in local_p2p_baseline_shards if comm_id in shard)
        for comm_id in p2p_specs_by_comm_id
    }

    local_results = []
    local_instrumented_step_samples_us: dict[str, tuple[float, ...]] = {}
    requested_repeat_counts = {
        int(configs[mode].repeat_iters) for mode in requested_modes
    }
    # CUPTI can very occasionally omit one activity in an otherwise complete
    # node range.  Capture a few spare whole steps so those corrupt paired
    # samples can be rejected without weakening exact activity validation.
    kineto_spare_iters = (
        6
        if set(requested_modes)
        == {INTERFERENCE_FREE_PROFILE_MODE, NORMAL_PROFILE_MODE}
        and len(requested_repeat_counts) == 1
        else 0
    )
    # Every policy must profile the same training state.  Restoring only at a
    # policy boundary is insufficient for data-dependent MoE kernels: small
    # numerical differences between the normal and IF schedules accumulate
    # across optimizer steps, alter expert token counts, and make grouped GEMM
    # dispatch different CUTLASS kernels.  Capture once, then restore before
    # every warmup/profile occurrence outside its timed interval.  This makes
    # each occurrence a repeat measurement of the same logical CUDA work.
    ray.get([
        actor.capture_profile_training_state.remote()
        for actor in actors.values()
    ])
    for mode in requested_modes:
        mode_config = configs[mode]
        ray.get([
            actor.restore_profile_training_state.remote()
            for actor in actors.values()
        ])
        ray.get([actor.synchronize_device.remote() for actor in actors.values()])
        ray.get([
            actor.start_cupti_node_profiler.remote(
                warmup_iters=mode_config.warmup_iters,
                active_iters=mode_config.repeat_iters + kineto_spare_iters,
            )
            for actor in actors.values()
        ])
        mode_step_samples_us: list[float] = []
        for iteration in range(
            mode_config.warmup_iters
            + mode_config.repeat_iters
            + kineto_spare_iters
        ):
            ray.get([
                actor.restore_profile_training_state.remote()
                for actor in actors.values()
            ])
            active = iteration >= mode_config.warmup_iters
            if active:
                ray.get([
                    actor.synchronize_device.remote()
                    for actor in actors.values()
                ])
                started = time.perf_counter()
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    record=False,
                    synchronize_before=not active,
                    profile_mode=mode,
                    advance_profiler=False,
                    # Allocator snapshots are diagnostics, not part of the
                    # represented training step.  In particular,
                    # memory_snapshot() can dominate short-kernel schedules.
                    capture_allocator_stats=False,
                )
                for actor in actors.values()
            ])
            if active:
                ray.get([
                    actor.synchronize_device.remote()
                    for actor in actors.values()
                ])
                mode_step_samples_us.append(
                    (time.perf_counter() - started) * 1_000_000.0
                )
            # Keep the profiler state transition and final Kineto flush outside
            # the same-ID synchronized step interval used by validation 4.
            ray.get([
                actor.step_pytorch_profiler.remote()
                for actor in actors.values()
            ])
        local_instrumented_step_samples_us[mode] = tuple(
            mode_step_samples_us
        )
        local_results.extend(ray.get([
            actor.stop_cupti_profile_result.remote(
                estimator=mode_config.estimator,
                profile_mode=mode,
                p2p_baselines_by_comm_id=local_p2p_baselines,
                trace_output_dir=trace_output_dir,
                trace_dp_ranks=(
                    None
                    if trace_dp_ranks is None
                    else tuple(int(rank) for rank in trace_dp_ranks)
                ),
            )
            for actor in actors.values()
        ]))

    # Profiling is observational: leave the actors at the state at which the
    # paired capture began rather than after the final policy's optimizer steps.
    ray.get([
        actor.restore_profile_training_state.remote()
        for actor in actors.values()
    ])

    # Keep policy-level wall-clock measurements separate from node timing.
    # They are diagnostics only; node durations above remain CUPTI-derived.
    local_uninstrumented_step_samples_us: dict[str, tuple[float, ...]] = {}
    if measure_uninstrumented_modes:
        local_uninstrumented_step_samples_us = piper_measure_profile_modes(
            loss_fn,
            warmup_iters=max(configs[mode].warmup_iters for mode in requested_modes),
            repeat_iters=max(configs[mode].repeat_iters for mode in requested_modes),
            profile_modes=requested_modes,
        )

    ray.get(store.publish_profile_shard.remote(
        generation, dp_rank,
        {
            "results": local_results,
            "step_samples_us": local_instrumented_step_samples_us,
            "uninstrumented_step_samples_us": local_uninstrumented_step_samples_us,
        },
    ))
    deadline = time.monotonic() + 600.0
    while not ray.get(store.profile_shards_ready.remote(generation, dp_degree)):
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for profile generation {generation}")
        time.sleep(0.05)
    shards = ray.get(store.get_profile_shards.remote(generation))
    all_results = [
        result for shard_rank in sorted(shards)
        for result in shards[shard_rank]["results"]
    ]
    raw_activity_equivalence = _compare_kineto_activity_signatures(all_results)
    rejected_iteration_ids = sorted({
        int(item["occurrence"])
        for mismatch in raw_activity_equivalence.get("mismatches", ())
        for item in mismatch.get("ranges", ())
        if item.get("occurrence") is not None
    })
    unfilterable_mismatches = [
        mismatch
        for mismatch in raw_activity_equivalence.get("mismatches", ())
        if not mismatch.get("ranges")
    ]
    requested_iterations = (
        next(iter(requested_repeat_counts))
        if len(requested_repeat_counts) == 1
        else 0
    )
    captured_iterations = min(
        (
            len(profile.samples_us)
            for result in all_results
            for profile in result.node_profiles.values()
            if profile.executed
        ),
        default=0,
    )
    retained_iteration_ids = [
        index
        for index in range(captured_iterations)
        if index not in set(rejected_iteration_ids)
    ][:requested_iterations]
    can_filter = (
        kineto_spare_iters > 0
        and not any(raw_activity_equivalence["missing_mode_ranks"].values())
        and not unfilterable_mismatches
        and len(retained_iteration_ids) == requested_iterations
    )
    if can_filter:
        all_results = _filter_kineto_profile_iterations(
            all_results, retained_iteration_ids
        )
        activity_equivalence = _compare_kineto_activity_signatures(all_results)
    else:
        retained_iteration_ids = list(range(captured_iterations))
        activity_equivalence = raw_activity_equivalence
    from .profile_validation import validate_kineto_profile_results

    kineto_validation = validate_kineto_profile_results(
        all_results,
        activity_equivalence,
        raise_on_error=False,
    )
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
    plan.profile_metadata["timing_source"] = "kineto_cupti"
    plan.profile_metadata["kineto_activity_equivalence"] = activity_equivalence
    plan.profile_metadata["kineto_sample_rejection"] = {
        "captured_iterations": captured_iterations,
        "requested_iterations": requested_iterations,
        "spare_iterations": kineto_spare_iters,
        "rejected_iteration_ids": rejected_iteration_ids,
        "retained_iteration_ids": retained_iteration_ids,
        "unused_spare_iteration_ids": [
            index
            for index in range(captured_iterations)
            if index not in set(rejected_iteration_ids)
            and index not in set(retained_iteration_ids)
        ],
        "raw_mismatched_node_ranges": int(
            raw_activity_equivalence.get("mismatched_node_ranges", 0)
        ),
        "retained_exact_activity_match": bool(
            activity_equivalence.get("exact_match", False)
        ),
    }
    plan.profile_metadata["automatic_validations"] = {
        "kineto_activity_and_timing": kineto_validation,
    }
    instrumented_by_mode_by_dp = {
        mode: {
            str(rank): [
                shards[rank]["step_samples_us"][mode][index]
                for index in retained_iteration_ids
            ]
            for rank in sorted(shards)
        }
        for mode in requested_modes
    }
    plan.profile_metadata[
        "instrumented_step_samples_us_by_mode_by_dp"
    ] = instrumented_by_mode_by_dp
    plan.profile_metadata[
        "instrumented_global_step_samples_us_by_mode"
    ] = {
        mode: [
            max(
                float(per_dp[str(rank)][sample])
                for rank in sorted(shards)
            )
            for sample in range(
                min(len(per_dp[str(rank)]) for rank in sorted(shards))
            )
        ]
        for mode, per_dp in instrumented_by_mode_by_dp.items()
    }
    if measure_uninstrumented_modes:
        by_mode_by_dp = {
            mode: {
                str(rank): list(shards[rank]["uninstrumented_step_samples_us"].get(mode, ()))
                for rank in sorted(shards)
            }
            for mode in requested_modes
        }
        plan.profile_metadata["uninstrumented_step_samples_us_by_mode_by_dp"] = by_mode_by_dp
        plan.profile_metadata["uninstrumented_global_step_samples_us_by_mode"] = {
            mode: [
                max(float(per_dp[str(rank)][sample]) for rank in sorted(shards))
                for sample in range(min(len(per_dp[str(rank)]) for rank in sorted(shards)))
            ]
            for mode, per_dp in by_mode_by_dp.items()
        }
    if output_path is not None:
        plan.profile_metadata["profile_output_path"] = str(
            Path(output_path).resolve()
        )
    piper_metadata.profiled_training_plan = plan
    if output_path is not None and dp_rank == 0:
        plan.save_json(output_path)
    # Do not let a nonzero DP driver surface a mandatory validation failure
    # before DP0 has durably written the diagnostic plan.  The harness cancels
    # sibling drivers as soon as one fails, which would otherwise discard the
    # exact mismatch report that explains the failure.
    ray.get(store.mark_profile_validation_ready.remote(generation, dp_rank))
    validation_deadline = time.monotonic() + 600.0
    while not ray.get(store.profile_validation_ready.remote(
        generation, dp_degree
    )):
        if time.monotonic() > validation_deadline:
            raise TimeoutError(
                "timed out waiting for profile validation artifact barrier"
            )
        time.sleep(0.05)
    if not kineto_validation["passed"]:
        from .profile_validation import ProfileAndSimulateValidationError

        raise ProfileAndSimulateValidationError(
            "Kineto profile validation failed: "
            + "; ".join(kineto_validation["errors"][:8])
        )
    return plan


def piper_profile_cupti_node_times(
    loss_fn,
    *,
    warmup_iters: int = 1,
    active_iters: int = 3,
    capture_dp_rank: int = 0,
    profile_modes: Sequence[str] = (
        INTERFERENCE_FREE_PROFILE_MODE,
        NORMAL_PROFILE_MODE,
    ),
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    """Collect CUPTI/Kineto CUDA activity totals per Piper DAG node.

    The DAG runs with the requested physical-stream policy, but no Piper
    per-node CUDA timing events are inserted.  Existing ``record_function``
    node ranges let Kineto correlate CUDA activity to the dispatching node.
    """
    if int(warmup_iters) < 0 or int(active_iters) <= 0:
        raise ValueError("invalid CUPTI warmup/active iteration counts")
    modes = tuple(normalize_profile_mode(mode) for mode in profile_modes)
    actors = piper_metadata.actors
    if not actors:
        raise RuntimeError("piper_setup() must create actors before CUPTI profiling")
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    pp_degree = int(os.environ["PIPER_PP_DEGREE"])
    capture_this_driver = dp_rank == int(capture_dp_rank)
    captured: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for mode in modes:
        ray.get([
            actor.synchronize_device.remote() for actor in actors.values()
        ])
        if capture_this_driver:
            ray.get([
                actor.start_cupti_node_profiler.remote(
                    warmup_iters=int(warmup_iters), active_iters=int(active_iters)
                )
                for actor in actors.values()
            ])
        for _ in range(int(warmup_iters) + int(active_iters)):
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    record=False,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])
        values = (
            ray.get([
                actor.stop_cupti_node_profiler.remote()
                for actor in actors.values()
            ])
            if capture_this_driver
            else [{} for _ in actors]
        )
        captured[mode] = {
            dp_rank * pp_degree + int(pp_rank): value
            for pp_rank, value in zip(sorted(actors), values)
        }
    return captured


def piper_measure_profile_modes(
    loss_fn,
    *,
    warmup_iters: int,
    repeat_iters: int,
    block_iters: int = 1,
    profile_modes: Sequence[str] = (
        INTERFERENCE_FREE_PROFILE_MODE,
        NORMAL_PROFILE_MODE,
    ),
) -> dict[str, tuple[float, ...]]:
    """Measure stream policies without per-node CUDA-event instrumentation.

    Every DP worker must call this function with the same modes and iteration
    counts.  Each policy starts from the same captured model, optimizer, and RNG
    state and then runs as one contiguous block.  Each returned sample is local
    driver wall time in microseconds; callers combine aligned DP samples with a
    maximum to obtain global step time.
    """

    if int(warmup_iters) < 0:
        raise ValueError("warmup_iters must be non-negative")
    if int(repeat_iters) <= 0:
        raise ValueError("repeat_iters must be positive")
    if int(block_iters) <= 0:
        raise ValueError("block_iters must be positive")
    modes = tuple(normalize_profile_mode(mode) for mode in profile_modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError(f"invalid profile modes: {modes!r}")
    actors = piper_metadata.actors
    if not actors:
        raise RuntimeError("piper_setup() must create actors before mode timing")

    samples: dict[str, list[float]] = {mode: [] for mode in modes}
    ray.get([
        actor.capture_profile_training_state.remote()
        for actor in actors.values()
    ])
    for mode in modes:
        ray.get([
            actor.restore_profile_training_state.remote()
            for actor in actors.values()
        ])
        ray.get([actor.synchronize_device.remote() for actor in actors.values()])
        for _ in range(int(warmup_iters)):
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    record=False,
                    synchronize_before=False,
                    synchronize=False,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])
        ray.get([actor.synchronize_device.remote() for actor in actors.values()])
        for _ in range(int(repeat_iters)):
            ray.get([
                actor.synchronize_device.remote() for actor in actors.values()
            ])
            started = time.perf_counter()
            for _ in range(int(block_iters)):
                ray.get([
                    actor.profile_run_dag.remote(
                        loss_fn=loss_fn,
                        record=False,
                        synchronize_before=False,
                        synchronize=False,
                        profile_mode=mode,
                    )
                    for actor in actors.values()
                ])
            ray.get([
                actor.synchronize_device.remote() for actor in actors.values()
            ])
            samples[mode].append(
                (time.perf_counter() - started)
                * 1_000_000.0
                / int(block_iters)
            )
    ray.get([
        actor.restore_profile_training_state.remote()
        for actor in actors.values()
    ])
    return {mode: tuple(values) for mode, values in samples.items()}


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
    from .profile_validation import ProfileAndSimulateValidationError

    kineto_validation = plan.profile_metadata.get(
        "automatic_validations", {}
    ).get("kineto_activity_and_timing")
    activity_equivalence = plan.profile_metadata.get(
        "kineto_activity_equivalence"
    )
    if kineto_validation is not None and not kineto_validation.get(
        "passed", False
    ):
        raise ProfileAndSimulateValidationError(
            "saved Kineto activity/timing validation did not pass"
        )
    if activity_equivalence is not None and not activity_equivalence.get(
        "exact_match", False
    ):
        raise ProfileAndSimulateValidationError(
            "saved normal/IF Kineto activity-equivalence validation did not pass"
        )
    from .simulator import simulate_profile_modes

    actual_us = (
        None
        if actual_step_samples_s is None
        else tuple(float(sample) * 1_000_000.0 for sample in actual_step_samples_s)
    )
    comparison = simulate_profile_modes(
        plan,
        actual_step_samples_us=actual_us,
        **simulation_kwargs,
    )
    from .profile_validation import (
        build_and_validate_attributed_trace,
        validate_measured_normal_samples,
        validate_simulation_endpoints,
    )

    endpoint_validation = validate_simulation_endpoints(comparison)
    dp_rank = int(os.environ.get("PIPER_DP_RANK", "0"))
    if dp_rank == 0:
        from .attribution import attribute_profile_mode_savings

        # One deterministic order is sufficient for the mandatory structural
        # validation because it must telescope exactly to the replay endpoint.
        # The report consumes only the order-invariant net checksum; higher-
        # sample Shapley attribution remains available through the offline CLI.
        attribution = attribute_profile_mode_savings(
            plan,
            permutations=1,
            seed=0,
            antithetic=False,
            strict=bool(simulation_kwargs.get("strict", True)),
        )
        endpoint_validation.update({
            "attribution_features": len(attribution.features),
            "attribution_permutations": attribution.permutations,
            "attribution_step_time_estimator": (
                attribution.step_time_estimator
            ),
            "attribution_sample_iteration_ids": list(
                attribution.sample_iteration_ids
            ),
            "normal_step_time_us": attribution.normal_step_time_us,
            "interference_free_step_time_us": (
                attribution.interference_free_step_time_us
            ),
            "attributed_step_savings_us": (
                attribution.attributed_step_savings_us
            ),
            "additivity_residual_us": attribution.additivity_residual_us,
            "positive_exposed_step_savings_us": (
                attribution.positive_exposed_step_savings_us
            ),
            "negative_exposed_step_savings_us": (
                attribution.negative_exposed_step_savings_us
            ),
            "net_exposed_step_savings_us": (
                attribution.net_exposed_step_savings_us
            ),
            "net_exposed_step_savings_pct": (
                attribution.net_exposed_step_savings_pct
            ),
            "exposed_savings_validation_residual_us": (
                attribution.exposed_savings_validation_residual_us
            ),
        })
        trace, trace_validation = build_and_validate_attributed_trace(
            comparison.normal,
            attribution,
        )
        measured_validation = validate_measured_normal_samples(
            plan,
            comparison,
            warning_threshold_pct=2.0,
        )
        automatic = plan.profile_metadata.setdefault(
            "automatic_validations", {}
        )
        automatic["simulation_endpoints"] = endpoint_validation
        automatic["attributed_trace_losslessness"] = trace_validation
        automatic["measured_vs_simulated_normal"] = measured_validation

        warning = measured_validation.get("warning")
        if warning:
            logger.warning("profile-and-simulate validation: %s", warning)
            warnings.warn(
                f"profile-and-simulate validation: {warning}",
                RuntimeWarning,
                stacklevel=2,
            )

        output_value = plan.profile_metadata.get("profile_output_path")
        if output_value:
            profile_path = Path(str(output_value))
            output_dir = profile_path.parent
            attribution_path = output_dir / "savings_attribution.json"
            trace_path = output_dir / "savings_attribution_trace.json"
            validation_path = output_dir / "profile_and_simulate_validation.json"
            attribution_path.write_text(
                json.dumps(attribution.to_dict(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            trace_path.write_text(
                json.dumps(trace, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            validation_path.write_text(
                json.dumps(automatic, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            plan.save_json(profile_path)
    return comparison


def _log_step_stats(step_time: float, log_memory: bool, actors: dict) -> None:
    stats = [f"step_time={step_time:.3f}s"]
    tokens = getattr(piper_metadata, "tokens_per_step", None)
    if tokens is not None:
        stats.append(f"throughput={tokens / step_time:.1f} tok/s")
    logger.info("  ".join(stats))
