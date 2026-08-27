import os
import time
from contextlib import contextmanager
from typing import Iterator

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



def _split_global_training_dag_by_pp_rank(training_dag: TrainingDAG) -> list[TrainingDAG]:
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
    if len(set(comp_device_keys)) != len(comp_device_keys):
        raise ValueError(
            f"expected distinct device sets across split components, got {comp_device_keys}"
        )

    # Materialize each component as a standalone TrainingDAG.
    subdags: list[TrainingDAG] = []
    for comp in components:
        sub = TrainingDAG()
        for uid in comp:
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
    per_pp_training_dags = _split_global_training_dag_by_pp_rank(training_dag)
    artifact_dir = getattr(piper_metadata, "artifact_dir", "out")
    for i, subdag in enumerate(per_pp_training_dags):
        zero_chains = _prune_zero_lifetime_metadata(subdag)
        resolve_total_order_per_stream(subdag)
        _add_inter_chain_temporal_edges(subdag, zero_chains)
        if getattr(piper_metadata, "visualize_dag", False):
            log_training_dag_dependencies(subdag)
            print_training_dag_order(subdag, label=f"pp{i}", rank=i, out_dir=artifact_dir)
            render_training_dag(subdag, output_path=f"{artifact_dir}/training_dag_pp{i}")
    piper_metadata.per_pp_training_dags = per_pp_training_dags

    logger.debug(
        "built TrainingDAG with %d nodes and %d edges, split into %d local PP-rank DAG(s)",
        len(training_dag.nodes),
        len(training_dag.edges),
        len(per_pp_training_dags),
    )

    def callback(*args, _gm=gm):
        logger.warning(
            "piper compiled callback invoked directly; running local graph execution"
        )
        return _gm(*args)

    return callback


class PiperResume(Exception):
    """Raised after promotion recovery completes.

    next_iter: first iteration to (re-)execute.
    """

    def __init__(self, next_iter: int):
        super().__init__(f"resume at iteration {next_iter}")
        self.next_iter = next_iter


def _promotion_cmd():
    """Return the active promotion command, or None when no standby machinery
    exists (M1 behavior) or no promotion is in progress."""
    if int(os.environ.get("PIPER_NUM_STANDBY", "0")) <= 0:
        return None
    return ray.get(piper_metadata.coordinator.get_cmd.remote(), timeout=10)


def piper_exec_dag(loss_fn, log_stats: bool = False, step_timeout: float | None = None) -> list:
    """Execute one training step using the loaded per-rank TrainingDAG.

    loss_fn: loss function forwarded to each actor's run_dag.
    log_stats: log step time and throughput after the step.
    step_timeout: optional seconds to wait for the step before logging that it
        is overdue; None disables the overdue check.

    Raises PiperResume after a standby promotion recovers this rank's step.
    """
    actors = piper_metadata.actors
    run_refs = [
        actor.run_dag.remote(loss_fn=loss_fn)
        for actor in actors.values()
    ]
    t0 = time.perf_counter()
    while True:
        try:
            results = ray.get(run_refs, timeout=step_timeout)
            break
        except ray.exceptions.GetTimeoutError:
            # TODO: stuck ranks cannot be handled here — this timeout
            # fires identically on the stuck rank and on peers blocked at
            # its rendezvous, so no local information can name the culprit.
            logger.warning(
                f"step exceeded step_timeout={step_timeout:.1f}s; "
                "still waiting on in-flight step (peer may be down)"
            )
        except (ray.exceptions.RayTaskError, ray.exceptions.RayActorError) as e:
            cmd = _promotion_cmd()
            my_rank = int(os.environ.get("PIPER_DP_RANK", "-1"))
            if (
                cmd is not None
                and cmd.get("op") == "promote"
                and my_rank != cmd.get("failed")
            ):
                # Fenced by the coordinator: the abort of our comms ended
                # this step; not a real failure of this rank.
                logger.info(
                    f"fenced by coordinator during promotion "
                    f"(step error: {type(e).__name__}: {str(e.cause)[:200] if hasattr(e, 'cause') else str(e)[:200]})"
                )
                survivor_actor = actors[0]
                ray.get(
                    survivor_actor.join_standby_group.remote(cmd["new_ranks"]),
                    timeout=180,
                )
                lc = ray.get(survivor_actor.get_last_committed.remote(), timeout=60)
                logger.info(
                    f"survivor: joined standby group {cmd['new_ranks']}; "
                    f"last_committed={lc}"
                )
                if my_rank == cmd.get("source"):
                    standby_rank = next(
                        r for r in cmd["new_ranks"] if r != cmd["source"]
                    )
                    standby_actor = ray.get(
                        piper_metadata.coordinator.get_actors.remote(standby_rank)
                    )[0]
                    state_ref = ray.get(
                        survivor_actor.make_state_ref.remote(), timeout=120
                    )
                    # Resume barrier: the survivor's live tensors must not
                    # be stepped until the transfer has landed.
                    ray.get(
                        standby_actor.load_state.remote([state_ref]),
                        timeout=300,
                    )
                    del state_ref
                    logger.info(
                        f"survivor: transferred state to standby "
                        f"{standby_rank}; resuming at iteration {lc + 1}"
                    )
                raise PiperResume(lc + 1) from e
            raise
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


def _log_step_stats(step_time: float, log_memory: bool, actors: dict) -> None:
    stats = [f"step_time={step_time:.3f}s"]
    tokens = getattr(piper_metadata, "tokens_per_step", None)
    if tokens is not None:
        stats.append(f"throughput={tokens / step_time:.1f} tok/s")
    logger.info("  ".join(stats))
