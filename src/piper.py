from contextlib import contextmanager
from typing import Iterator

import torch.fx as fx

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
from .state import LOG_LEVEL, create_logger
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


class DagBackend:
    """TorchDynamo backend: split the traced graph by Piper annotations and
    lower the schedule directives into one TrainingDAG per PP rank.

    schedule_directives: parsed schedule program.
    visualize_dag: render per-rank DAG artifacts under ``artifact_dir``.
    artifact_dir: output directory for visualizations.
    """

    def __init__(
        self,
        schedule_directives: list[dict],
        visualize_dag: bool = False,
        artifact_dir: str = "out",
    ):
        self.schedule_directives = list(schedule_directives)
        self.visualize_dag = visualize_dag
        self.artifact_dir = artifact_dir
        self.training_dag: TrainingDAG | None = None
        self.per_pp_training_dags: list[TrainingDAG] | None = None

    def __call__(self, gm, example_inputs, **kwargs):
        del example_inputs, kwargs
        _top_level_gm, annotation_segments = split_gm_by_annotations(gm)

        if not annotation_segments:
            raise ValueError(
                "No Piper annotations found in the traced graph. Wrap model compute "
                "with src.piper.annotate(...) before compiling with Piper."
            )

        training_dag = build_training_dag(annotation_segments)
        _validate_schedule_tags_exist(training_dag, self.schedule_directives)
        apply_schedule_directives(training_dag, self.schedule_directives)
        self.training_dag = training_dag
        per_pp_training_dags = _split_global_training_dag_by_pp_rank(training_dag)
        for i, subdag in enumerate(per_pp_training_dags):
            zero_chains = _prune_zero_lifetime_metadata(subdag)
            resolve_total_order_per_stream(subdag)
            _add_inter_chain_temporal_edges(subdag, zero_chains)
            if self.visualize_dag:
                log_training_dag_dependencies(subdag)
                print_training_dag_order(subdag, label=f"pp{i}", rank=i, out_dir=self.artifact_dir)
                render_training_dag(subdag, output_path=f"{self.artifact_dir}/training_dag_pp{i}")
        self.per_pp_training_dags = per_pp_training_dags

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
