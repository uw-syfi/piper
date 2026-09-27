import torch
import copy
from dataclasses import dataclass, field
from pathlib import Path

from .state import create_logger, LOG_LEVEL
from .dag import TrainingDAG
from .fx import _serialize_graphmodule
from .piper import DagBackend, _reset_annotation_state
from .schedule import (
    derive_schedule_info,
    load_schedule_directives,
)

logger = create_logger("compile", LOG_LEVEL)

torch._dynamo.config.capture_scalar_outputs = True


def _prepare_training_dags_for_rpc(per_pp_training_dags: list) -> list:
    """Return a Ray-safe copy of per-PP training DAGs.

    Raw fx.GraphModule objects in node_meta are replaced by serialized strings
    under ``gm_data`` to avoid Ray deserialization invoking FX tracing on HOPs.
    """
    rpc_dags = copy.deepcopy(per_pp_training_dags)
    for dag in rpc_dags:
        for node in dag.nodes.values():
            if getattr(node, "node_kind", None) != "COMPUTE":
                continue
            if getattr(node, "compute_subkind", None) != "FWD":
                continue
            meta = getattr(node, "node_meta", None)
            if not isinstance(meta, dict):
                raise ValueError(f"Malformed COMPUTE node metadata for uid={getattr(node, 'uid', '<unknown>')}")
            gm = meta.get("gm")
            if gm is None:
                raise ValueError(
                    f"Malformed COMPUTE/FWD node {getattr(node, 'uid', '<unknown>')}: missing node_meta['gm']"
                )
            meta["gm_data"] = _serialize_graphmodule(gm)
            del meta["gm"]
    return rpc_dags


def _compute_loss_pp_ranks(per_pp_training_dags: list) -> list[int]:
    ranks: list[int] = []
    for pp_rank, dag in enumerate(per_pp_training_dags):
        if any(
            isinstance(getattr(node, "node_meta", None), dict)
            and bool(node.node_meta.get("compute_loss", False))
            for node in dag.nodes.values()
        ):
            ranks.append(pp_rank)
    return ranks


def _resolve_model_constructor_value(value, schedule_info: dict):
    if callable(value):
        return value(schedule_info)
    return value


def _artifact_dir_for_schedule(schedule_directives_file: str) -> str:
    parent = Path(schedule_directives_file).parent
    return str(parent if str(parent) else Path("out"))


@dataclass
class DagProgram:
    """Compiled training program: one TrainingDAG shard per PP rank.

    shards: per-PP-rank DAGs, Ray-safe (GraphModules serialized).
    input_pp_ranks: PP ranks that consume the model inputs.
    loss_pp_ranks: PP ranks that consume the labels and compute the loss.
    schedule_info: pp/dp/mbs facts derived from the schedule.
    schedule_directives: the schedule program that was applied.
    """

    shards: list[TrainingDAG]
    input_pp_ranks: list[int]
    loss_pp_ranks: list[int]
    schedule_info: dict = field(default_factory=dict)
    schedule_directives: list[dict] = field(default_factory=list)
    kind: str = "dag"

    @property
    def pp(self) -> int:
        return len(self.shards)

    def shard_for(self, pp_rank: int) -> TrainingDAG:
        return self.shards[pp_rank]


def compile(
    model_cls,
    model_args=(),
    example_inputs=None,
    schedule_file: str | None = None,
    dtype=torch.bfloat16,
    model_kwargs=None,
    visualize_dag: bool = False,
    artifact_dir: str | None = None,
) -> DagProgram:
    """Trace ``model_cls`` on the meta device and lower the schedule into per-PP DAGs.

    model_cls, model_args, model_kwargs: model constructor; args/kwargs may be
        callables of the schedule info.
    example_inputs: example forward inputs (only shapes/dtypes are used).
    schedule_file: JSON schedule directives.
    dtype: parameter dtype the model is cast to before tracing.
    visualize_dag: render per-rank DAG artifacts under ``artifact_dir``.
    """
    if schedule_file is None:
        raise ValueError("compile requires schedule_file")
    if example_inputs is None:
        raise ValueError("compile requires example_inputs")

    # Clear Dynamo's global cache so an earlier compile in this process (e.g.
    # a different model size or schedule) cannot leak in via a stale cache hit.
    torch._dynamo.reset()

    schedule_directives = load_schedule_directives(schedule_file)
    schedule_info = derive_schedule_info(schedule_directives, schedule_file)
    pp_degree = int(schedule_info["pp_degree"])

    resolved_model_args = _resolve_model_constructor_value(model_args, schedule_info)
    resolved_model_kwargs = _resolve_model_constructor_value(model_kwargs or {}, schedule_info)
    with torch.device("meta"):
        model = model_cls(*tuple(resolved_model_args or ()), **dict(resolved_model_kwargs or {}))
        if dtype is not None:
            model = model.to(dtype)

    num_params = sum(p.numel() for p in model.parameters())
    if dtype == torch.bfloat16:
        param_size_gb = num_params * 2 / (1024**3)
    elif dtype == torch.float32:
        param_size_gb = num_params * 4 / (1024**3)
    else:
        raise ValueError(f"Unsupported model dtype: {dtype}")
    logger.info(f"Model size: {num_params/(1e6):.0f} M parameters ({param_size_gb:.2f} GB), dtype: {dtype}")

    backend = DagBackend(
        schedule_directives,
        visualize_dag=visualize_dag,
        artifact_dir=artifact_dir or _artifact_dir_for_schedule(schedule_file),
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    compile_inputs = [x.to(device="meta") for x in example_inputs]
    _reset_annotation_state()
    _ = compiled(*compile_inputs)

    per_pp_training_dags = backend.per_pp_training_dags
    if not per_pp_training_dags:
        raise RuntimeError("piper backend did not produce per_pp_training_dags")
    if len(per_pp_training_dags) != pp_degree:
        raise RuntimeError(
            f"Expected {pp_degree} per-PP DAGs, got {len(per_pp_training_dags)}"
        )
    shards = _prepare_training_dags_for_rpc(per_pp_training_dags)

    loss_pp_ranks = _compute_loss_pp_ranks(shards)
    if not loss_pp_ranks:
        loss_pp_ranks = [pp_degree - 1]
        logger.warning(
            "No compute_loss node found in per-PP DAGs; falling back to labels on pp_rank=%s",
            loss_pp_ranks[0],
        )
    return DagProgram(
        shards=shards,
        input_pp_ranks=[0],
        loss_pp_ranks=loss_pp_ranks,
        schedule_info=dict(schedule_info),
        schedule_directives=list(schedule_directives),
    )
