import ray
import torch
import copy
import time
from pathlib import Path

from .actor import _create_actors
from .state import (
    piper_metadata,
    create_logger,
    LOG_LEVEL,
)
from .fx import _serialize_graphmodule
from .model_state import DEFAULT_MODEL_SEED
from .piper import _reset_annotation_state, piper
from .schedule import (
    derive_schedule_info,
    load_schedule_directives,
)
logger = create_logger("compile", LOG_LEVEL)

_COMPILED_DATA_ACTOR = "piper_compiled_data"

@ray.remote
class _CompiledDataStore:
    """Coordinator-owned monotonic clock used to align actor profiles."""

    def monotonic_time_ns(self) -> int:
        """Return the coordinator's monotonic clock for actor calibration."""

        return time.monotonic_ns()

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


def _dag_device_key(dag) -> tuple[int, ...]:
    keys = {
        tuple(sorted(int(rank) for rank in node.device))
        for node in dag.nodes.values()
        if getattr(node, "device", None) is not None
    }
    if len(keys) != 1:
        raise ValueError(
            "Each local ActorGroup DAG must have exactly one place.devices set; "
            f"got {sorted(keys)}"
        )
    return next(iter(keys))


def _actor_group_dags_by_id(per_group_training_dags: list, schedule_info: dict) -> dict[int, object]:
    """Match local DAGs to schedule ActorGroups by exact device set."""
    group_id_by_devices = {
        tuple(group["devices"]): int(group["id"])
        for group in schedule_info.get("actor_groups", [])
    }
    result = {}
    for dag in per_group_training_dags:
        device_key = _dag_device_key(dag)
        try:
            group_id = group_id_by_devices[device_key]
        except KeyError as exc:
            raise ValueError(
                f"Local DAG placement {device_key} does not match a schedule ActorGroup; "
                f"known={sorted(group_id_by_devices)}"
            ) from exc
        if group_id in result:
            raise ValueError(
                f"ActorGroup {group_id} produced multiple disconnected local DAGs; "
                "add schedule order edges to connect its chunks"
            )
        result[group_id] = dag
    expected = set(group_id_by_devices.values())
    if set(result) != expected:
        raise ValueError(
            "Local DAGs do not cover every schedule ActorGroup; "
            f"missing={sorted(expected - set(result))}"
        )
    return result


def _load_training_dags_on_actors(per_group_training_dags: list, schedule_info: dict) -> None:
    """Load one local DAG onto every actor in its schedule-defined group."""
    actor_groups = schedule_info.get("actor_groups", [])
    dag_by_group = _actor_group_dags_by_id(per_group_training_dags, schedule_info)
    ray.get([
        piper_metadata.actors[int(global_rank)].load_training_dag.remote(
            dag_by_group[int(group["id"])]
        )
        for group in actor_groups
        for global_rank in group["devices"]
    ])


def _input_actor_group_id(dag_by_group: dict[int, object]) -> int:
    """Find the group owning the first traced forward segment."""
    candidates = []
    for group_id, dag in dag_by_group.items():
        for node in dag.nodes.values():
            if node.node_kind != "COMPUTE" or node.compute_subkind != "FWD":
                continue
            meta = node.node_meta if isinstance(node.node_meta, dict) else {}
            candidates.append(
                (
                    int(meta.get("stage_id", 1 << 30)),
                    int(meta.get("segment_id", 1 << 30)),
                    str(node.uid),
                    int(group_id),
                )
            )
    if not candidates:
        raise ValueError("ActorGroup DAGs contain no forward compute node")
    return min(candidates)[-1]


def _resolve_model_constructor_value(value, schedule_info: dict):
    if callable(value):
        return value(schedule_info)
    return value


def _artifact_dir_for_schedule(schedule_directives_file: str) -> str:
    parent = Path(schedule_directives_file).parent
    return str(parent if str(parent) else Path("out"))


def piper_setup(
    model_class,
    model_args=(),
    model_kwargs={},
    model_dtype=torch.bfloat16,
    optim_fn=None,
    example_inputs=None,
    example_outputs=None,
    activation_checkpointing=False,
    num_checkpoints=1,
    use_inductor: bool = False,
    no_nvtx: bool = False,
    pg=None,
    nsight=False,
    temp_dir: str = None,
    visualize_dag: bool = False,
    const_attrs: dict = None,
    pp_outer: bool = False,
    schedule_directives_file: str | None = None,
    model_seed: int = DEFAULT_MODEL_SEED,
):
    """
    Compile a model with the TrainingDAG backend.

    The model and example inputs are always traced on the meta device so the
    full model is never materialized on the driver GPU at compile time.

    Args:
        model: The model to compile.
        optim_fn: Callable ``(params) -> Optimizer`` used to create optimizers.
        example_inputs: Example inputs for tracing.
        example_outputs: Example outputs (labels) for tracing.
        use_inductor: When true, actors torch.compile stage GraphModules
            during _load_stage with the default inductor backend.
        schedule_directives_file: JSON schedule description consumed by the
            TrainingDAG backend. Piper derives its ActorGroups and microbatch
            metadata from this file.
        model_seed: Shared seed used to initialize every lifted model parameter
            deterministically by its eager ``state_dict`` name.
    """

    # Clear Dynamo's global compilation cache so that a previous piper_setup
    # call (e.g. with a different model size or schedule) in the same Python
    # worker process cannot contaminate this compilation via a stale cache hit.
    torch._dynamo.reset()

    schedule_info = {}
    if schedule_directives_file is not None:
        schedule_directives = load_schedule_directives(schedule_directives_file)
        schedule_info = derive_schedule_info(schedule_directives, schedule_directives_file)
    else:
        raise ValueError("piper_setup requires schedule_directives_file")

    piper_metadata.visualize_dag = visualize_dag
    piper_metadata.artifact_dir = _artifact_dir_for_schedule(schedule_directives_file)
    piper_metadata.schedule_directives = list(schedule_directives or [])
    piper_metadata.schedule_directives_file = schedule_directives_file
    piper_metadata.schedule_info = dict(schedule_info)

    # Reset DAG/compile fields so stale data from a prior run never leaks into
    # this run if the backend is somehow not re-invoked.
    piper_metadata.training_dag = None
    piper_metadata.per_pp_training_dags = None
    piper_metadata.compiled_data_store = None
    piper_metadata.profiled_training_plan = None

    actor_groups = tuple(
        tuple(int(rank) for rank in group["devices"])
        for group in schedule_info["actor_groups"]
    )
    pp_degree = len(actor_groups)
    _create_actors(
        actor_groups,
        optim_fn,
        profile=nsight,
        no_nvtx=no_nvtx,
        pg=pg,
        temp_dir=temp_dir,
        use_inductor=use_inductor,
        model_seed=model_seed,
    )

    piper_metadata.compiled_data_store = _CompiledDataStore.options(
        name=_COMPILED_DATA_ACTOR,
        lifetime="detached",
        num_cpus=0,
        get_if_exists=True,
    ).remote()

    # The sole program driver controls every actor, so it can obtain the
    # process-group rendezvous endpoint directly from global rank zero.
    master_addr, master_port = ray.get(
        piper_metadata.actors[0].get_node_ip_and_free_port.remote()
    )
    logger.debug(
        "Actor process-group master address: %s:%s", master_addr, master_port
    )
    ray.get([
        actor._join_process_groups.remote(master_addr, master_port)
        for actor in piper_metadata.actors.values()
    ])

    # Push non-trainable constant tensor attributes (e.g. freqs_cis, rope_cache, mask)
    # to all actors so _load_stage can initialize them correctly instead of zero-filling.
    _const_attrs = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in (const_attrs or {}).items()
    }
    if _const_attrs:
        ray.get([
            actor.load_const_attrs.remote(_const_attrs)
            for actor in piper_metadata.actors.values()
        ])

    # Build and compile once, then distribute each local DAG to every member
    # of the exact ActorGroup selected by its place directive.
    resolved_model_args = _resolve_model_constructor_value(
        model_args, piper_metadata.schedule_info
    )
    resolved_model_kwargs = _resolve_model_constructor_value(
        model_kwargs, piper_metadata.schedule_info
    )
    with torch.device("meta"):
        model = model_class(
            *tuple(resolved_model_args or ()),
            **dict(resolved_model_kwargs or {}),
        )
        if model_dtype is not None:
            model = model.to(model_dtype)

    num_params = sum(p.numel() for p in model.parameters())
    if model_dtype == torch.bfloat16:
        param_size_gb = num_params * 2 / (1024**3)
    elif model_dtype == torch.float32:
        param_size_gb = num_params * 4 / (1024**3)
    else:
        raise ValueError(f"Unsupported model dtype: {model_dtype}")
    logger.info(
        "Model size: %.0f M parameters (%.2f GB), dtype: %s",
        num_params / 1e6,
        param_size_gb,
        model_dtype,
    )

    compiled = torch.compile(model, backend=piper, fullgraph=True)
    compile_inputs = [x.to(device="meta") for x in example_inputs]
    _reset_annotation_state()
    _ = compiled(*compile_inputs)

    local_training_dags = getattr(piper_metadata, "per_pp_training_dags", None)
    if not local_training_dags:
        raise RuntimeError(
            "piper backend did not produce per_pp_training_dags; "
            "backend transition incomplete"
        )
    if len(local_training_dags) != pp_degree:
        raise RuntimeError(
            f"Expected {pp_degree} ActorGroup DAGs, got {len(local_training_dags)}"
        )
    dag_by_group = _actor_group_dags_by_id(local_training_dags, schedule_info)
    ordered_training_dags = [
        dag_by_group[group_id] for group_id in range(pp_degree)
    ]
    training_dags_rpc = _prepare_training_dags_for_rpc(ordered_training_dags)
    piper_metadata.per_pp_training_dags = training_dags_rpc
    _load_training_dags_on_actors(training_dags_rpc, schedule_info)

    dag_by_group = _actor_group_dags_by_id(training_dags_rpc, schedule_info)
    input_group_id = _input_actor_group_id(dag_by_group)
    input_group = schedule_info["actor_groups"][input_group_id]
    piper_metadata.schedule_info["input_actor_group_id"] = input_group_id
    piper_metadata.schedule_info["input_actor_group_size"] = len(
        input_group["devices"]
    )
    ray.get([
        piper_metadata.actors[int(global_rank)].load_input.remote(example_inputs)
        for global_rank in input_group["devices"]
    ])
    loss_group_ids = {
        group_id
        for group_id, dag in dag_by_group.items()
        if any(
            isinstance(getattr(node, "node_meta", None), dict)
            and bool(node.node_meta.get("compute_loss", False))
            for node in dag.nodes.values()
        )
    }
    if not loss_group_ids:
        raise ValueError("ActorGroup DAGs contain no compute_loss node")
    ray.get([
        piper_metadata.actors[int(global_rank)].load_labels.remote(example_outputs)
        for group in schedule_info["actor_groups"]
        if int(group["id"]) in loss_group_ids
        for global_rank in group["devices"]
    ])
