import ray
import torch
import copy
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
            # Dynamo-backed SymInts retain weak references to their ShapeEnv
            # and are not pickleable by Ray. ``graphargs`` are capture-time
            # tensor templates used for parameter realization and static P2P
            # schemas, not the executable shape program (which lives in the
            # serialized GraphModule). Materialize their current shape hints
            # in the RPC-only copy. Routed row counts remain runtime-dynamic
            # and are allocated from exchanged routing metadata.
            concrete_graphargs = []
            for arg in meta.get("graphargs", ()):
                if not isinstance(arg, torch.Tensor):
                    concrete_graphargs.append(arg)
                    continue
                shape = tuple(int(dim) for dim in arg.shape)
                concrete = torch.empty(shape, dtype=arg.dtype, device="meta")
                requires_grad = bool(arg.requires_grad)
                if isinstance(arg, torch.nn.Parameter):
                    concrete = torch.nn.Parameter(
                        concrete,
                        requires_grad=requires_grad,
                    )
                else:
                    concrete.requires_grad_(requires_grad)
                concrete_graphargs.append(concrete)
            meta["graphargs"] = concrete_graphargs
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


def _partition_actor_group_batch(
    tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    devices: list[int] | tuple[int, ...],
    *,
    value_name: str,
) -> list[list[torch.Tensor]]:
    """Split a global microbatch equally across an ActorGroup's local ranks.

    ActorGroup rank order defines data-lane order: local rank ``i`` receives
    the ``i``th contiguous slice along tensor dimension zero.  Model constants
    belong in ``const_attrs``; every value passed here is therefore required to
    carry the same leading global-microbatch dimension.
    """

    width = len(devices)
    if width <= 0:
        raise ValueError(f"{value_name} ActorGroup must not be empty")
    if not isinstance(tensors, (list, tuple)) or not tensors:
        raise ValueError(f"{value_name} must be a non-empty tensor sequence")

    batch_size: int | None = None
    for tensor_idx, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{value_name}[{tensor_idx}] must be a tensor, got "
                f"{type(tensor).__name__}"
            )
        if tensor.ndim == 0:
            raise ValueError(
                f"{value_name}[{tensor_idx}] must have a leading global "
                "microbatch dimension"
            )
        tensor_batch = int(tensor.shape[0])
        if batch_size is None:
            batch_size = tensor_batch
        elif tensor_batch != batch_size:
            raise ValueError(
                f"all {value_name} tensors must have the same leading "
                f"dimension; expected {batch_size}, got {tensor_batch} at "
                f"index {tensor_idx}"
            )

    assert batch_size is not None
    if batch_size < width or batch_size % width:
        raise ValueError(
            f"global {value_name} microbatch size {batch_size} must be at "
            f"least and divisible by ActorGroup width {width}; "
            f"devices={list(devices)}"
        )

    local_batch_size = batch_size // width
    return [
        [
            tensor.narrow(0, local_rank * local_batch_size, local_batch_size)
            for tensor in tensors
        ]
        for local_rank in range(width)
    ]


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
    dynamic_input_dims: dict[int, tuple[int, ...]] | None = None,
):
    """
    Compile a model with the TrainingDAG backend.

    The model and example inputs are always traced on the meta device so the
    full model is never materialized on the driver GPU at compile time.

    Args:
        model: The model to compile.
        optim_fn: Callable ``(params) -> Optimizer`` used to create optimizers.
        example_inputs: Example inputs for tracing. Dimension zero is the
            global microbatch and is split equally across the input
            ActorGroup in local-rank order.
        example_outputs: Example outputs (labels). Dimension zero is split
            equally across every ActorGroup that computes the loss.
        use_inductor: When true, actors torch.compile stage GraphModules
            during _load_stage with the default inductor backend.
        schedule_directives_file: JSON schedule description consumed by the
            TrainingDAG backend. Piper derives its ActorGroups and microbatch
            metadata from this file.
        model_seed: Shared seed used to initialize every lifted model parameter
            deterministically by its eager ``state_dict`` name.
        dynamic_input_dims: Optional input-index to dimension-index mapping.
            Marked dimensions remain symbolic through annotation splitting so
            routed subgraphs can receive variable row counts.
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

    def compile_for_inputs(inputs: list[torch.Tensor]) -> list:
        torch._dynamo.reset()
        compile_inputs = [tensor.to(device="meta") for tensor in inputs]
        for input_idx, dims in (dynamic_input_dims or {}).items():
            if input_idx < 0 or input_idx >= len(compile_inputs):
                raise ValueError(
                    f"dynamic input index {input_idx} is outside "
                    f"[0, {len(compile_inputs)})"
                )
            tensor = compile_inputs[input_idx]
            for dim in sorted(set(dims)):
                if dim < 0 or dim >= tensor.ndim:
                    raise ValueError(
                        f"dynamic dimension {dim} is outside input[{input_idx}] "
                        f"rank {tensor.ndim}"
                    )
                torch._dynamo.mark_dynamic(tensor, dim)
        piper_metadata.training_dag = None
        piper_metadata.per_pp_training_dags = None
        compiled = torch.compile(model, backend=piper, fullgraph=True)
        _reset_annotation_state()
        _ = compiled(*compile_inputs)
        dags = getattr(piper_metadata, "per_pp_training_dags", None)
        if not dags:
            raise RuntimeError(
                "piper backend did not produce per_pp_training_dags; "
                "backend transition incomplete"
            )
        return dags

    # Placement is known only after annotations have been traced and schedule
    # directives have been applied. Use the driver's global example once to
    # discover the input ActorGroup, then make the authoritative capture with
    # one representative rank's actual local input shape. This keeps dense P2P
    # buffer metadata consistent with execution rather than relying on a
    # symbolic global batch dimension to be rebound after RPC serialization.
    discovery_dags = compile_for_inputs(list(example_inputs))
    discovery_by_group = _actor_group_dags_by_id(discovery_dags, schedule_info)
    input_group_id = _input_actor_group_id(discovery_by_group)
    input_group = schedule_info["actor_groups"][input_group_id]
    input_shards = _partition_actor_group_batch(
        example_inputs,
        input_group["devices"],
        value_name="input",
    )
    local_training_dags = (
        discovery_dags
        if len(input_group["devices"]) == 1
        else compile_for_inputs(input_shards[0])
    )
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
    final_input_group_id = _input_actor_group_id(dag_by_group)
    if final_input_group_id != input_group_id:
        raise RuntimeError(
            "input ActorGroup changed when tracing the per-rank input shape: "
            f"discovery={input_group_id}, final={final_input_group_id}"
        )
    ordered_training_dags = [
        dag_by_group[group_id] for group_id in range(pp_degree)
    ]
    training_dags_rpc = _prepare_training_dags_for_rpc(ordered_training_dags)
    piper_metadata.per_pp_training_dags = training_dags_rpc
    _load_training_dags_on_actors(training_dags_rpc, schedule_info)

    dag_by_group = _actor_group_dags_by_id(training_dags_rpc, schedule_info)
    piper_metadata.schedule_info["input_actor_group_id"] = input_group_id
    piper_metadata.schedule_info["input_actor_group_size"] = len(
        input_group["devices"]
    )
    ray.get([
        piper_metadata.actors[int(global_rank)].load_input.remote(
            input_shards[local_rank]
        )
        for local_rank, global_rank in enumerate(input_group["devices"])
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
    piper_metadata.schedule_info["loss_actor_group_ids"] = sorted(loss_group_ids)
    label_shards_by_group_id = {
        int(group["id"]): _partition_actor_group_batch(
            [example_outputs],
            group["devices"],
            value_name="label",
        )
        for group in schedule_info["actor_groups"]
        if int(group["id"]) in loss_group_ids
    }
    ray.get([
        piper_metadata.actors[int(global_rank)].load_labels.remote(
            label_shards_by_group_id[int(group["id"])][local_rank][0]
        )
        for group in schedule_info["actor_groups"]
        if int(group["id"]) in loss_group_ids
        for local_rank, global_rank in enumerate(group["devices"])
    ])
