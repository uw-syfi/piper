import ray
import torch
import copy
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from .actor import _create_actors
from .state import (
    piper_metadata,
    create_logger,
    LOG_LEVEL,
)
from .config import ActorConfig
from .dag import TrainingDAG
from .fx import _serialize_graphmodule
from .piper import DagBackend, _reset_annotation_state
from .schedule import (
    derive_schedule_info,
    load_schedule_directives,
)
from .topology import Topology

logger = create_logger("compile", LOG_LEVEL)

_RANK0_ADDR_ACTOR = "piper_rank0_addr"
_COMPILED_DATA_ACTOR = "piper_compiled_data"

@ray.remote
class _Rank0AddrStore:
    """Named actor used to share global rank-0's IP+port across independent dp_rank workers."""
    def __init__(self, addr: str, port: int):
        self._addr = addr
        self._port = port
    def get(self):
        return self._addr, self._port


@ray.remote
class _CompiledDataStore:
    """Named actor used to share compiled training-DAG data from dp_rank=0 to dp_rank>0."""
    def __init__(self):
        self._training_dag_data = None
        self._training_dag_ready = False

    def publish_training_dag_data(self, data: dict):
        self._training_dag_data = data
        self._training_dag_ready = True

    def is_ready(self) -> bool:
        return self._training_dag_ready

    def is_training_dag_ready(self) -> bool:
        return self._training_dag_ready

    def get(self) -> dict:
        if not self.is_ready():
            return None
        return dict(self._training_dag_data)

    def get_training_dag_data(self) -> dict:
        return self._training_dag_data





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
):
    """
    Compile a model with the TrainingDAG backend and load it onto this DP
    rank's actors (one driver process per DP rank, launched by the coordinator).

    Args:
        model: The model to compile.
        optim_fn: Callable ``(params) -> Optimizer`` used to create optimizers.
        example_inputs: Example inputs for tracing.
        example_outputs: Example outputs (labels) for tracing.
        use_inductor: When true, actors torch.compile stage GraphModules
            during _load_stage with the default inductor backend.
        schedule_directives_file: JSON schedule description consumed by the
            TrainingDAG backend. When provided, Piper derives pp/dp/mbs
            schedule metadata internally.
    """
    if schedule_directives_file is None:
        raise ValueError("piper_setup requires schedule_directives_file")
    schedule_directives = load_schedule_directives(schedule_directives_file)
    schedule_info = derive_schedule_info(schedule_directives, schedule_directives_file)

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

    topology = Topology(dp=int(schedule_info["dp_degree"]), pp=int(schedule_info["pp_degree"]))
    pp_degree = topology.pp
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    config = ActorConfig(
        use_inductor=use_inductor,
        no_nvtx=no_nvtx,
        nsight=nsight,
        temp_dir=temp_dir,
        pp_outer=pp_outer,
    )
    _create_actors(topology, dp_rank, optim_fn, config, pg=pg)

    # All dp_ranks must agree on a single master_addr: the IP of the actor with
    # global_rank=0 (pp_rank=0, dp_rank=0).  dp_rank=0 publishes it via a named
    # Ray actor; other dp_ranks wait until it's available.
    if dp_rank == 0:
        # Create the compiled-data store early so dp_rank>0 can poll for it.
        piper_metadata.compiled_data_store = _CompiledDataStore.options(
            name=_COMPILED_DATA_ACTOR,
            lifetime="detached",
            num_cpus=0,
            get_if_exists=True,
        ).remote()
        ray.get(piper_metadata.compiled_data_store.is_ready.remote())

        # Get IP and a free port from actor 0's node — it will be the TCPStore server.
        master_addr, master_port = ray.get(piper_metadata.actors[0].get_node_ip_and_free_port.remote())
        addr_store = _Rank0AddrStore.options(
            name=_RANK0_ADDR_ACTOR,
            lifetime="detached",
            num_cpus=0,
        ).remote(master_addr, master_port)
        ray.get(addr_store.get.remote())
    else:
        master_addr = master_port = None
        deadline = time.monotonic() + 600
        while master_addr is None:
            try:
                store = ray.get_actor(_RANK0_ADDR_ACTOR)
                master_addr, master_port = ray.get(store.get.remote())
            except ValueError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for Ray actor {_RANK0_ADDR_ACTOR}"
                    )
                time.sleep(0.05)
            except Exception:
                logger.exception("Failed while waiting for Ray actor %s", _RANK0_ADDR_ACTOR)
                raise

    logger.debug(f"DP rank {dp_rank}: Master address for process groups: {master_addr}:{master_port}")

    groups = topology.groups()
    ray.get([
        actor.join_groups.remote(master_addr, master_port, groups)
        for actor in piper_metadata.actors.values()
    ])

    if dp_rank == 0:
        try:
            ray.kill(ray.get_actor(_RANK0_ADDR_ACTOR))
        except ValueError:
            logger.debug("Ray actor %s already absent during cleanup", _RANK0_ADDR_ACTOR)
        except Exception:
            logger.exception("Failed to clean up Ray actor %s", _RANK0_ADDR_ACTOR)
            raise

    # Push non-trainable constant tensor attributes (e.g. freqs_cis, rope_cache, mask)
    # to all actors so _load_stage can initialize them correctly instead of zero-filling.
    # Must happen on every dp_rank since each dp_rank owns its own actors.
    _const_attrs = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in (const_attrs or {}).items()
    }
    if _const_attrs:
        ray.get([
            actor.load_const_attrs.remote(_const_attrs)
            for actor in piper_metadata.actors.values()
        ])

    if dp_rank == 0:
        # dp_rank=0 compiles and publishes the program; dp_rank>0 wait for it.
        program = compile(
            model_class,
            model_args=model_args,
            example_inputs=example_inputs,
            schedule_file=schedule_directives_file,
            dtype=model_dtype,
            model_kwargs=model_kwargs,
            visualize_dag=visualize_dag,
            artifact_dir=piper_metadata.artifact_dir,
        )
        ray.get(
            piper_metadata.compiled_data_store.publish_training_dag_data.remote(
                {"program": program}
            )
        )
    else:
        logger.debug(f"DP rank {dp_rank} waiting for dp_rank=0 training-DAG data...")
        training_dag_data = None
        deadline = time.monotonic() + 600
        while training_dag_data is None:
            try:
                store = ray.get_actor(_COMPILED_DATA_ACTOR)
                if ray.get(store.is_training_dag_ready.remote()):
                    training_dag_data = ray.get(store.get_training_dag_data.remote())
            except ValueError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for Ray actor {_COMPILED_DATA_ACTOR}"
                    )
            except Exception:
                logger.exception("Failed while waiting for Ray actor %s", _COMPILED_DATA_ACTOR)
                raise
            if training_dag_data is None:
                time.sleep(0.2)
        program = training_dag_data["program"]
        if program.pp != pp_degree:
            raise RuntimeError(
                f"Expected {pp_degree} per-PP DAGs from dp_rank=0, got {program.pp}"
            )

    piper_metadata.per_pp_training_dags = program.shards
    ray.get([
        piper_metadata.actors[pp_rank].load_training_dag.remote(program.shard_for(pp_rank))
        for pp_rank in range(pp_degree)
    ])

    ray.get([
        piper_metadata.actors[pp_rank].load_input.remote(example_inputs)
        for pp_rank in program.input_pp_ranks
    ])
    ray.get([
        piper_metadata.actors[pp_rank].load_labels.remote(example_outputs)
        for pp_rank in program.loss_pp_ranks
    ])
