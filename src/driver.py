"""Single-driver runtime: placement, actors, process groups, program load and stepping."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import ray
import torch
from ray.util.placement_group import placement_group, placement_group_table

from .actor import _bundle_index, _spawn_actor
from .config import ActorConfig, FTConfig
from .device import get_device
from .state import LOG_LEVEL, create_logger
from .topology import Topology

logger = create_logger("driver", LOG_LEVEL)


@dataclass
class StepResult:
    """Outcome of one ``Runtime.step``.

    losses: microbatch losses per DP slot.
    next_iter: iteration to run next.
    recovery: None, or details of an in-step recovery.
    """

    losses: list[list[float]]
    next_iter: int
    recovery: Any = None


class StepFailed(RuntimeError):
    """A step task failed on ``slots``; ``kind`` is ``"actor_died"`` or ``"task_error"``."""

    def __init__(self, slots, kind: str, cause: BaseException | None = None):
        super().__init__(f"step failed on slots {list(slots)} ({kind}): {cause}")
        self.slots = list(slots)
        self.kind = kind
        self.cause = cause


def create_placement_group(topology: Topology, config: ActorConfig):
    """One bundle per DP replica (per stage when ``pp_outer``), plus one per standby."""
    accel = get_device().accelerator_resource
    if config.pp_outer:
        per_bundle, num_bundles, strategy = topology.dp, topology.pp, "STRICT_SPREAD"
    else:
        per_bundle, num_bundles, strategy = topology.pp, topology.dp, "SPREAD"
    bundle = {"CPU": per_bundle, **({accel: per_bundle} if accel else {})}
    standby = {"CPU": 1, **({accel: 1} if accel else {})}
    return placement_group([bundle] * num_bundles + [standby] * topology.num_standby, strategy=strategy)


class Runtime:
    """Actors, process groups and the loaded program of one training job.

    actors: ``{(dp, pp): actor}`` for every trainer slot.
    standbys: spare actors that joined the world but hold no slot.
    groups: explicit rank lists of every process group.
    """

    def __init__(self, topology: Topology, ft: FTConfig, config: ActorConfig, pg):
        self.topology = topology
        self.ft = ft
        self.config = config
        self.pg = pg
        self.actors: dict[tuple[int, int], Any] = {}
        self.standbys: list[Any] = []
        self.groups = topology.groups()
        self.program = None

    @property
    def all_actors(self) -> list[Any]:
        return list(self.actors.values()) + list(self.standbys)

    def _spawn_actors(self) -> None:
        topo, cfg = self.topology, self.config
        for slot in topo.slots():
            self.actors[slot] = _spawn_actor(
                slot, topo.rank(*slot), topo, cfg, pg=self.pg, bundle_index=_bundle_index(slot, cfg)
            )
        first_standby_bundle = topo.pp if cfg.pp_outer else topo.dp
        for i, rank in enumerate(topo.standby_ranks):
            self.standbys.append(
                _spawn_actor(None, rank, topo, cfg, pg=self.pg, bundle_index=first_standby_bundle + i)
            )

    def _join_groups(self) -> None:
        # Rank 0's node hosts the TCPStore every rank rendezvous on.
        master_addr, master_port = ray.get(self.actors[(0, 0)].get_node_ip_and_free_port.remote())
        logger.debug(f"process-group rendezvous at {master_addr}:{master_port}")
        ray.get([
            actor.join_groups.remote(master_addr, master_port, self.groups)
            for actor in self.all_actors
        ])

    def load(self, program, optim_fn: Callable, const_attrs: dict | None = None) -> None:
        """Ship one program shard per PP rank to every replica and standby.

        program: a ``DagProgram`` (``shard_for``, ``input_pp_ranks``, ``loss_pp_ranks``).
        optim_fn: callable ``(params) -> Optimizer`` applied on each actor.
        const_attrs: non-trainable model attributes (e.g. rope caches) by bare name.
        """
        if program.pp != self.topology.pp:
            raise ValueError(f"program has {program.pp} PP shards but topology has pp={self.topology.pp}")
        if self.standbys and self.topology.pp != 1:
            raise NotImplementedError("standby actors require pp == 1")

        attrs = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in (const_attrs or {}).items()
        }
        if attrs:
            ray.get([actor.load_const_attrs.remote(attrs) for actor in self.all_actors])

        refs = [
            actor.load_training_dag.remote(program.shard_for(pp), optim_fn)
            for (_dp, pp), actor in self.actors.items()
        ]
        refs += [actor.load_training_dag.remote(program.shard_for(0), optim_fn) for actor in self.standbys]
        ray.get(refs)
        self.program = program

    def step(self, it: int, inputs: list[list[torch.Tensor]], labels: list[torch.Tensor], loss_fn) -> StepResult:
        """Run iteration ``it`` on every DP slot.

        inputs: one list of model inputs per DP slot.
        labels: one label tensor per DP slot.
        loss_fn: ``(output, labels) -> loss`` applied on the loss ranks.
        """
        if self.program is None:
            raise RuntimeError("Runtime.load must be called before step")
        dp = self.topology.dp
        if len(inputs) != dp or len(labels) != dp:
            raise ValueError(f"step expects {dp} input lists and {dp} label tensors, got {len(inputs)} and {len(labels)}")

        pending: dict[Any, tuple[int, int]] = {}
        for dp_rank in range(dp):
            for pp in self.program.input_pp_ranks:
                pending[self.actors[(dp_rank, pp)].load_input.remote(inputs[dp_rank])] = (dp_rank, pp)
            for pp in self.program.loss_pp_ranks:
                pending[self.actors[(dp_rank, pp)].load_labels.remote(labels[dp_rank])] = (dp_rank, pp)
        run_refs = {actor.run_dag.remote(loss_fn=loss_fn, it=it): slot for slot, actor in self.actors.items()}
        pending.update(run_refs)

        losses: list[list[float]] = [[] for _ in range(dp)]
        for ref, result in self._wait_first_completed(pending, it):
            if ref in run_refs and isinstance(result, dict):
                losses[run_refs[ref][0]].extend(result.get("losses", []))
        return StepResult(losses=losses, next_iter=it + 1)

    def _wait_first_completed(self, pending: dict[Any, tuple[int, int]], it: int):
        """Yield ``(ref, result)`` as tasks finish; a failure raises ``StepFailed`` at once."""
        refs = list(pending)
        timeout = self.ft.step_timeout
        while refs:
            done, refs = ray.wait(refs, num_returns=1, timeout=timeout)
            if not done:
                # A stuck rank and its peers blocked at the rendezvous time out
                # identically, so no local information names the culprit.
                logger.warning(
                    f"iteration {it} exceeded step_timeout={timeout:.1f}s; "
                    f"still waiting on slots {sorted({pending[r] for r in refs})}"
                )
                continue
            ref = done[0]
            try:
                result = ray.get(ref)
            except ray.exceptions.RayActorError as exc:
                raise StepFailed([pending[ref]], "actor_died", exc) from exc
            except ray.exceptions.RayError as exc:
                raise StepFailed([pending[ref]], "task_error", exc) from exc
            yield ref, result


def setup(topology: Topology, ft: FTConfig | None = None, actors: ActorConfig | None = None) -> Runtime:
    """Create the placement group, ``dp * pp + num_standby`` actors and their process groups.

    topology: parallel degrees and standby count.
    ft: fault-tolerance settings.
    actors: per-actor runtime settings.
    """
    if not ray.is_initialized():
        ray.init()
    ft = ft or FTConfig()
    config = actors or ActorConfig()
    pg = create_placement_group(topology, config)
    ray.get(pg.ready(), timeout=600)
    logger.debug(placement_group_table(pg))
    rt = Runtime(topology, ft, config, pg)
    rt._spawn_actors()
    rt._join_groups()
    return rt
