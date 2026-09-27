from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FTConfig:
    """Fault-tolerance settings.

    step_timeout: seconds after which an in-flight step is logged as overdue;
        None disables the check.
    """

    step_timeout: float | None = None


@dataclass
class ActorConfig:
    """Per-actor runtime knobs that do not depend on the model or schedule.

    use_inductor: torch.compile stage GraphModules on the actor.
    no_nvtx: disable NVTX ranges.
    nsight: launch actors under Nsight Systems.
    temp_dir: TMPDIR for Nsight reports.
    pp_outer: place one pipeline stage per bundle (all DP replicas of a stage
        colocated) instead of one DP replica per bundle.
    num_gpus: GPU fraction requested per actor.
    max_concurrency: Ray actor concurrency.
    actor_cls: Ray actor class to spawn; defaults to ``PiperActor``.
    env_vars: extra environment for the actor runtime_env.
    """

    use_inductor: bool = False
    no_nvtx: bool = False
    nsight: bool = False
    temp_dir: str | None = None
    pp_outer: bool = False
    num_gpus: float = 0.6
    max_concurrency: int = 1
    actor_cls: Any = None
    env_vars: dict[str, str] = field(default_factory=dict)
