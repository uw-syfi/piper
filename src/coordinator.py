import asyncio
import ray
from typing import Callable
import os

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .state import create_logger, LOG_LEVEL
from .schedule import load_schedule_info

_PROMOTION_SIGNAL_ACTOR = "piper_promotion_signal"


@ray.remote(num_cpus=0)
class PromotionSignal:
    """Named actor: promotion command channel + per-dp_rank actor registry.

    cmd: None until set; then {"op": "promote", "failed": int, "source": int,
    "new_ranks": [int, int]} or {"op": "shutdown"}.
    """

    def __init__(self):
        self._cmd = None
        self._event = asyncio.Event()
        self._actors = {}

    def register_actors(self, dp_rank, handles):
        self._actors[dp_rank] = handles

    def get_actors(self, dp_rank):
        return self._actors.get(dp_rank)

    def set(self, cmd):
        self._cmd = cmd
        self._event.set()

    def get(self):
        return self._cmd

    async def wait_for_cmd(self):
        await self._event.wait()
        return self._cmd


# Coordinator needs GPUs when using profiling to infer stage boundaries
# @ray.remote(num_gpus=0.1)

# Use manual stage annotations- more stable
@ray.remote(num_gpus=0.1)
def run_dp_rank(dp_rank, dp_degree, pp_degree, world_size, training_func: Callable, *args, num_standby=0, **kwargs):
    logger = create_logger("coordinator", LOG_LEVEL)
    logger.debug(f"Running DP rank {dp_rank+1} of {dp_degree}")

    os.environ["PIPER_DP_RANK"] = str(dp_rank)
    os.environ["PIPER_DP_DEGREE"] = str(dp_degree)
    os.environ["PIPER_PP_DEGREE"] = str(pp_degree)
    os.environ["PIPER_WORLD_SIZE"] = str(world_size)
    os.environ["PIPER_NUM_STANDBY"] = str(num_standby)
    os.environ["TORCH_LOGS"] = "+graph_breaks"
    return training_func(*args, **kwargs)


@ray.remote
class PiperProgramCoordinator:
    """Central Actor that Coordinates all the DP replicas of a single pipeline"""

    def __init__(
        self,
        pp_outer: bool = False,
        schedule_directives_file: str | None = None,
        num_standby: int = 0,
    ):
        if schedule_directives_file is None:
            raise ValueError("PiperProgramCoordinator requires schedule_directives_file")
        info = load_schedule_info(schedule_directives_file)
        self.dp_degree = info["dp_degree"]
        self.pp_degree = info["pp_degree"]
        self.num_standby = int(num_standby)
        if self.num_standby > 0 and (self.pp_degree > 1 or pp_outer):
            raise NotImplementedError(
                "standby ranks are only supported with pp_degree == 1 and pp_inner placement"
            )
        # Standby ranks are enrolled in the NCCL world but excluded from all
        # dp/ep training groups.
        self.world_size = self.dp_degree * self.pp_degree + self.num_standby
        # pp_outer=True means one PP stage per node (placement bundles keyed by
        # pp_rank). In that mode DP drivers are spread across the pp bundles.
        self.pp_outer = pp_outer

    def run_program(self, training_func: Callable, pg, *args, **kwargs):
        from .compile import _RANK0_ADDR_ACTOR, _COMPILED_DATA_ACTOR
        logger = create_logger("coordinator", LOG_LEVEL)
        try:
            ray.kill(ray.get_actor(_RANK0_ADDR_ACTOR))
        except ValueError:
            logger.debug("No stale Ray actor named %s to kill", _RANK0_ADDR_ACTOR)
        except Exception:
            logger.exception("Failed to kill stale Ray actor named %s", _RANK0_ADDR_ACTOR)
            raise
        # Kill any stale compiled-data store from a previous run so that
        # dp_rank>0 workers cannot read outdated (e.g. wrong-model) stage data.
        try:
            ray.kill(ray.get_actor(_COMPILED_DATA_ACTOR))
        except ValueError:
            logger.debug("No stale Ray actor named %s to kill", _COMPILED_DATA_ACTOR)
        except Exception:
            logger.exception("Failed to kill stale Ray actor named %s", _COMPILED_DATA_ACTOR)
            raise

        sig = None
        if self.num_standby > 0:
            try:
                ray.kill(ray.get_actor(_PROMOTION_SIGNAL_ACTOR))
            except ValueError:
                pass
            sig = PromotionSignal.options(
                name=_PROMOTION_SIGNAL_ACTOR, num_cpus=0
            ).remote()
            ray.get(sig.get.remote())  # ensure the name is resolvable before drivers start

        n_ranks = self.dp_degree + self.num_standby
        refs = [
            run_dp_rank.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=(
                        dp_rank % self.pp_degree if self.pp_outer else dp_rank
                    ),
                )
            ).remote(
                dp_rank,
                self.dp_degree,
                self.pp_degree,
                self.world_size,
                training_func,
                *args,
                num_standby=self.num_standby,
                **kwargs,
            )
            for dp_rank in range(n_ranks)
        ]
        ref_to_rank = {ref: dp_rank for dp_rank, ref in enumerate(refs)}
        trainer_pending = set(refs[: self.dp_degree])
        standby_refs = set(refs[self.dp_degree:])

        # React to whichever dp_rank task finishes or fails first. Standby
        # tasks are excluded from failure accounting.
        pending = list(refs)
        results = []
        failures = 0
        promoted = False
        shutdown_sent = False
        alive_standby = list(range(self.dp_degree, n_ranks))
        while pending:
            if (
                sig is not None
                and not promoted
                and not shutdown_sent
                and not trainer_pending
            ):
                # All trainer tasks resolved without promotion: release the
                # parked standby so the job can terminate.
                logger.info("all trainer tasks resolved; sending shutdown to standby")
                ray.get(sig.set.remote({"op": "shutdown"}))
                shutdown_sent = True
            done, pending = ray.wait(pending, num_returns=1)
            rank = ref_to_rank[done[0]]
            trainer_pending.discard(done[0])
            try:
                res = ray.get(done[0])
                if done[0] in standby_refs:
                    # Case 1 (no failure): standby was shut down, returns None,
                    # no metrics to record — drop is correct.
                    # TODO(phase-2), case 2 (promoted): the standby trained as a
                    # replacement and returns real metrics; append to results.
                    logger.info(f"standby dp_rank {rank} task finished")
                else:
                    results.append(res)
            except Exception:
                if done[0] in standby_refs:
                    logger.exception(
                        f"standby dp_rank {rank} task failed; failover disabled"
                    )
                    if rank in alive_standby:
                        alive_standby.remove(rank)
                    continue
                failures += 1
                logger.exception(
                    f"a dp_rank task failed ({failures}/{self.dp_degree})"
                )
                if (
                    sig is not None
                    and alive_standby
                    and not promoted
                    and failures < self.dp_degree # at least one trainer survives
                ):
                    promoted = True
                    self._promote(sig, logger, failed_rank=rank,
                                  standby_rank=alive_standby[0])
                    continue
                # No standby available for this failure: fail fast.
                raise
        return results

    def _promote(self, sig, logger, failed_rank: int, standby_rank: int):
        """Promote a standby to replace a failed trainer rank.

        failed_rank: dp_rank whose task failed.
        standby_rank: dp_rank of the standby taking over.
        """
        survivors = [
            r for r in range(self.dp_degree) if r != failed_rank
        ]
        source = survivors[0]
        # pp_degree == 1 (enforced in __init__), so global rank == dp_rank.
        new_ranks = sorted([source, standby_rank])
        cmd = {
            "op": "promote",
            "failed": failed_rank,
            "source": source,
            "new_ranks": new_ranks,
        }
        logger.info(f"promoting standby {standby_rank} for failed rank {failed_rank}: {cmd}")
        # Order matters: the signal must be readable BEFORE any abort lands,
        # so fenced drivers can distinguish the abort from a genuine failure.
        ray.get(sig.set.remote(cmd))
        for rank in survivors:
            handles = ray.get(sig.get_actors.remote(rank))
            for handle in handles:
                ray.get(handle.abort_comms.remote(), timeout=60)


def create_piper_placement_group(
    schedule_directives_file: str, pp_outer: bool = False, num_standby: int = 0
):
    info = load_schedule_info(schedule_directives_file)
    pp_degree = info["pp_degree"]
    dp_degree = info["dp_degree"]

    if pp_outer:
        if num_standby > 0:
            raise NotImplementedError("standby ranks require pp_inner placement")
        drivers_per_bundle = (dp_degree + pp_degree - 1) // pp_degree
        return placement_group(
            [{"CPU": dp_degree + drivers_per_bundle, "GPU": dp_degree}] * pp_degree,
            strategy="STRICT_SPREAD",
        )

    # One extra bundle per standby rank: its GPU is reserved up front.
    return placement_group(
        [{"CPU": pp_degree, "GPU": pp_degree}] * (dp_degree + num_standby),
        strategy="SPREAD",
    )
