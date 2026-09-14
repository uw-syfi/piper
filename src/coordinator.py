import asyncio
import ray
from typing import Callable
import os

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .state import create_logger, LOG_LEVEL
from . import state as piper_state
from .schedule import load_schedule_info


# Coordinator needs GPUs when using profiling to infer stage boundaries
# @ray.remote(num_gpus=0.1)

# Use manual stage annotations- more stable
@ray.remote(num_gpus=0.1)
def run_dp_rank(
    dp_rank, dp_degree, pp_degree, world_size, training_func: Callable, *args,
    coordinator=None, **kwargs,
):
    logger = create_logger("coordinator", LOG_LEVEL)
    logger.debug(f"Running DP rank {dp_rank+1} of {dp_degree}")

    os.environ["PIPER_DP_RANK"] = str(dp_rank)
    os.environ["PIPER_DP_DEGREE"] = str(dp_degree)
    os.environ["PIPER_PP_DEGREE"] = str(pp_degree)
    os.environ["PIPER_WORLD_SIZE"] = str(world_size)
    os.environ["TORCH_LOGS"] = "+graph_breaks"
    # Via the module: Ray ships this function by value, so a directly
    # referenced `piper_metadata` would be a pickled copy, not the singleton.
    piper_state.piper_metadata.coordinator = coordinator
    return training_func(*args, **kwargs)


@ray.remote
class PiperProgramCoordinator:
    """Central actor that coordinates all the DP replicas of a single
    pipeline: spawns and supervises the dp_rank driver tasks.
    """

    def __init__(
        self,
        pp_outer: bool = False,
        schedule_directives_file: str | None = None,
    ):
        if schedule_directives_file is None:
            raise ValueError("PiperProgramCoordinator requires schedule_directives_file")
        info = load_schedule_info(schedule_directives_file)
        self.dp_degree = info["dp_degree"]
        self.pp_degree = info["pp_degree"]
        self.world_size = self.dp_degree * self.pp_degree
        # pp_outer=True means one PP stage per node (placement bundles keyed by
        # pp_rank). In that mode DP drivers are spread across the pp bundles.
        self.pp_outer = pp_outer

    async def run_program(self, training_func: Callable, pg, *args, **kwargs):
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

        self_handle = ray.get_runtime_context().current_actor

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
                coordinator=self_handle,
                **kwargs,
            )
            for dp_rank in range(self.dp_degree)
        ]
        tasks = [asyncio.ensure_future(ref) for ref in refs]

        pending = set(tasks)
        results = []
        failures = 0
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    results.append(task.result())
                except Exception:
                    failures += 1
                    logger.exception(
                        f"a dp_rank task failed ({failures}/{self.dp_degree})"
                    )
                    raise
        return results


def create_piper_placement_group(schedule_directives_file: str, pp_outer: bool = False):
    info = load_schedule_info(schedule_directives_file)
    pp_degree = info["pp_degree"]
    dp_degree = info["dp_degree"]

    if pp_outer:
        drivers_per_bundle = (dp_degree + pp_degree - 1) // pp_degree
        return placement_group(
            [{"CPU": dp_degree + drivers_per_bundle, "GPU": dp_degree}] * pp_degree,
            strategy="STRICT_SPREAD",
        )

    return placement_group(
        [{"CPU": pp_degree, "GPU": pp_degree}] * dp_degree,
        strategy="SPREAD",
    )
