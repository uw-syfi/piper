import os

import ray
from typing import Callable

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .state import create_logger, LOG_LEVEL
from .schedule import load_schedule_info
from .device import get_device


@ray.remote
def run_dp_rank(dp_rank, dp_degree, pp_degree, world_size, training_func: Callable, *args, **kwargs):
    logger = create_logger("coordinator", LOG_LEVEL)
    logger.debug(f"Running DP rank {dp_rank+1} of {dp_degree}")

    os.environ["PIPER_DP_RANK"] = str(dp_rank)
    os.environ["PIPER_DP_DEGREE"] = str(dp_degree)
    os.environ["PIPER_PP_DEGREE"] = str(pp_degree)
    os.environ["PIPER_WORLD_SIZE"] = str(world_size)
    os.environ["TORCH_LOGS"] = "+graph_breaks"
    return training_func(*args, **kwargs)


@ray.remote
class PiperProgramCoordinator:
    """Central Actor that Coordinates all the DP replicas of a single pipeline"""

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

        run_options: dict = {}
        accel = get_device().accelerator_resource
        if accel == "GPU":
            # Coordinator needs GPUs when using profiling to infer stage
            # boundaries, if manual stage annotations are not being used.
            # TODO(swang): Is this necessary?
            run_options["num_gpus"] = 0.1
        if pg is not None:
            run_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
            )

        refs = []
        for dp_rank in range(self.dp_degree):
            dp_options = dict(run_options)
            if pg is not None:
                dp_options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=(
                        dp_rank % self.pp_degree if self.pp_outer else dp_rank
                    ),
                )
            refs.append(
                run_dp_rank.options(**dp_options).remote(
                    dp_rank,
                    self.dp_degree,
                    self.pp_degree,
                    self.world_size,
                    training_func,
                    *args,
                    **kwargs,
                )
            )
        return ray.get(refs)


def create_piper_placement_group(schedule_directives_file: str, pp_outer: bool = False):
    info = load_schedule_info(schedule_directives_file)
    pp_degree = info["pp_degree"]
    dp_degree = info["dp_degree"]

    accel = get_device().accelerator_resource

    if pp_outer and accel is not None:
        drivers_per_bundle = (dp_degree + pp_degree - 1) // pp_degree
        bundle = {"CPU": dp_degree + drivers_per_bundle, accel: dp_degree}
        num_bundles = pp_degree
        strategy = "STRICT_SPREAD"
    else:
        bundle = {"CPU": max(pp_degree, 1)}
        if accel is not None:
            bundle[accel] = pp_degree
        num_bundles = dp_degree
        strategy = "SPREAD"

    return placement_group([bundle] * num_bundles, strategy=strategy)
