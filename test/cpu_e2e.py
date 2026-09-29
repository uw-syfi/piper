"""Run Piper end to end on CPU: Ray actors, gloo process groups, real steps.

The helpers here drive the same path as ``examples/test_harness.py``: a
``PiperProgramCoordinator`` starts one ``run_dp_rank`` task per DP rank, and
each task calls ``piper_setup`` and ``piper_exec_dag``.
"""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

import pytest
import ray
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ray_cpu():
    """Start a local CPU-only Ray cluster for one test module."""
    os.environ["PIPER_DEVICE"] = "cpu"
    ray.init(
        num_cpus=16,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
        # Ray socket paths must stay under 108 bytes, so avoid a long TMPDIR.
        _temp_dir=f"/tmp/piper_test_ray_{getpass.getuser()}",
        runtime_env={
            "env_vars": {"PIPER_DEVICE": "cpu", "PYTHONPATH": str(REPO_ROOT)},
        },
    )
    try:
        yield
    finally:
        ray.shutdown()


def run_program(tmp_path: Path, schedule: list[dict], train_fn, *args, **kwargs) -> list:
    """Write *schedule*, run *train_fn* once per DP rank, return the per-rank results."""
    from src.coordinator import PiperProgramCoordinator

    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(schedule), encoding="utf-8")
    coordinator = PiperProgramCoordinator.remote(schedule_directives_file=str(path))
    try:
        return ray.get(coordinator.run_program.remote(train_fn, None, *args, **kwargs))
    finally:
        ray.kill(coordinator)


def kill_actors() -> None:
    """Kill this replica's Piper actors so the next test gets their CPUs back."""
    from src.state import piper_metadata

    for actor in piper_metadata.actors.values():
        ray.kill(actor)
    piper_metadata.actors.clear()


def dp_rank() -> int:
    return int(os.environ["PIPER_DP_RANK"])


def params_by_rank() -> dict[int, dict[str, torch.Tensor]]:
    """Return every actor's trainable parameters, keyed by PP rank."""
    from src.state import piper_metadata

    return {
        pp_rank: ray.get(actor.get_params_cpu.remote())
        for pp_rank, actor in piper_metadata.actors.items()
    }


def train(model_class, schedule_path: str, batches: list, steps: int, **setup_kwargs) -> dict:
    """Set up Piper on this DP rank, run *steps* steps on its batch, and report.

    *batches* holds one ``(inputs, labels)`` pair per DP rank, with every
    microbatch stacked along dim 0. Returns the
    parameters before and after training, keyed by PP rank, and the losses.
    """
    from src.compile import piper_setup
    from src.piper import piper_exec_dag
    from src.schedule import load_schedule_info
    from src.state import piper_metadata

    try:
        inputs, labels = batches[dp_rank()]
        num_mbs = load_schedule_info(schedule_path)["num_microbatches"]
        setup_kwargs.setdefault("model_dtype", torch.float32)
        setup_kwargs.setdefault("optim_fn", lambda params: torch.optim.SGD(params, lr=1.0))
        piper_setup(
            model_class,
            example_inputs=[t.chunk(num_mbs)[0] for t in inputs],
            example_outputs=labels.chunk(num_mbs)[0],
            schedule_directives_file=schedule_path,
            **setup_kwargs,
        )
        actors = piper_metadata.actors
        ray.get(
            [actors[0].load_input.remote(list(inputs))]
            + [actor.load_labels.remote(labels) for actor in actors.values()]
        )
        initial = params_by_rank()
        losses = [piper_exec_dag(torch.nn.functional.mse_loss) for _ in range(steps)]
        return {"initial": initial, "losses": losses, "final": params_by_rank()}
    finally:
        kill_actors()


def dynamo_param_name(fqn: str) -> str:
    """Return the placeholder name Dynamo gives parameter *fqn*, e.g. ``a.weight``."""
    *modules, leaf = fqn.split(".")
    return "l_self_" + "".join(f"modules_{m}_" for m in modules) + f"parameters_{leaf}_"


def load_initial(model: torch.nn.Module, initial: dict) -> torch.nn.Module:
    """Copy Piper's *initial* weights into *model*."""
    with torch.no_grad():
        for fqn, param in model.named_parameters():
            param.copy_(initial[dynamo_param_name(fqn)])
    return model


def reference_sgd_step(model: torch.nn.Module, initial: dict, batches: list) -> dict:
    """Take one SGD step (lr=1) on the concatenated *batches*, from Piper's *initial* weights.

    Returns the updated parameters under their Dynamo placeholder names.
    """
    load_initial(model, initial)
    inputs = torch.cat([batch[0][0] for batch in batches])
    labels = torch.cat([batch[1] for batch in batches])
    torch.nn.functional.mse_loss(model(inputs), labels).backward()
    return {
        dynamo_param_name(fqn): (param - param.grad).detach()
        for fqn, param in model.named_parameters()
    }


def merged(params: dict[int, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Merge per-PP-rank parameter dicts into one."""
    return {name: value for per_rank in params.values() for name, value in per_rank.items()}
