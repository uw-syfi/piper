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

    *batches* holds one ``(inputs, labels)`` pair per DP rank. Returns the
    parameters before and after training, keyed by PP rank, and the losses.
    """
    from src.compile import piper_setup
    from src.piper import piper_exec_dag

    try:
        inputs, labels = batches[dp_rank()]
        setup_kwargs.setdefault("model_dtype", torch.float32)
        setup_kwargs.setdefault("optim_fn", lambda params: torch.optim.SGD(params, lr=1.0))
        piper_setup(
            model_class,
            example_inputs=inputs,
            example_outputs=labels,
            schedule_directives_file=schedule_path,
            **setup_kwargs,
        )
        initial = params_by_rank()
        losses = [piper_exec_dag(torch.nn.functional.mse_loss) for _ in range(steps)]
        return {"initial": initial, "losses": losses, "final": params_by_rank()}
    finally:
        kill_actors()
