from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.run_qwen_examples import RUNS


pytestmark = pytest.mark.gpu
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORRECTNESS_MODELS = ("9M", "1B")
_AFD_EXPERT_ATOL = 2e-4
_AFD_RUNS = (
    (
        "afd_pp2_single_node",
        "examples/base-schedules/afd_e2e_pp2.json",
        "afd",
        2,
        2,
    ),
    (
        "afd_pp4_single_node",
        "examples/base-schedules/afd_e2e_pp4.json",
        "afd",
        4,
        2,
    ),
)
_CORRECTNESS_RUNS = (*RUNS, *_AFD_RUNS)


@pytest.mark.parametrize("model", _CORRECTNESS_MODELS)
@pytest.mark.parametrize(
    ("name", "base_schedule", "schedule", "ranks", "microbatches"),
    _CORRECTNESS_RUNS,
    ids=[run[0] for run in _CORRECTNESS_RUNS],
)
def test_qwen_parameter_parity(
    model: str,
    name: str,
    base_schedule: str,
    schedule: str,
    ranks: int,
    microbatches: int,
) -> None:
    # Ray appends a long session/sockets suffix and AF_UNIX paths are limited
    # to 107 bytes on Linux, so keep this deliberately short.
    temp_dir = os.environ.get(
        "PIPER_E2E_RAY_TEMP_DIR", f"/tmp/pe2e-{os.getpid()}"
    )
    command = [
        sys.executable,
        "examples/test_harness.py",
        "--test-file",
        "examples/test_qwen_correctness.py",
        "--model",
        model,
        "--base-schedule",
        base_schedule,
        "--schedule",
        schedule,
        "--ranks",
        str(ranks),
        "--mbs",
        str(microbatches),
        "--batch-size",
        "2",
        "--seq-len",
        "16",
        "--steps",
        "2",
        # SGD makes gradient-scale errors directly observable. Adam largely
        # cancels a uniform scale factor and its fused update can also differ
        # by a BF16 ULP when Piper groups parameters into stage optimizers.
        "--optimizer",
        "sgd",
        "--no-use-inductor",
        "--temp-dir",
        temp_dir,
        "--ray-namespace",
        f"qwen-e2e-{os.getpid()}-{model.lower()}-{name}",
    ]
    if schedule == "afd":
        # Disaggregation changes the grouped-GEMM padding/layout and therefore
        # BF16 accumulation order.  Keep initialization bitwise-exact (checked
        # by the test program) while allowing the observed one-ULP training
        # difference in expert parameters.
        command.extend(
            [
                "--rtol",
                "0",
                "--expert-atol",
                str(_AFD_EXPERT_ATOL),
            ]
        )
    ray_address = os.environ.get("PIPER_E2E_RAY_ADDRESS")
    if ray_address:
        host, separator, port = ray_address.rpartition(":")
        if not separator or not host or not port.isdigit():
            raise ValueError(
                "PIPER_E2E_RAY_ADDRESS must have the form HOST:PORT, got "
                f"{ray_address!r}"
            )
        command.extend(["--address", host, "--port", port])
    placement_node = os.environ.get("PIPER_E2E_PLACEMENT_NODE_ADDRESS")
    if placement_node:
        command.extend(["--placement-node-address", placement_node])
    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=2 * 60,
        check=False,
    )
    assert result.returncode == 0, (
        f"Qwen {model} correctness run {name!r} failed with exit code "
        f"{result.returncode}:\n{result.stdout}"
    )
