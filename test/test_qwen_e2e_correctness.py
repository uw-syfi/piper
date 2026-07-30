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


@pytest.mark.parametrize("model", _CORRECTNESS_MODELS)
@pytest.mark.parametrize(
    ("name", "base_schedule", "schedule", "ranks", "microbatches"),
    RUNS,
    ids=[run[0] for run in RUNS],
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
    temp_dir = f"/tmp/pe2e-{os.getpid()}"
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
        "--no-use-inductor",
        "--temp-dir",
        temp_dir,
        "--ray-namespace",
        f"qwen-e2e-{os.getpid()}-{model.lower()}-{name}",
    ]
    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20 * 60,
        check=False,
    )
    assert result.returncode == 0, (
        f"Qwen {model} correctness run {name!r} failed with exit code "
        f"{result.returncode}:\n{result.stdout}"
    )
