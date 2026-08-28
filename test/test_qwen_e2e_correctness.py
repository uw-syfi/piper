# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.run_qwen_examples import RUNS


pytestmark = pytest.mark.gpu
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORRECTNESS_MODELS = ("9M",)
# Expert parameters (names containing ``.moe.experts.``) use the 2e-4 absolute
# tolerance because sharded grouped-expert execution changes kernel padding,
# layout, and BF16 accumulation order. Everything else uses zero relative
# tolerance and the larger of 1e-7 absolute error or two BF16 spacings measured
# at the initial/eager-final value; initialization itself remains bitwise exact.
_ROUTED_DENSE_ATOL = 1e-7
_ROUTED_EXPERT_ATOL = 2e-4
_AFD_RUNS = (
    (
        "afd_pp2_single_node",
        "test/fixtures/schedules/afd_e2e_pp2.json",
        "afd",
        2,
        2,
    ),
    (
        "afd_pp4_single_node",
        "test/fixtures/schedules/afd_e2e_pp4.json",
        "afd",
        4,
        2,
    ),
)
_CRITICAL_RUN_NAMES = {
    "zero3_1f1b",
    "bucket100_1f1b",
    "dualpipev",
    "interleaved_1f1b",
    "afd_pp2_single_node",
    "afd_pp4_single_node",
}
_CORRECTNESS_RUNS = tuple(
    run for run in (*RUNS, *_AFD_RUNS) if run[0] in _CRITICAL_RUN_NAMES
)


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
        "examples/run_harness.py",
        "--test-file",
        "examples/qwen_correctness.py",
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
        "4" if schedule == "afd" else "2",
        "--seq-len",
        "16",
        "--steps",
        "2",
        # SGD makes gradient-synchronization errors directly observable. Adam
        # largely cancels a uniform scale factor and its fused update can differ
        # by a BF16 ULP when Piper groups parameters into stage optimizers.
        "--optimizer",
        "sgd",
        "--no-use-inductor",
        "--temp-dir",
        temp_dir,
        "--ray-namespace",
        f"qwen-e2e-{os.getpid()}-{model.lower()}-{name}",
    ]
    # Routed regrouping changes grouped-kernel padding/accumulation order. In
    # addition, each replica accumulates its distinct BF16 gradient before the
    # per-microbatch collective, whereas the eager reference visits shards in
    # sequence. Keep initialization bitwise-exact (checked by the test program)
    # while allowing tightly bounded routed-training differences.
    command.extend(
        [
            "--rtol",
            "0",
            "--atol",
            str(_ROUTED_DENSE_ATOL),
            "--dense-max-ulps",
            "2",
            "--expert-atol",
            str(_ROUTED_EXPERT_ATOL),
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
