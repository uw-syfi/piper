# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from artifact.scripts import make_piper_base_schedule
from src.schedule import load_schedule_directives


def _generate(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    *,
    virtual_stages: int = 1,
    layout: str = "linear",
    ep: bool = True,
    bucket_size: int | None = None,
) -> list[dict]:
    argv = [
        "make_piper_base_schedule.py",
        "--pp",
        "2",
        "--dp",
        "16",
        "--virtual-stages",
        str(virtual_stages),
        "--layout",
        layout,
        "--zero-stage",
        "3",
        "--output",
        str(output),
    ]
    if ep:
        argv.append("--ep")
    if bucket_size is not None:
        argv.extend(["--bucket-size", str(bucket_size)])
    monkeypatch.setattr(sys, "argv", argv)
    assert make_piper_base_schedule.main() == 0
    return load_schedule_directives(str(output))


@pytest.mark.parametrize("ep", [False, True])
def test_zero3_omits_unspecified_bucket_size_and_uses_dense_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ep: bool,
) -> None:
    directives = _generate(monkeypatch, tmp_path / "schedule.json", ep=ep)
    replicate = [d for d in directives if d["op"] == "replicate"]

    assert replicate
    assert all(d["filter"]["DP"] == "-" for d in replicate)
    assert all((d["filter"].get("EP") == "-") == ep for d in replicate)
    assert all(d["shard_params"] and d["shard_grads"] for d in replicate)
    assert all("bucket_size" not in d for d in replicate)


def test_zero3_preserves_explicit_bucket_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directives = _generate(
        monkeypatch,
        tmp_path / "schedule.json",
        bucket_size=128,
    )

    replicate = [d for d in directives if d["op"] == "replicate"]
    assert replicate
    assert all(d["bucket_size"] == 128 for d in replicate)
