#!/usr/bin/env python3
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

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Piper base schedule for Qwen e2e evals.")
    parser.add_argument("--pp", type=int, required=True, help="Physical pipeline ranks.")
    parser.add_argument("--dp", type=int, required=True, help="Data-parallel replicas.")
    parser.add_argument("--virtual-stages", type=int, default=1)
    parser.add_argument("--layout", choices=("linear", "v"), default="linear")
    parser.add_argument("--zero-stage", type=int, default=1, choices=(0, 1, 2, 3))
    parser.add_argument("--ep", action="store_true")
    parser.add_argument("--bucket-size", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def devices_for_stage(stage: int, physical_pp: int, dp: int, layout: str) -> list[int]:
    if layout == "v" and stage >= physical_pp:
        physical_rank = (2 * physical_pp - 1 - stage) % physical_pp
    else:
        physical_rank = stage % physical_pp
    return [physical_rank + replica * physical_pp for replica in range(dp)]


def main() -> int:
    args = parse_args()
    if args.pp <= 0 or args.dp <= 0 or args.virtual_stages <= 0:
        raise SystemExit("--pp, --dp, and --virtual-stages must be positive")

    directives: list[dict] = []
    stage_count = args.pp * args.virtual_stages
    for stage in range(stage_count):
        devices = devices_for_stage(stage, args.pp, args.dp, args.layout)
        directives.append(
            {
                "op": "place",
                "filter": {"PP": stage},
                "devices": devices,
                "stream": "pp_stream",
            }
        )

    if args.dp > 1:
        for stage in range(stage_count):
            devices = devices_for_stage(stage, args.pp, args.dp, args.layout)
            directive = {
                "op": "replicate",
                # DP-absent regions are the dense replication domain. Qwen's
                # routed expert regions currently carry EP (but no DP) tags,
                # so retain EP-absence as a compatibility guard when expert
                # sharding is enabled.
                "filter": {
                    "PP": stage,
                    "DP": "-",
                    **({"EP": "-"} if args.ep else {}),
                },
                "devices": devices,
                "reduce_stream": "reduce_stream",
            }
            if args.zero_stage == 2:
                directive["shard_grads"] = True
            elif args.zero_stage == 3:
                directive["gather_stream"] = "gather_stream"
                directive["shard_grads"] = True
                directive["shard_params"] = True
                if args.bucket_size is not None:
                    directive["bucket_size"] = int(args.bucket_size)
            elif args.bucket_size is not None:
                directive["bucket_size"] = int(args.bucket_size)
            directives.append(directive)

    if args.ep:
        for stage in range(stage_count):
            directives.append(
                {
                    "op": "shard",
                    "filter": {"PP": stage, "EP": "*"},
                    "devices": devices_for_stage(stage, args.pp, args.dp, args.layout),
                    "stream": "ep_stream",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(directives, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
