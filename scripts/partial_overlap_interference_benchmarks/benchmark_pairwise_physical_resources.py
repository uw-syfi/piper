#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Collect frozen pairwise physical-resource points with Hopper NVML GPM.

This diagnostic-only runner does no geometry search.  It consumes the frozen
representative selection and the final fixed offsets from the timing corpus.
Every logical point executes a configurable number of fixed-count warmup and
measured bursts (three each by default). Each burst repeats the exact operation/pair long enough for
the documented GPM sampling interval, with a fixed non-overlapping period.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from scripts.partial_overlap_interference_benchmarks.benchmark_pairwise_overlap_cases import (
    BASE_SEQ,
    CLOCK_CALIBRATION_PROBES,
    PairwiseOverlapRank,
    communicator_plan,
    load_a2a_profiles,
    load_cases,
)
from scripts.partial_overlap_interference_benchmarks.pairwise_overlap_benchmark import (
    PPOuterTopology,
    assert_default_nccl_environment,
    nccl_debug_environment,
    sanitize_nccl_environment,
    validate_observed_placement,
)
from scripts.partial_overlap_interference_benchmarks.physical_resource_metrics import (
    NvmlGpmCollector,
    ib_counter_deltas,
    physical_metric_metadata,
    read_ib_port_data_counters,
)

WARMUPS = 3
MEASURED_TRIALS = 3
MINIMUM_BURST_US = 120_000.0
# Warmup-to-measured jitter was 2.8% in the P2P smoke.  A fixed 15% repetition
# margin keeps every active endpoint above the 120 ms contract without adapting
# measured trials or changing the exact three-trial design.
ACTIVE_BURST_SAFETY_FACTOR = 1.15


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, document: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def frozen_delays(
    samples_path: Path, concurrent_rows: list[dict[str, str]]
) -> dict[str, dict[int, float]]:
    requested = {
        (row["case_key"], row["dispatch_mode"], float(row["requested_overlap_pct"]))
        for row in concurrent_rows
    }
    values: dict[tuple[str, str, float], dict[int, list[float]]] = {}
    for row in read_csv(samples_path):
        key = (
            row["case_key"], row["dispatch_mode"],
            float(row["requested_overlap_pct"]),
        )
        if key not in requested or row.get("analysis_included") != "True":
            continue
        evidence = json.loads(row["delay_rebase_evidence"])
        delays = evidence["final_delays_ms"]
        by_index = values.setdefault(key, {})
        for index, value in delays.items():
            by_index.setdefault(int(index), []).append(float(value))
    result = {}
    for row in concurrent_rows:
        key = (
            row["case_key"], row["dispatch_mode"],
            float(row["requested_overlap_pct"]),
        )
        if key not in values:
            raise RuntimeError(f"no frozen delay evidence for {key}")
        result[row["point_id"]] = {
            index: statistics.median(candidates)
            for index, candidates in values[key].items()
        }
    return result


def prepare_points(
    concurrent_path: Path,
    isolated_path: Path,
    samples_path: Path,
    assignments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    concurrent = read_csv(concurrent_path)
    isolated = read_csv(isolated_path)
    delays = frozen_delays(samples_path, concurrent)
    signature_anchors = {
        row["signature_id"]: float(row["anchor_duration_us"])
        for row in isolated
    }
    points = []
    for row in isolated:
        assignment = assignments[row["source_case_key"]]
        matches = [
            index for index, participant in enumerate(assignment["participants"])
            if participant["event_id"] == row["event_id"]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{row['signature_id']} resolves to operation indices {matches}"
            )
        repeat_count = max(
            1, math.ceil(MINIMUM_BURST_US / float(row["anchor_duration_us"]))
        )
        points.append({
            **row,
            "point_id": f"ISOLATED::{row['signature_id']}",
            "measurement_scope": "isolated",
            "case_key": row["source_case_key"],
            "active_indices": matches,
            "order": matches,
            "delays_ms": {matches[0]: 0.0},
            "repeat_count": repeat_count,
            "expected_span_us": float(row["anchor_duration_us"]),
        })
    for row in concurrent:
        victim = int(row["victim_index"])
        assignment = assignments[row["case_key"]]
        if len(assignment["participants"]) != 2:
            raise RuntimeError(f"physical point must be pairwise: {row['point_id']}")
        aggressor = 1 - victim
        order = [victim, aggressor] if row["dispatch_mode"] == "victim_first" else [aggressor, victim]
        expected_span = max(
            signature_anchors[row["isolated_victim_signature_id"]],
            signature_anchors[row["isolated_aggressor_signature_id"]],
        )
        points.append({
            **row,
            "measurement_scope": "concurrent",
            "active_indices": [0, 1],
            "order": order,
            "delays_ms": delays[row["point_id"]],
            "repeat_count": max(1, math.ceil(MINIMUM_BURST_US / expected_span)),
            "expected_span_us": expected_span,
            "aggressor_index": aggressor,
        })
    return points


class PhysicalDiagnosticRank(PairwiseOverlapRank):
    def run_physical_point(
        self,
        assignment: dict[str, Any],
        point: dict[str, Any],
        seq_len: int,
        a2a_profiles: dict[str, Any],
        warmup_trials: int = WARMUPS,
        measured_trials: int = MEASURED_TRIALS,
    ) -> dict[str, Any]:
        import torch.distributed as dist

        torch = self.torch
        participants = assignment["participants"]
        operations = [self._operation(row, seq_len, a2a_profiles) for row in participants]
        streams = [torch.cuda.Stream(priority=0) if op is not None else None for op in operations]
        control_stream = torch.cuda.Stream(priority=-1)
        active = [int(index) for index in point["active_indices"]]
        order = [int(index) for index in point["order"]]
        delays = {int(key): float(value) for key, value in point["delays_ms"].items()}
        repeat_count = int(point["repeat_count"])
        maximum_delay_ms = max(delays.values(), default=0.0)
        period_ms = maximum_delay_ms + float(point["expected_span_us"]) / 1000.0 * 1.35 + 0.25
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        operation_active = any(operations[index] is not None for index in active)
        collector = NvmlGpmCollector(gpu_uuid).open() if operation_active else None
        designated_ib_reader = self.global_rank in {0, 1}

        def wait_global(target_ns: int) -> None:
            while True:
                remaining = target_ns - self._global_now_ns()
                if remaining <= 0:
                    return
                if remaining > 500_000:
                    time.sleep((remaining - 200_000) / 1e9)

        def burst(trial_kind: str, trial_index: int, measured: bool) -> dict[str, Any]:
            dist.barrier()
            latest = torch.tensor([self._global_now_ns()], dtype=torch.int64)
            dist.all_reduce(latest, op=dist.ReduceOp.MAX)
            origin_ns = int(latest.item()) + 5_000_000
            anchor = torch.cuda.Event(enable_timing=True)
            before_ns = time.monotonic_ns()
            anchor.record()
            anchor.synchronize()
            anchor_global_ns = (before_ns + time.monotonic_ns()) // 2 + self.clock_offset_ns
            ib_before = read_ib_port_data_counters() if measured and designated_ib_reader else []
            metric_start = (
                collector.start() if measured and collector is not None else None
            )
            records = []
            active_gpu_us = 0.0
            for repeat_index in range(repeat_count):
                repeat_origin = origin_ns + int(round(repeat_index * period_ms * 1e6))
                wait_global(repeat_origin)
                gate = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(control_stream):
                    gate.record()
                launched = {}
                for index in order:
                    op, stream = operations[index], streams[index]
                    if op is None or stream is None:
                        continue
                    wait_global(repeat_origin + int(round(delays.get(index, 0.0) * 1e6)))
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    stream.wait_event(gate)
                    with torch.cuda.stream(stream):
                        start.record()
                        result = op["launch"]()
                        if op["kind"] == "compute":
                            end.record()
                    launched[index] = (op, stream, start, end, result)
                for index in order:
                    if index not in launched:
                        continue
                    op, stream, start, end, result = launched[index]
                    if op["kind"] == "communication":
                        with torch.cuda.stream(stream):
                            works = result if isinstance(result, (list, tuple)) else [result]
                            for work in works:
                                work.wait()
                            end.record()
                for index, (op, _stream, start, end, _result) in launched.items():
                    end.synchronize()
                    duration_ms = float(start.elapsed_time(end))
                    start_ns = int(anchor_global_ns + anchor.elapsed_time(start) * 1e6)
                    end_ns = int(anchor_global_ns + anchor.elapsed_time(end) * 1e6)
                    records.append({
                        "trial_kind": trial_kind,
                        "trial_index": trial_index,
                        "repeat_index": repeat_index,
                        "operation_index": index,
                        "participant_id": participants[index]["participant_id"],
                        "category": participants[index]["category"],
                        "global_rank": self.global_rank,
                        "duration_us": duration_ms * 1000.0,
                        "gpu_start_global_ns": start_ns,
                        "gpu_end_global_ns": end_ns,
                    })
                    active_gpu_us += duration_ms * 1000.0
            torch.cuda.synchronize()
            physical = (
                collector.finish(metric_start)
                if metric_start is not None and collector is not None else None
            )
            ib_after = read_ib_port_data_counters() if measured and designated_ib_reader else []
            ib = ib_counter_deltas(ib_before, ib_after) if measured and designated_ib_reader else []
            if measured and physical is not None and not physical["gpm_interval_valid"]:
                raise RuntimeError(
                    f"GPM interval too short: {physical['gpm_interval_us']} us"
                )
            return {
                "global_rank": self.global_rank,
                "node_id": self.node_id,
                "node_ip": self.node_ip,
                "gpu_uuid": gpu_uuid,
                "operation_active": operation_active,
                "records": records,
                "physical": physical,
                "ib_deltas": ib,
                "repeat_count": repeat_count,
                "period_ms": period_ms,
                "active_duration_us_sum": active_gpu_us,
            }

        try:
            warmups = [burst("warmup", index, False) for index in range(warmup_trials)]
            # Trace anchors can substantially overestimate short P2P endpoints.  Calibrate
            # the measured burst geometry from the work that actually ran in all three
            # warmups.  Active-union time controls repetition count, while pair span controls
            # the serialized repetition period.  Both are durations computed locally before
            # cross-rank reductions; cross-host clock offsets therefore cannot bias them.
            def interval_union_ns(records: list[dict[str, Any]]) -> int:
                intervals = sorted(
                    (int(row["gpu_start_global_ns"]), int(row["gpu_end_global_ns"]))
                    for row in records
                )
                total = 0
                union_start = union_end = None
                for start_ns, end_ns in intervals:
                    if union_start is None:
                        union_start, union_end = start_ns, end_ns
                    elif start_ns <= union_end:
                        union_end = max(union_end, end_ns)
                    else:
                        total += union_end - union_start
                        union_start, union_end = start_ns, end_ns
                if union_start is not None:
                    total += union_end - union_start
                return total

            warmup_local_active_union_us = []
            warmup_local_pair_span_us = []
            for warmup in warmups:
                by_repeat: dict[int, list[dict[str, Any]]] = {}
                for record in warmup["records"]:
                    by_repeat.setdefault(int(record["repeat_index"]), []).append(record)
                repeat_active_us = []
                repeat_span_us = []
                for repeat_index in range(int(warmup["repeat_count"])):
                    records = by_repeat.get(repeat_index, [])
                    if records:
                        repeat_active_us.append(interval_union_ns(records) / 1000.0)
                        repeat_span_us.append(
                            (max(int(row["gpu_end_global_ns"]) for row in records)
                             - min(int(row["gpu_start_global_ns"]) for row in records))
                            / 1000.0
                        )
                if repeat_active_us:
                    warmup_local_active_union_us.append(statistics.median(repeat_active_us))
                    warmup_local_pair_span_us.append(statistics.median(repeat_span_us))
            # NCCL's first use may pay communicator/channel setup that is absent from
            # steady state.  Three warmup bursts let the median reject that one-time cost.
            local_active_us = (
                statistics.median(warmup_local_active_union_us)
                if warmup_local_active_union_us else float(2**62)
            )
            local_span_us = (
                statistics.median(warmup_local_pair_span_us)
                if warmup_local_pair_span_us else 0.0
            )
            min_active = torch.tensor([local_active_us], dtype=torch.float64)
            max_span = torch.tensor([local_span_us], dtype=torch.float64)
            dist.all_reduce(min_active, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_span, op=dist.ReduceOp.MAX)
            min_active_union_us = float(min_active.item())
            observed_span_us = float(max_span.item())
            if min_active_union_us <= 0 or min_active_union_us >= float(2**61):
                raise RuntimeError("warmups contained no active operation intervals")
            if observed_span_us <= 0:
                raise RuntimeError("warmups contained no active operation timestamps")
            period_ms = observed_span_us / 1000.0 + 0.25
            repeat_count = max(
                1,
                math.ceil(
                    MINIMUM_BURST_US * ACTIVE_BURST_SAFETY_FACTOR
                    / min_active_union_us
                ),
            )
            measured = [burst("measured", index + 1, True) for index in range(measured_trials)]
            return {
                "global_rank": self.global_rank,
                "warmup_trial_count": len(warmups),
                "measured_trial_count": len(measured),
                "warmup_calibrated_observed_span_us": observed_span_us,
                "warmup_calibrated_min_active_union_us": min_active_union_us,
                "warmup_local_active_union_us": warmup_local_active_union_us,
                "warmup_local_pair_span_us": warmup_local_pair_span_us,
                "measured_repeat_count": repeat_count,
                "measured_period_ms": period_ms,
                "measured": measured,
            }
        finally:
            if collector is not None:
                collector.close()
            del operations
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="10.65.24.131:6379")
    parser.add_argument("--node-ips", default="10.65.24.131,10.65.16.181")
    parser.add_argument("--matrix", type=Path, default=Path("../out/20260809_pairwise_overlap_foundation/EXACT_KERNEL_EXECUTION_MATRIX.json"))
    parser.add_argument("--a2a-profiles", type=Path, default=Path("../out/20260809_pairwise_overlap_foundation/ragged_a2a_payloads.json"))
    parser.add_argument("--concurrent", type=Path, default=Path("../out/interference-benchmarks/physical-resource-diagnostics/selected_concurrent_points.csv"))
    parser.add_argument("--isolated", type=Path, default=Path("../out/interference-benchmarks/physical-resource-diagnostics/isolated_operation_signatures.csv"))
    parser.add_argument("--samples", type=Path, default=Path("../out/interference-benchmarks/merged-results/tables/samples.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--measured-trials", type=int, default=MEASURED_TRIALS)
    parser.add_argument(
        "--only-point", action="append",
        help="Run only this point id; repeat the option to run a small smoke subset",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    audit = sanitize_nccl_environment(phase="physical_resource_driver_before_ray_or_torch_import")
    assert_default_nccl_environment()
    args = parse_args()
    if args.warmups < 1 or args.measured_trials < 1:
        raise ValueError("--warmups and --measured-trials must both be positive")
    concurrent = read_csv(args.concurrent)
    isolated = read_csv(args.isolated)
    case_keys = sorted({row["case_key"] for row in concurrent} | {row["source_case_key"] for row in isolated})
    assignments_list = load_cases(args.matrix, case_keys)
    assignments = {row["case_key"]: row for row in assignments_list}
    points = prepare_points(args.concurrent, args.isolated, args.samples, assignments)
    if args.only_point:
        requested = set(args.only_point)
        points = [row for row in points if row["point_id"] in requested]
        missing = requested - {row["point_id"] for row in points}
        if missing:
            raise ValueError(f"unknown --only-point values: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "collection_manifest.json"
    checkpoint_path = args.output_dir / "checkpoint.json"
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "warmups_per_point": args.warmups,
        "measured_trials_per_point": args.measured_trials,
        "minimum_burst_us": MINIMUM_BURST_US,
        "active_burst_safety_factor": ACTIVE_BURST_SAFETY_FACTOR,
        "manual_metric_pass_multiplier": 1,
        "points": points,
        "physical_metric_metadata": physical_metric_metadata(),
        "nccl_environment_audit": audit,
    }
    atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(manifest_path)
        return

    checkpoint = {"schema_version": 1, "completed": {}, "points": [row["point_id"] for row in points]}
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    class GlobalClock:
        def now_ns(self) -> int:
            return time.monotonic_ns()

    node_ips = tuple(item.strip() for item in args.node_ips.split(","))
    topology = PPOuterTopology(node_ips=node_ips)
    comm_plan = communicator_plan(assignments_list)
    ray.init(address=args.address, runtime_env={"env_vars": nccl_debug_environment()}, log_to_driver=True)
    pg = placement_group(
        [{"CPU": 8, "GPU": 8, f"node:{node_ip}": 0.001} for node_ip in node_ips],
        strategy="STRICT_SPREAD",
    )
    actors = []
    clock = None
    try:
        ray.get(pg.ready(), timeout=300)
        actor_type = ray.remote(num_cpus=1, num_gpus=1, max_restarts=0)(PhysicalDiagnosticRank)
        for rank in range(topology.world_size):
            pp = topology.physical_pp_rank(rank)
            actors.append(actor_type.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=pp
            )).remote(rank, topology.node_ip_for_rank(rank)))
        placement = ray.get([actor.metadata.remote() for actor in actors], timeout=180)
        validate_observed_placement(topology, placement)
        master_addr, master_port = ray.get(actors[0].rendezvous.remote())
        ray.get([actor.initialize.remote(master_addr, master_port, topology.__dict__, comm_plan) for actor in actors], timeout=900)
        clock_type = ray.remote(num_cpus=0, max_concurrency=32)(GlobalClock)
        clock = clock_type.remote()
        ray.get([actor.calibrate_clock.remote(clock, CLOCK_CALIBRATION_PROBES, "physical_suite") for actor in actors], timeout=180)
        for index, point in enumerate(points, 1):
            if point["point_id"] in checkpoint["completed"]:
                continue
            print(f"[{index}/{len(points)}] {point['point_id']}", flush=True)
            result = ray.get([
                actor.run_physical_point.remote(
                    assignments[point["case_key"]], point, BASE_SEQ,
                    load_a2a_profiles(args.a2a_profiles),
                    args.warmups, args.measured_trials,
                ) for actor in actors
            ], timeout=900)
            point_path = args.output_dir / "points" / f"{point['point_id'].replace(':', '_')}.json"
            atomic_json(point_path, {"point": point, "rank_results": result})
            checkpoint["completed"][point["point_id"]] = str(point_path)
            atomic_json(checkpoint_path, checkpoint)
    finally:
        try:
            shutdowns = [actor.shutdown.remote() for actor in actors]
            ray.get(shutdowns, timeout=60)
        except Exception:
            pass
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if clock is not None:
            ray.kill(clock, no_restart=True)
        remove_placement_group(pg)
        ray.shutdown()
    print(checkpoint_path)


if __name__ == "__main__":
    main()
