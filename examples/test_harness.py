"""Build a schedule JSON and run a model test with it.

This is a small harness around model example modules such as
``examples.test_qwen`` and ``examples.test_llama``. It consumes schedule/runtime
arguments, parses the selected test module's arguments, starts Ray, creates the
Piper placement group, and runs the model through ``PiperProgramCoordinator``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import ray
from ray.util.placement_group import placement_group_table

from src.coordinator import PiperProgramCoordinator, create_piper_placement_group
from src.schedule import load_schedule_directives
from src.visualization import visualize_order_directives

from build_schedule import (
    build_1f1b_schedule,
    build_dualpipev_schedule,
    build_interleaved_1f1b_schedule,
    build_zerobubble_schedule,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a full schedule JSON from a base schedule and generated order "
            "directives, then run a model test with it."
        )
    )
    parser.add_argument(
        "--test-file",
        "--test",
        required=True,
        help=(
            "Example file to run directly, e.g. examples/test_qwen.py."
        ),
    )
    parser.add_argument(
        "--base-schedule",
        required=True,
        type=Path,
        help="Input JSON schedule without order directives.",
    )
    parser.add_argument(
        "--schedule",
        "--schedule-name",
        dest="schedule",
        required=True,
        choices=("1f1b", "interleaved_1f1b", "zerobubble", "dualpipev", "custom"),
        help=(
            "Order directive schedule variant to append. Use 'custom' to keep the "
            "split/order directives already in --base-schedule untouched."
        ),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=None,
        help="Number of PP ranks. Required unless --schedule custom.",
    )
    parser.add_argument(
        "--mbs",
        type=int,
        default=None,
        help="Number of microbatches. Required unless --schedule custom.",
    )
    parser.add_argument(
        "--virtual-stages",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-existing-order",
        action="store_true",
        help="Keep any order directives already present in the base schedule.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the generated schedule and print the test command without running it.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Render the generated schedule and per-rank TrainingDAG visualizations.",
    )
    parser.add_argument(
        "--pytorch-profiler",
        action="store_true",
        help="Run extra iterations under torch.profiler and combine the per-actor "
             "chrome traces into one trace per dp-rank under out/.",
    )
    parser.add_argument(
        "--address",
        default="",
        help="Ray head address to connect to. Omit to start a local Ray session.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4567,
        help="Ray head port to connect to when --address is set.",
    )
    parser.add_argument(
        "--temp-dir",
        default="/tmp/piper/ray_tmp",
        help="Ray temp directory.",
    )
    parser.add_argument(
        "--ray-namespace",
        default=None,
        help="Ray namespace. Defaults to the selected test module name.",
    )
    parser.add_argument(
        "--num-standby",
        type=int,
        default=0,
        help="Number of standby DP ranks to reserve (one extra GPU bundle each).",
    )
    args, test_args = parser.parse_known_args()

    if args.schedule != "custom":
        missing = [
            flag for flag, value in (("--ranks", args.ranks), ("--mbs", args.mbs))
            if value is None
        ]
        if missing:
            parser.error(
                f"the following arguments are required unless --schedule custom: {', '.join(missing)}"
            )
        if args.virtual_stages is not None:
            parser.error(
                "--virtual-stages is internal; it is 2 for interleaved_1f1b/dualpipev "
                "and 1 otherwise"
            )
        args.virtual_stages = _virtual_stages_for_schedule(args.schedule)

    # Each run gets its own timestamped output directory under out/, holding its
    # results.csv (one row per DP rank), schedule viz, and combined profiles.
    run_dir = Path("out") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    generated_schedule, viz_path = build_schedule_file(args, run_dir)
    forwarded_args = _replace_schedule_arg(test_args, generated_schedule)
    if args.viz:
        forwarded_args.append("--viz")

    profile_dir: Path | None = None
    if args.pytorch_profiler and not args.dry_run:
        # Absolute path on the shared filesystem so remote actors (possibly on
        # other nodes) write to the same place the harness later reads.
        profile_dir = (run_dir / "pytorch_profiles").resolve()
        if profile_dir.exists():
            for stale in profile_dir.glob("dp*_pp*.json"):
                stale.unlink()
        profile_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run_args = list(forwarded_args)
        if args.pytorch_profiler:
            dry_run_args.append("--pytorch-profiler")
        _validate_test_args(args.test_file, dry_run_args)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--test-file",
            args.test_file,
            "--base-schedule",
            str(args.base_schedule),
            "--schedule",
            args.schedule,
        ]
        if args.ranks is not None:
            cmd.extend(["--ranks", str(args.ranks)])
        if args.mbs is not None:
            cmd.extend(["--mbs", str(args.mbs)])
        cmd.extend(dry_run_args)
        print(" ".join(cmd))
        print(f"generated_schedule={generated_schedule}")
        if viz_path is not None:
            print(f"schedule_viz={viz_path}")
        return

    if args.pytorch_profiler:
        forwarded_args.append("--pytorch-profiler")
    dp_metrics = _run_test_module(args, forwarded_args, profile_dir=profile_dir)
    if dp_metrics:
        rows = _metrics_rows(dp_metrics)
        results_csv = run_dir / "results.csv"
        _write_results_csv(results_csv, rows)
        print(f"metrics written to {results_csv}")
    if profile_dir is not None:
        _combine_pytorch_profiles(
            profile_dir,
            run_dir,
            schedule=args.schedule,
            pp=args.ranks if args.ranks is not None else "X",
            mbs=args.mbs if args.mbs is not None else "X",
        )


def _validate_test_args(test_file: str, test_args: list[str]) -> None:
    _, test_module = _load_test_file(test_file)
    test_module.parse_args(test_args)


def _run_test_module(
    args: argparse.Namespace,
    test_args: list[str],
    *,
    profile_dir: Path | None = None,
) -> list[dict]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    module_name, test_module = _load_test_file(args.test_file)
    test_ns = test_module.parse_args(test_args)
    if hasattr(test_ns, "temp_dir"):
        test_ns.temp_dir = args.temp_dir
    test_ns.profile_dir = str(profile_dir) if profile_dir is not None else ""
    if hasattr(test_module, "_derive_num_stages"):
        test_ns.num_stages = test_module._derive_num_stages(test_ns.schedule_directives_file)

    namespace = args.ray_namespace or module_name.rsplit(".", 1)[-1].replace("test_", "")
    ray.init(
        address=f"{args.address}:{args.port}" if args.address else None,
        namespace=namespace,
        log_to_driver=True,
        include_dashboard=False,
        _temp_dir=args.temp_dir,
    )
    try:
        pp_outer = getattr(test_ns, "pp_outer", False)
        num_standby = getattr(args, "num_standby", 0)
        pg = create_piper_placement_group(
            test_ns.schedule_directives_file, pp_outer=pp_outer,
            num_standby=num_standby,
        )
        ray.get(pg.ready(), timeout=600)
        logging.getLogger(module_name).info(placement_group_table(pg))
        coordinator = PiperProgramCoordinator.remote(
            schedule_directives_file=test_ns.schedule_directives_file,
            pp_outer=pp_outer,
            num_standby=num_standby,
        )
        handles = coordinator.run_program.remote(test_module.main, pg, test_ns, pg)
        dp_metrics = ray.get(handles)
        return dp_metrics
    finally:
        ray.shutdown()


def _combine_pytorch_profiles(
    profile_dir: Path,
    out_dir: Path,
    schedule: str,
    pp: int | str,
    mbs: int | str,
) -> None:
    """Group per-actor chrome traces (dp{dp}_pp{pp}.json) by dp-rank and merge
    each group into one trace per dp-rank under out_dir. The combined filename
    encodes the run configuration: schedule, pp degree, dp degree, mbs, dp-rank.

    Each pp-rank's events keep their own pid namespace via a collision-free
    remap, and process_name metadata is prefixed with the pp rank so the merged
    timeline groups work by stage. The intermediate per-actor traces and the
    (now-empty) profile_dir are removed once combining succeeds.
    """
    name_re = re.compile(r"^dp(\d+)_pp(\d+)\.json$")
    groups: dict[int, dict[int, Path]] = {}
    for path in sorted(profile_dir.glob("dp*_pp*.json")):
        m = name_re.match(path.name)
        if not m:
            continue
        dp_rank, pp_rank = int(m.group(1)), int(m.group(2))
        groups.setdefault(dp_rank, {})[pp_rank] = path

    if not groups:
        print(f"pytorch profiler: no trace files found in {profile_dir}")
        return

    dp_degree = len(groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    consumed: list[Path] = []
    for dp_rank, pp_paths in sorted(groups.items()):
        combined_events: list = []
        next_pid = 0
        for pp_rank, path in sorted(pp_paths.items()):
            with path.open(encoding="utf-8") as f:
                trace = json.load(f)
            events = trace.get("traceEvents", [])
            annotated = _annotate_cuda_events_with_dag_labels(events)
            pid_map: dict = {}
            for ev in events:
                opid = ev.get("pid")
                if opid is not None and opid not in pid_map:
                    pid_map[opid] = next_pid
                    next_pid += 1
            for ev in events:
                if "pid" in ev and ev["pid"] in pid_map:
                    ev["pid"] = pid_map[ev["pid"]]
                if ev.get("name") == "process_name":
                    pname = ev.get("args", {}).get("name", "")
                    ev["args"]["name"] = f"dp{dp_rank} pp{pp_rank}: {pname}"
                combined_events.append(ev)
            consumed.append(path)
            if annotated:
                print(
                    f"pytorch profiler: annotated {annotated} CUDA event(s) "
                    f"with DAG labels for dp{dp_rank} pp{pp_rank}"
                )
        out_name = (
            f"pytorch_profile_{schedule}_pp{pp}_dp{dp_degree}"
            f"_mbs{mbs}_dprank{dp_rank}.json"
        )
        out_path = out_dir / out_name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"traceEvents": combined_events}, f)
        print(
            f"pytorch profiler: combined {len(pp_paths)} pp-rank trace(s) "
            f"-> {out_path}"
        )

    for path in consumed:
        path.unlink(missing_ok=True)
    try:
        profile_dir.rmdir()
    except OSError:
        pass


def _is_dag_node_label(name: object) -> bool:
    return isinstance(name, str) and ":uid" in name and not name.startswith("##")


def _annotate_cuda_events_with_dag_labels(events: list[dict]) -> int:
    """Prefix GPU-side profiler events with their enclosing Piper DAG node.

    ``torch.profiler`` emits Piper ``record_function`` ranges on the CPU thread,
    while CUDA kernels and copies appear on GPU lanes with PyTorch/Inductor
    names such as ``## Call CompiledFxGraph ...``.  CPU launch events and GPU
    events share ``args["External id"]``; use that to carry the enclosing DAG
    node label onto the GPU events in the combined Chrome trace.  Then rewrite
    nested non-DAG GPU annotations so the visible annotation stack stays rooted
    in Piper's TrainingDAG labels instead of Inductor's opaque graph hashes.
    """
    ranges_by_thread: dict[tuple[object, object], list[tuple[float, float, str]]] = {}
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") != "user_annotation":
            continue
        label = ev.get("name")
        if not _is_dag_node_label(label):
            continue
        ts = ev.get("ts")
        dur = ev.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        ranges_by_thread.setdefault((ev.get("pid"), ev.get("tid")), []).append(
            (float(ts), float(ts) + float(dur), label)
        )
    for ranges in ranges_by_thread.values():
        ranges.sort(key=lambda item: (item[0], -item[1]))

    external_id_to_label: dict[object, str] = {}
    active: dict[tuple[object, object], list[tuple[float, str]]] = {}
    cpu_events = sorted(
        (
            ev for ev in events
            if ev.get("ph") == "X"
            and ev.get("cat") != "kernel"
            and str(ev.get("cat", "")).startswith(("cpu_op", "user_annotation"))
        ),
        key=lambda ev: float(ev.get("ts", 0.0) or 0.0),
    )

    range_idx = {key: 0 for key in ranges_by_thread}
    for ev in cpu_events:
        ts = ev.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        key = (ev.get("pid"), ev.get("tid"))
        ranges = ranges_by_thread.get(key)
        if not ranges:
            continue
        stack = active.setdefault(key, [])
        while stack and stack[-1][0] < float(ts):
            stack.pop()
        idx = range_idx[key]
        while idx < len(ranges) and ranges[idx][0] <= float(ts):
            _start, end, label = ranges[idx]
            if end >= float(ts):
                stack.append((end, label))
            idx += 1
        range_idx[key] = idx
        if not stack:
            continue
        ext_id = (ev.get("args") or {}).get("External id")
        if ext_id is not None:
            external_id_to_label[ext_id] = stack[-1][1]

    annotated = 0
    cuda_event_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") not in cuda_event_cats:
            continue
        args = ev.setdefault("args", {})
        ext_id = args.get("External id")
        label = external_id_to_label.get(ext_id)
        if not label:
            continue
        name = ev.get("name", "")
        prefix = f"{label}::"
        if isinstance(name, str) and name.startswith(prefix):
            continue
        args.setdefault("piper_dag_node", label)
        args.setdefault("original_cuda_name", name)
        ev["name"] = f"{prefix}{name}"
        annotated += 1
    annotated += _rewrite_nested_gpu_annotations(events)
    return annotated


def _rewrite_nested_gpu_annotations(events: list[dict]) -> int:
    gpu_annotations = [
        ev for ev in events
        if ev.get("ph") == "X" and ev.get("cat") == "gpu_user_annotation"
    ]
    dag_ranges_by_lane: dict[tuple[object, object], list[tuple[float, float, str]]] = {}
    for ev in gpu_annotations:
        label = ev.get("name")
        if not _is_dag_node_label(label):
            continue
        ts = ev.get("ts")
        dur = ev.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        dag_ranges_by_lane.setdefault((ev.get("pid"), ev.get("tid")), []).append(
            (float(ts), float(ts) + float(dur), label)
        )
    for ranges in dag_ranges_by_lane.values():
        ranges.sort(key=lambda item: (item[0], item[1] - item[0]))

    cuda_events_by_lane: dict[tuple[object, object], list[dict]] = {}
    for ev in events:
        if ev.get("ph") == "X" and ev.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}:
            cuda_events_by_lane.setdefault((ev.get("pid"), ev.get("tid")), []).append(ev)
    for lane_events in cuda_events_by_lane.values():
        lane_events.sort(key=lambda ev: float(ev.get("ts", 0.0) or 0.0))

    rewritten = 0
    for ev in gpu_annotations:
        name = ev.get("name")
        if _is_dag_node_label(name):
            continue
        ts = ev.get("ts")
        dur = ev.get("dur")
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        start = float(ts)
        end = start + float(dur)
        lane = (ev.get("pid"), ev.get("tid"))
        label = _enclosing_dag_gpu_label(dag_ranges_by_lane.get(lane, []), start, end)
        if label is None:
            label = _contained_cuda_dag_label(
                cuda_events_by_lane.get(lane, []), start, end
            )
        if label is None:
            continue
        args = ev.setdefault("args", {})
        args.setdefault("original_gpu_annotation", name)
        args.setdefault("piper_dag_node", label)
        ev["name"] = label
        rewritten += 1
    return rewritten


def _enclosing_dag_gpu_label(
    dag_ranges: list[tuple[float, float, str]], start: float, end: float
) -> str | None:
    label = None
    best_duration = None
    for dag_start, dag_end, dag_label in dag_ranges:
        if dag_start > start:
            break
        if dag_end < end:
            continue
        duration = dag_end - dag_start
        if best_duration is None or duration < best_duration:
            label = dag_label
            best_duration = duration
    return label


def _contained_cuda_dag_label(
    cuda_events: list[dict], start: float, end: float
) -> str | None:
    labels: dict[str, float] = {}
    for ev in cuda_events:
        ev_start = ev.get("ts")
        ev_dur = ev.get("dur")
        if not isinstance(ev_start, (int, float)) or not isinstance(ev_dur, (int, float)):
            continue
        ev_start = float(ev_start)
        if ev_start > end:
            break
        ev_end = ev_start + float(ev_dur)
        if ev_end < start:
            continue
        label = (ev.get("args") or {}).get("piper_dag_node")
        if not label:
            continue
        overlap = max(0.0, min(end, ev_end) - max(start, ev_start))
        labels[label] = labels.get(label, 0.0) + overlap
    if not labels:
        return None
    return max(labels.items(), key=lambda item: item[1])[0]


def _metrics_rows(dp_metrics: list[dict]) -> list[dict]:
    """Summarize the raw metrics into one CSV row per DP rank (no averaging
    across DP ranks).

    Each row's timing scalars come from that DP rank's own iter times; peak
    memory is taken from that DP rank's per-(global=pp)-rank values and emitted
    as one peak_memory_pp{i}_gb column per pp rank.
    """
    rows: list[dict] = []
    for m in sorted(dp_metrics, key=lambda d: int(d.get("dp_rank", 0))):
        iter_times = m.get("iter_times_s") or []
        mean_iter = statistics.fmean(iter_times) if iter_times else float("nan")
        std_iter = statistics.pstdev(iter_times) if len(iter_times) > 1 else 0.0
        tokens = (
            (m.get("batch_size") or 0)
            * (m.get("num_microbatches") or 1)
            * (m.get("seq_len") or 0)
        )
        throughput = tokens / mean_iter if iter_times and mean_iter else float("nan")

        row: dict = {
            "dp_rank": m.get("dp_rank"),
            "model": m.get("model"),
            "schedule": m.get("schedule"),
            "pp": m.get("pp"),
            "dp": m.get("dp"),
            "batch_size": m.get("batch_size"),
            "num_microbatches": m.get("num_microbatches"),
            "seq_len": m.get("seq_len"),
            "samples": len(iter_times),
            "iter_time_mean_s": mean_iter,
            "iter_time_std_s": std_iter,
            "throughput_tokens_per_s": throughput,
        }
        # peak_memory_by_rank is keyed by global rank; within a DP replica the
        # global ranks ascend with pp stage (both pp-inner and pp-outer layouts),
        # so enumerate them as local stage indices for columns that align across
        # DP-rank rows.
        peak = m.get("peak_memory_by_rank") or {}
        for stage, rank in enumerate(sorted(peak, key=lambda r: int(r))):
            row[f"peak_memory_pp{stage}_gb"] = float(peak[rank]) / (1024 ** 3)
        rows.append(row)
    return rows


def _write_results_csv(csv_path: Path, rows: list[dict]) -> None:
    """Write ``rows`` (one per DP rank) to a fresh per-run results CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def build_schedule_file(args: argparse.Namespace, run_dir: Path) -> tuple[Path, str | None]:
    base = _load_schedule(args.base_schedule)
    if args.schedule == "custom":
        # Pass the base schedule through untouched: trust its existing split and
        # order directives. Only used for visualization below.
        schedule = list(base)
        order_directives = [
            d for d in schedule
            if isinstance(d, dict) and d.get("op") == "order"
        ]
    else:
        if args.keep_existing_order:
            schedule = list(base)
        else:
            schedule = [
                directive for directive in base
                if not (isinstance(directive, dict) and directive.get("op") == "order")
            ]
        _set_num_microbatches(schedule, args.mbs)

        if args.schedule == "1f1b":
            order_directives = build_1f1b_schedule(args.ranks, args.mbs)
        elif args.schedule == "zerobubble":
            order_directives = build_zerobubble_schedule(args.ranks, args.mbs)
        elif args.schedule == "dualpipev":
            order_directives = build_dualpipev_schedule(args.ranks, args.mbs)
        else:
            order_directives = build_interleaved_1f1b_schedule(
                args.ranks,
                args.mbs,
                args.virtual_stages,
            )

        _validate_generated_order_placement(
            schedule,
            order_directives,
            schedule_name=args.schedule,
            ranks=args.ranks,
        )
        schedule.extend(order_directives)
    schedule_stem = _schedule_attr_stem(
        args.schedule,
        args.ranks,
        args.mbs,
        args.virtual_stages,
        base_schedule=args.base_schedule,
    )
    run_schedule = run_dir / f"{schedule_stem}.json"
    with run_schedule.open("w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)
        f.write("\n")

    viz_path = None
    if args.viz:
        viz_output = run_dir / f"{schedule_stem}.png"
        viz_path = visualize_order_directives(order_directives, viz_output)
    return run_schedule, viz_path


def _schedule_attr_stem(
    schedule_name: str,
    ranks: int | None,
    mbs: int | None,
    virtual_stages: int | None,
    base_schedule: Path | None = None,
) -> str:
    if schedule_name == "custom":
        # ranks/mbs/virtual_stages aren't required for custom; derive a stable
        # name from the base schedule filename instead.
        if base_schedule is not None:
            return f"custom_{Path(base_schedule).stem}"
        return "custom"
    suffix = f"{schedule_name}_pp{ranks}_mbs{mbs}"
    if schedule_name == "interleaved_1f1b":
        suffix += f"_v{virtual_stages}"
    return suffix


def _virtual_stages_for_schedule(schedule_name: str) -> int:
    if schedule_name in ("interleaved_1f1b", "dualpipev"):
        return 2
    return 1


def _load_schedule(path: Path) -> list[dict]:
    return load_schedule_directives(str(path))


def _set_num_microbatches(schedule: list[dict], n_mbs: int) -> None:
    """Ensure the schedule has exactly one split directive with num_microbatches=n_mbs.

    If a split directive already exists, its num_microbatches is overwritten.
    Otherwise a default MB-split directive is appended.
    """
    found = False
    for directive in schedule:
        if directive.get("op") == "split":
            directive["num_microbatches"] = n_mbs
            found = True
    if not found:
        schedule.append({
            "op": "split",
            "filter": {},
            "dim_name": "MB",
            "num_microbatches": n_mbs,
        })


def _validate_generated_order_placement(
    base_schedule: list[dict],
    order_directives: list[dict],
    *,
    schedule_name: str,
    ranks: int | None,
) -> None:
    """Generated order rows are per physical rank and must be device-local."""
    stage_devices = _stage_devices_from_place_directives(base_schedule)
    if not stage_devices:
        return

    errors: list[str] = []
    for row_idx, directive in enumerate(order_directives):
        row_devices: set[tuple[int, ...]] = set()
        for slot_idx, slot in enumerate(_order_filter_slots(directive.get("filters", []))):
            slot_devices: set[tuple[int, ...]] = set()
            slot_pps: list[int] = []
            for flt in slot:
                spec = _filter_to_dict(flt)
                pp = spec.get("PP")
                if pp is None or pp == "*":
                    continue
                pp = int(pp)
                slot_pps.append(pp)
                dev = stage_devices.get(pp)
                if dev is None:
                    errors.append(
                        f"row {row_idx} slot {slot_idx} references PP{pp}, "
                        "but no matching place directive exists"
                    )
                    continue
                slot_devices.add(dev)
            if len(slot_devices) > 1:
                errors.append(
                    f"row {row_idx} slot {slot_idx} groups stages {slot_pps} "
                    f"across devices {sorted(slot_devices)}"
                )
            row_devices.update(slot_devices)
        if len(row_devices) > 1:
            errors.append(
                f"row {row_idx} orders stages across devices {sorted(row_devices)}"
            )

    if not errors:
        return

    hint = ""
    if schedule_name == "dualpipev" and ranks is not None:
        expected = [
            (rank, 2 * ranks - 1 - rank)
            for rank in range(ranks)
        ]
        hint = (
            " DualPipeV uses V-layout virtual stages; each pair "
            f"{expected} must be placed on the same physical device set."
        )
    raise ValueError(
        "generated order directives are not compatible with the base placement: "
        + "; ".join(errors[:4])
        + hint
    )


def _stage_devices_from_place_directives(schedule: list[dict]) -> dict[int, tuple[int, ...]]:
    stage_devices: dict[int, tuple[int, ...]] = {}
    for directive in schedule:
        if directive.get("op") != "place":
            continue
        devices = directive.get("devices", directive.get("device"))
        if not isinstance(devices, list):
            continue
        spec = _filter_to_dict(directive.get("filter", []))
        pp = spec.get("PP")
        if pp is None or pp == "*":
            continue
        stage_devices[int(pp)] = tuple(sorted(int(d) for d in devices))
    return stage_devices


def _order_filter_slots(filters: object) -> list[list[object]]:
    if not isinstance(filters, list):
        raise ValueError(f"order directive requires filters list, got {type(filters)}")
    slots: list[list[object]] = []
    for item in filters:
        if isinstance(item, list) and item:
            if not all(isinstance(flt, dict) for flt in item):
                raise ValueError(f"invalid order filter group: {item}")
            slots.append(list(item))
        else:
            raise ValueError(f"invalid order filter group: {item}")
    return slots


def _filter_to_dict(flt: object) -> dict:
    if isinstance(flt, dict):
        return dict(flt)
    raise ValueError(f"invalid filter: {flt}")


def _replace_schedule_arg(test_args: list[str], schedule_path: Path) -> list[str]:
    return _replace_arg(test_args, "--schedule-directives-file", str(schedule_path))


def _replace_arg(test_args: list[str], flag: str, value: str) -> list[str]:
    """Drop any existing occurrence of ``flag`` (and its value) from forwarded
    test args, then append ``flag value`` so the harness setting wins."""
    out: list[str] = []
    skip_next = False
    for arg in test_args:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        if arg.startswith(f"{flag}="):
            continue
        out.append(arg)
    out.extend([flag, value])
    return out


def _load_test_file(test_file: str):
    path = Path(test_file)
    if path.suffix != ".py":
        raise ValueError(f"--test-file must be a Python file path, got: {test_file}")
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"test file does not exist: {path}")

    sys.path.insert(0, str(path.parent))
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Python file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module_name, module


if __name__ == "__main__":
    main()
