#!/usr/bin/env python3
"""Consolidate and rank the Qwen3.5-397B P2P communication benchmarks.

The P2P x all-gather campaign was intentionally split across several jobs to
bound allocation time and memory use.  Two settings were subsequently rerun
with more samples:

* 256 MiB full all-gather payload, 8 sequential kernels
* the 1,940 MiB embedding all-gather payload, 1 kernel

This script treats the corresponding validation CSVs as authoritative for
those exact settings, computes ratio-of-median-times alongside the benchmark's
sample-aligned median slowdown, and emits report-ready consolidated/ranked
CSVs.  It is deliberately a local post-processing tool: it does not import
Ray or PyTorch and does not access GPUs or the network.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_AG_DIR = Path("out/interference_benchmarks/qwen397b_p2p_allgather")
DEFAULT_BIDIR_CSV = Path(
    "out/interference_benchmarks/qwen397b_bidirectional_p2p/summary.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "out/interference_benchmarks/qwen397b_p2p_collective_interference"
)

CAMPAIGN_FILENAMES = (
    "count1_moderate.csv",
    "count1_large.csv",
    "count2.csv",
    "count4.csv",
    "count8.csv",
    "count16.csv",
)
VALIDATION_FILENAMES = (
    "validation_count8_256.csv",
    "validation_embedding.csv",
)

# These fields identify an exact P2P x all-gather experiment.  In particular,
# "case" is intentionally omitted because it is a presentation label.
AG_EXPERIMENT_KEY = (
    "queue_order",
    "p2p_direction",
    "p2p_payload",
    "p2p_mib",
    "allgather_payload",
    "allgather_full_mib_per_kernel",
    "allgather_shard_mib_per_rank",
    "allgather_kernel_count",
)

AG_DERIVED_FIELDS = (
    "p2p_slowdown_ratio_of_medians",
    "allgather_slowdown_ratio_of_medians",
    "pair_overhead_ratio_of_medians",
)
BIDIR_DERIVED_FIELDS = (
    "ab_slowdown_ratio_of_medians",
    "ba_slowdown_ratio_of_medians",
    "pair_overhead_ratio_of_medians",
)

# The campaign deliberately holds total all-gather sequence volume constant
# while varying how many collectives carry it.  Labels are included to ensure
# that a similarly sized but semantically different row cannot enter a sweep
# accidentally.
FIXED_TOTAL_VOLUME_SWEEPS = {
    256.0: (
        ("custom_256mib", 256.0, 1),
        ("attention_gated_q", 128.0, 2),
        ("attention_o", 64.0, 4),
        ("custom_32mib", 32.0, 8),
        ("shared_expert_gate_up", 16.0, 16),
    ),
    1024.0: (
        ("custom_1024mib", 1024.0, 1),
        ("custom_512mib", 512.0, 2),
        ("custom_256mib", 256.0, 4),
        ("attention_gated_q", 128.0, 8),
        ("attention_o", 64.0, 16),
    ),
}
FIXED_TOTAL_QUEUE_ORDERS = ("p2p_first", "allgather_sequence_first")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allgather-dir",
        type=Path,
        default=DEFAULT_AG_DIR,
        help="Directory containing the completed P2P x all-gather CSVs.",
    )
    parser.add_argument(
        "--bidirectional-csv",
        type=Path,
        default=DEFAULT_BIDIR_CSV,
        help="Completed bidirectional P2P summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination for consolidated and ranked CSVs.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"required input CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"input CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def finite_float(row: Mapping[str, str], field: str) -> float:
    raw = row.get(field, "")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric field {field}={raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric field {field}={raw!r}")
    return value


def ratio(row: Mapping[str, str], numerator: str, denominator: str) -> float:
    denominator_value = finite_float(row, denominator)
    if denominator_value == 0.0:
        raise ZeroDivisionError(f"cannot divide by zero-valued {denominator}")
    return finite_float(row, numerator) / denominator_value


def format_ratio(value: float) -> str:
    return f"{value:.12g}"


def experiment_key(row: Mapping[str, str]) -> tuple[str, ...]:
    missing = [field for field in AG_EXPERIMENT_KEY if field not in row]
    if missing:
        raise ValueError(f"P2P x all-gather row is missing key fields: {missing}")
    return tuple(row[field] for field in AG_EXPERIMENT_KEY)


def ensure_compatible_headers(
    reference: Sequence[str], candidate: Sequence[str], path: Path
) -> None:
    if list(reference) != list(candidate):
        raise ValueError(
            f"CSV schema mismatch in {path}: expected {list(reference)}, "
            f"found {list(candidate)}"
        )


def consolidate_allgather(
    allgather_dir: Path,
) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
    input_names = CAMPAIGN_FILENAMES + VALIDATION_FILENAMES
    fieldnames: list[str] | None = None
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    source_for_key: dict[tuple[str, ...], str] = {}
    replacement_counts = {name: 0 for name in VALIDATION_FILENAMES}

    for filename in input_names:
        path = allgather_dir / filename
        current_fields, rows = read_csv(path)
        if fieldnames is None:
            fieldnames = current_fields
        else:
            ensure_compatible_headers(fieldnames, current_fields, path)

        is_validation = filename in VALIDATION_FILENAMES
        for source_row in rows:
            key = experiment_key(source_row)
            if key in selected and not is_validation:
                previous_source = source_for_key[key]
                raise ValueError(
                    "duplicate non-validation experiment in "
                    f"{previous_source} and {filename}: {key}"
                )
            if key in selected and is_validation:
                replacement_counts[filename] += 1
            row = dict(source_row)
            row["source_csv"] = filename
            row["p2p_slowdown_ratio_of_medians"] = format_ratio(
                ratio(row, "p2p_concurrent_median_ms", "p2p_solo_median_ms")
            )
            row["allgather_slowdown_ratio_of_medians"] = format_ratio(
                ratio(
                    row,
                    "allgather_concurrent_median_ms",
                    "allgather_solo_median_ms",
                )
            )
            row["pair_overhead_ratio_of_medians"] = format_ratio(
                ratio(row, "normal_pair_median_ms", "ideal_pair_median_ms")
            )
            selected[key] = row
            source_for_key[key] = filename

    if fieldnames is None:
        raise ValueError("no P2P x all-gather inputs were read")
    for filename, count in replacement_counts.items():
        if count == 0:
            raise ValueError(
                f"{filename} did not replace any lower-sample campaign rows"
            )

    rows = list(selected.values())
    rows.sort(
        key=lambda row: (
            int(float(row["allgather_kernel_count"])),
            finite_float(row, "allgather_full_mib_per_kernel"),
            row["allgather_payload"],
            row["p2p_direction"],
            row["queue_order"],
        )
    )
    output_fields = ["source_csv", *fieldnames, *AG_DERIVED_FIELDS]
    return output_fields, rows, replacement_counts


def add_bidirectional_ratios(
    bidirectional_csv: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames, input_rows = read_csv(bidirectional_csv)
    rows: list[dict[str, str]] = []
    for source_row in input_rows:
        row = dict(source_row)
        row["source_csv"] = bidirectional_csv.name
        row["ab_slowdown_ratio_of_medians"] = format_ratio(
            ratio(row, "ab_concurrent_ms_median", "ab_solo_ms_median")
        )
        row["ba_slowdown_ratio_of_medians"] = format_ratio(
            ratio(row, "ba_concurrent_ms_median", "ba_solo_ms_median")
        )
        row["pair_overhead_ratio_of_medians"] = format_ratio(
            ratio(row, "normal_pair_ms_median", "ideal_pair_ms_median")
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            finite_float(row, "payload_mib"),
            row["mode"],
            row["queue_first"],
        )
    )
    return ["source_csv", *fieldnames, *BIDIR_DERIVED_FIELDS], rows


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ranked_copy(
    rows: Sequence[dict[str, str]], slowdown_field: str
) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            finite_float(row, slowdown_field),
            finite_float(row, "sample_count"),
        ),
        reverse=True,
    )


def flatten_bidirectional_directions(
    rows: Sequence[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = [
        "rank",
        "source_csv",
        "model",
        "payload_name",
        "payload_mib",
        "mode",
        "queue_first",
        "direction",
        "samples",
        "solo_median_ms",
        "concurrent_median_ms",
        "slowdown_median_aligned",
        "slowdown_ratio_of_medians",
    ]
    flattened: list[dict[str, str]] = []
    for row in rows:
        for prefix, direction in (("ab", "A_to_B"), ("ba", "B_to_A")):
            flattened.append(
                {
                    "source_csv": row["source_csv"],
                    "model": row["model"],
                    "payload_name": row["payload_name"],
                    "payload_mib": row["payload_mib"],
                    "mode": row["mode"],
                    "queue_first": row["queue_first"],
                    "direction": direction,
                    "samples": row["samples"],
                    "solo_median_ms": row[f"{prefix}_solo_ms_median"],
                    "concurrent_median_ms": row[f"{prefix}_concurrent_ms_median"],
                    "slowdown_median_aligned": row[
                        f"{prefix}_slowdown_median"
                    ],
                    "slowdown_ratio_of_medians": row[
                        f"{prefix}_slowdown_ratio_of_medians"
                    ],
                }
            )
    flattened.sort(
        key=lambda row: finite_float(row, "slowdown_median_aligned"),
        reverse=True,
    )
    for index, row in enumerate(flattened, start=1):
        row["rank"] = str(index)
    return fieldnames, flattened


def summarize_allgather_setting_ranges(
    rows: Sequence[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    group_fields = (
        "allgather_payload",
        "allgather_full_mib_per_kernel",
        "allgather_shard_mib_per_rank",
        "allgather_kernel_count",
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    output_fields = [
        *group_fields,
        "allgather_total_sequence_mib",
        "source_csv",
        "sample_count",
        "setting_row_count",
        "p2p_slowdown_ratio_of_medians_min",
        "p2p_slowdown_ratio_of_medians_max",
        "p2p_max_direction",
        "p2p_max_queue_order",
        "allgather_slowdown_ratio_of_medians_min",
        "allgather_slowdown_ratio_of_medians_max",
        "allgather_max_direction",
        "allgather_max_queue_order",
    ]
    output_rows: list[dict[str, str]] = []
    for key, setting_rows in groups.items():
        sources = sorted({row["source_csv"] for row in setting_rows})
        sample_counts = sorted(
            {int(float(row["sample_count"])) for row in setting_rows}
        )
        if len(sources) != 1 or len(sample_counts) != 1:
            raise ValueError(
                f"setting {key} has inconsistent sources/samples: "
                f"{sources}, {sample_counts}"
            )
        p2p_min = min(
            setting_rows,
            key=lambda row: finite_float(
                row, "p2p_slowdown_ratio_of_medians"
            ),
        )
        p2p_max = max(
            setting_rows,
            key=lambda row: finite_float(
                row, "p2p_slowdown_ratio_of_medians"
            ),
        )
        ag_min = min(
            setting_rows,
            key=lambda row: finite_float(
                row, "allgather_slowdown_ratio_of_medians"
            ),
        )
        ag_max = max(
            setting_rows,
            key=lambda row: finite_float(
                row, "allgather_slowdown_ratio_of_medians"
            ),
        )
        full_mib = finite_float(setting_rows[0], "allgather_full_mib_per_kernel")
        kernel_count = int(
            finite_float(setting_rows[0], "allgather_kernel_count")
        )
        output_rows.append(
            {
                **dict(zip(group_fields, key)),
                "allgather_total_sequence_mib": format_ratio(
                    full_mib * kernel_count
                ),
                "source_csv": sources[0],
                "sample_count": str(sample_counts[0]),
                "setting_row_count": str(len(setting_rows)),
                "p2p_slowdown_ratio_of_medians_min": p2p_min[
                    "p2p_slowdown_ratio_of_medians"
                ],
                "p2p_slowdown_ratio_of_medians_max": p2p_max[
                    "p2p_slowdown_ratio_of_medians"
                ],
                "p2p_max_direction": p2p_max["p2p_direction"],
                "p2p_max_queue_order": p2p_max["queue_order"],
                "allgather_slowdown_ratio_of_medians_min": ag_min[
                    "allgather_slowdown_ratio_of_medians"
                ],
                "allgather_slowdown_ratio_of_medians_max": ag_max[
                    "allgather_slowdown_ratio_of_medians"
                ],
                "allgather_max_direction": ag_max["p2p_direction"],
                "allgather_max_queue_order": ag_max["queue_order"],
            }
        )

    output_rows.sort(
        key=lambda row: (
            finite_float(row, "allgather_total_sequence_mib"),
            int(float(row["allgather_kernel_count"])),
            row["allgather_payload"],
        )
    )
    return output_fields, output_rows


def summarize_fixed_total_volume(
    rows: Sequence[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    index: dict[tuple[str, float, int, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["allgather_payload"],
            finite_float(row, "allgather_full_mib_per_kernel"),
            int(finite_float(row, "allgather_kernel_count")),
            row["queue_order"],
            row["p2p_direction"],
        )
        if key in index:
            raise ValueError(f"duplicate consolidated sweep row: {key}")
        index[key] = row

    output_fields = [
        "allgather_total_sequence_mib",
        "allgather_payload",
        "allgather_full_mib_per_kernel",
        "allgather_kernel_count",
        "queue_order",
        "source_csv",
        "sample_count",
        "p2p_slowdown_ratio_of_medians_forward",
        "p2p_slowdown_ratio_of_medians_backward",
        "p2p_slowdown_ratio_of_medians_across_directions_median",
        "allgather_slowdown_ratio_of_medians_forward",
        "allgather_slowdown_ratio_of_medians_backward",
        "allgather_slowdown_ratio_of_medians_across_directions_median",
    ]
    output_rows: list[dict[str, str]] = []
    for total_mib, settings in FIXED_TOTAL_VOLUME_SWEEPS.items():
        for payload, full_mib, kernel_count in settings:
            observed_total = full_mib * kernel_count
            if not math.isclose(observed_total, total_mib):
                raise AssertionError(
                    f"invalid fixed-volume setting: {payload} "
                    f"{full_mib} MiB x {kernel_count} != {total_mib} MiB"
                )
            for queue_order in FIXED_TOTAL_QUEUE_ORDERS:
                direction_rows = {
                    direction: index.get(
                        (
                            payload,
                            full_mib,
                            kernel_count,
                            queue_order,
                            direction,
                        )
                    )
                    for direction in ("forward", "backward")
                }
                missing = [
                    direction
                    for direction, row in direction_rows.items()
                    if row is None
                ]
                if missing:
                    raise ValueError(
                        f"fixed-volume setting is missing {missing}: "
                        f"{payload}, {full_mib} MiB x{kernel_count}, "
                        f"{queue_order}"
                    )
                forward = direction_rows["forward"]
                backward = direction_rows["backward"]
                assert forward is not None and backward is not None
                sources = sorted(
                    {forward["source_csv"], backward["source_csv"]}
                )
                samples = sorted(
                    {
                        int(float(forward["sample_count"])),
                        int(float(backward["sample_count"])),
                    }
                )
                if len(sources) != 1 or len(samples) != 1:
                    raise ValueError(
                        "fixed-volume direction rows have inconsistent "
                        f"sources/samples: {sources}, {samples}"
                    )
                p2p_values = [
                    finite_float(
                        forward, "p2p_slowdown_ratio_of_medians"
                    ),
                    finite_float(
                        backward, "p2p_slowdown_ratio_of_medians"
                    ),
                ]
                ag_values = [
                    finite_float(
                        forward, "allgather_slowdown_ratio_of_medians"
                    ),
                    finite_float(
                        backward, "allgather_slowdown_ratio_of_medians"
                    ),
                ]
                output_rows.append(
                    {
                        "allgather_total_sequence_mib": format_ratio(
                            total_mib
                        ),
                        "allgather_payload": payload,
                        "allgather_full_mib_per_kernel": format_ratio(
                            full_mib
                        ),
                        "allgather_kernel_count": str(kernel_count),
                        "queue_order": queue_order,
                        "source_csv": sources[0],
                        "sample_count": str(samples[0]),
                        "p2p_slowdown_ratio_of_medians_forward": format_ratio(
                            p2p_values[0]
                        ),
                        "p2p_slowdown_ratio_of_medians_backward": format_ratio(
                            p2p_values[1]
                        ),
                        "p2p_slowdown_ratio_of_medians_across_directions_median": format_ratio(
                            statistics.median(p2p_values)
                        ),
                        "allgather_slowdown_ratio_of_medians_forward": format_ratio(
                            ag_values[0]
                        ),
                        "allgather_slowdown_ratio_of_medians_backward": format_ratio(
                            ag_values[1]
                        ),
                        "allgather_slowdown_ratio_of_medians_across_directions_median": format_ratio(
                            statistics.median(ag_values)
                        ),
                    }
                )
    return output_fields, output_rows


def add_ranks(rows: Sequence[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    ranked_rows: list[dict[str, str]] = []
    for index, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        row["rank"] = str(index)
        ranked_rows.append(row)
    return ["rank"], ranked_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ag_fields, ag_rows, replacements = consolidate_allgather(args.allgather_dir)
    bidir_fields, bidir_rows = add_bidirectional_ratios(args.bidirectional_csv)

    write_csv(
        args.output_dir / "p2p_allgather_consolidated.csv",
        ag_fields,
        ag_rows,
    )

    p2p_ranked = ranked_copy(ag_rows, "p2p_slowdown_median")
    rank_prefix, p2p_ranked = add_ranks(p2p_ranked)
    write_csv(
        args.output_dir / "p2p_slowdowns_ranked.csv",
        [*rank_prefix, *ag_fields],
        p2p_ranked,
    )

    ag_ranked = ranked_copy(ag_rows, "allgather_slowdown_median")
    rank_prefix, ag_ranked = add_ranks(ag_ranked)
    write_csv(
        args.output_dir / "allgather_slowdowns_ranked.csv",
        [*rank_prefix, *ag_fields],
        ag_ranked,
    )

    write_csv(
        args.output_dir / "bidirectional_p2p_consolidated.csv",
        bidir_fields,
        bidir_rows,
    )
    bidir_rank_fields, bidir_ranked = flatten_bidirectional_directions(bidir_rows)
    write_csv(
        args.output_dir / "bidirectional_p2p_slowdowns_ranked.csv",
        bidir_rank_fields,
        bidir_ranked,
    )

    setting_range_fields, setting_range_rows = summarize_allgather_setting_ranges(
        ag_rows
    )
    write_csv(
        args.output_dir / "p2p_allgather_setting_ranges.csv",
        setting_range_fields,
        setting_range_rows,
    )

    fixed_volume_fields, fixed_volume_rows = summarize_fixed_total_volume(ag_rows)
    write_csv(
        args.output_dir / "fixed_total_volume_summary.csv",
        fixed_volume_fields,
        fixed_volume_rows,
    )

    print(f"Selected {len(ag_rows)} P2P x all-gather rows")
    for filename in VALIDATION_FILENAMES:
        print(f"  {filename}: replaced {replacements[filename]} rows")
    print(f"Selected {len(bidir_rows)} bidirectional P2P rows")
    print(f"Summarized {len(setting_range_rows)} all-gather settings")
    print(f"Summarized {len(fixed_volume_rows)} fixed-volume/order settings")
    print(f"Wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
