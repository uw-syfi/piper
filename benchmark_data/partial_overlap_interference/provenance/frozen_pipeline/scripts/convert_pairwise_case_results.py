#!/usr/bin/env python3
"""Convert raw distributed-case JSON into analysis source tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from src.pairwise_overlap_analysis_filter import (
    analysis_inclusion_counts,
    validator_equivalent_sample_disposition,
)
from src.pairwise_overlap_correctness import result_correctness_status
from src.pairwise_overlap_identity import (
    result_source_run_id,
    run_id_for_row,
    sample_identity,
)


COMM = {"a2a", "ag", "rs", "p2p_send", "p2p_recv"}
_DTYPE_WIDTH = {
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "half": 2,
    "float": 4,
    "float32": 4,
    "long": 8,
    "int64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dtype_width(participant: dict[str, Any], metadata: dict[str, Any]) -> int | None:
    signature = participant.get("signature", {})
    raw = metadata.get("dtype", signature.get("dtype"))
    if raw in (None, ""):
        return None
    normalized = str(raw).rsplit(".", 1)[-1].lower()
    return _DTYPE_WIDTH.get(normalized)


def payload_bytes(participant: dict[str, Any]) -> int | str:
    """Return the legacy anchor input payload from the participant signature."""

    signature = participant["signature"]
    numel = int(signature.get("in_msg_nelems", -1))
    width = _dtype_width(participant, {})
    if numel < 0 or width is None:
        return ""
    return numel * width


def _operation_metadata_index(
    result: dict[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for record in result.get("operation_metadata", []):
        key = (int(record["operation_index"]), int(record["global_rank"]))
        metadata = record.get("metadata", record)
        if key in indexed and indexed[key] != metadata:
            raise ValueError(f"conflicting operation metadata for {key}")
        indexed[key] = metadata
    return indexed


def operation_payload_bytes(
    participant: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> tuple[int | str, int | str]:
    """Return rank-local input/output bytes, preferring runtime metadata."""

    metadata = metadata or {}
    width = _dtype_width(participant, metadata)
    if width is None:
        return "", ""
    signature = participant.get("signature", {})
    input_numel = metadata.get("input_numel", metadata.get("numel"))
    output_numel = metadata.get("output_numel", metadata.get("numel"))
    if input_numel is None:
        input_numel = signature.get("in_msg_nelems")
    if output_numel is None:
        output_numel = signature.get("out_msg_nelems")
    input_bytes: int | str = (
        int(input_numel) * width if input_numel is not None else ""
    )
    output_bytes: int | str = (
        int(output_numel) * width if output_numel is not None else ""
    )
    return input_bytes, output_bytes


def convert(document: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sample_rows = []
    participant_rows = []
    communication_rows = []
    for result in document["results"]:
        assignment = result["assignment"]
        participants = assignment["participants"]
        case_key = assignment["case_key"]
        source_run_id = result_source_run_id(document, result)
        run_id = run_id_for_row(document, result)
        correct = result_correctness_status(result)
        metadata_index = _operation_metadata_index(result)
        expected_operation_indices = range(len(participants))
        for sample in result["samples"]:
            seq_len = int(sample["seq_len"])
            sample_id = sample_identity(source_run_id, case_key, sample)
            analysis_disposition = validator_equivalent_sample_disposition(
                sample,
                result_correct=correct,
                expected_operation_indices=expected_operation_indices,
            )
            analysis_included = analysis_disposition.startswith("included_")
            adaptive_accounting = {
                "adaptive_controller_disposition": sample.get(
                    "adaptive_controller_disposition", ""
                ),
                "adaptive_controller_ingested": sample.get(
                    "adaptive_controller_ingested", ""
                ),
                "adaptive_controller_valid_sample": sample.get(
                    "adaptive_controller_valid_sample", ""
                ),
                "adaptive_controller_invalid_reasons": json.dumps(
                    sample.get("adaptive_controller_invalid_reasons", []),
                    sort_keys=True,
                ),
                "analysis_included": analysis_included,
                "analysis_disposition": analysis_disposition,
            }
            serialized_sample = {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (list, dict)) else value
                for key, value in sample.items()
            }
            matching = [
                row for row in result["participants"]
                if row["mode"] == sample["dispatch_order"]
                and row["sample_index"] == sample["sample_index"]
            ]
            actual_comm_input_payloads: list[int] = []
            for row in matching:
                index = int(row["operation_index"])
                global_rank = int(row["global_rank"])
                participant = participants[index]
                role = "victim" if index == sample["victim_index"] else "aggressor"
                participant_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_run_id": source_run_id,
                        "run_id": run_id,
                        "case_id": assignment["case_id"],
                        "case_key": case_key,
                        "seq_len": seq_len,
                        "role": role,
                        "operation_index": index,
                        "participant_id": participant["participant_id"],
                        "family": participant["category"],
                        "kernel_alias": participant["kernel_alias"],
                        "global_rank": global_rank,
                        "stream": participant["stream_role"],
                        "start_us": float(row["gpu_start_global_ns"]) / 1000.0,
                        "end_us": float(row["gpu_end_global_ns"]) / 1000.0,
                        "duration_us": float(row["duration_ms"]) * 1000.0,
                        **adaptive_accounting,
                    }
                )
                if participant["category"] in COMM:
                    input_bytes, output_bytes = operation_payload_bytes(
                        participant, metadata_index.get((index, global_rank))
                    )
                    if input_bytes != "":
                        actual_comm_input_payloads.append(int(input_bytes))
                    communication_rows.append(
                        {
                            "sample_id": sample_id,
                            "source_run_id": source_run_id,
                            "run_id": run_id,
                            "case_id": assignment["case_id"],
                            "case_key": case_key,
                            "seq_len": seq_len,
                            "collective": participant["category"],
                            "operation_index": index,
                            "participant_id": participant["participant_id"],
                            "kernel_alias": participant["kernel_alias"],
                            "global_rank": global_rank,
                            "api_start_us": float(row["host_launch_global_ns"]) / 1000.0,
                            "gpu_start_us": float(row["gpu_start_global_ns"]) / 1000.0,
                            "gpu_end_us": float(row["gpu_end_global_ns"]) / 1000.0,
                            "api_completion_us": float(row["host_complete_global_ns"]) / 1000.0,
                            "api_duration_us": (
                                float(row["host_complete_global_ns"])
                                - float(row["host_launch_global_ns"])
                            ) / 1000.0,
                            "launch_to_gpu_start_us": (
                                float(row["gpu_start_global_ns"])
                                - float(row["host_launch_global_ns"])
                            ) / 1000.0,
                            "duration_us": float(row["duration_ms"]) * 1000.0,
                            # Endpoint summaries intentionally consume the
                            # rank-local input payload through this legacy name.
                            "payload_bytes": input_bytes,
                            "input_payload_bytes": input_bytes,
                            "output_payload_bytes": output_bytes,
                            **adaptive_accounting,
                        }
                    )
            sample_rows.append({
                "sample_id": sample_id,
                "source_run_id": source_run_id,
                "run_id": run_id,
                **serialized_sample,
                "payload_bytes": (
                    max(actual_comm_input_payloads)
                    if actual_comm_input_payloads else ""
                ),
                "correct": correct,
                **adaptive_accounting,
            })
    return {
        "samples.csv": sample_rows,
        "participants.csv": participant_rows,
        "communications.csv": communication_rows,
    }


def main() -> None:
    args = parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    tables = convert(document)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.output_dir / name, rows)
    manifest = {
        "source": str(args.input),
        "tables": {name: len(rows) for name, rows in tables.items()},
        "analysis_sample_inclusion": analysis_inclusion_counts(
            tables["samples.csv"]
        ),
        "clock_calibration": document["clock_calibration"],
        "clock_method": (
            "Cristian midpoint offset to one Ray clock actor, sequential 32-probe "
            "best-RTT calibration; GPU timestamps mapped with a synchronized CUDA "
            "anchor event for every stage"
        ),
    }
    (args.output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["tables"], sort_keys=True))


if __name__ == "__main__":
    main()
