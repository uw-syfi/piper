import json
import os
from typing import Any


def load_schedule_directives(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        directives = json.load(f)
    if not isinstance(directives, list):
        raise ValueError(f"schedule directives file must contain a list, got: {type(directives)}")

    for i, directive in enumerate(directives):
        if not isinstance(directive, dict):
            raise ValueError(f"schedule directive[{i}] must be an object, got: {type(directive)}")
        _validate_directive_shape(directive, i)
        if directive.get("op") == "split" and directive.get("num_microbatches") == "__MBS__":
            raise ValueError(
                "split.num_microbatches must be encoded in the schedule JSON; "
                "'__MBS__' is no longer supported"
            )
    return directives


def load_schedule_info(path: str) -> dict[str, Any]:
    return derive_schedule_info(load_schedule_directives(path), path)


def derive_schedule_info(directives: list[dict], schedule_path: str) -> dict[str, Any]:
    pp_to_devices: dict[int, list[int]] = {}
    num_microbatches = None

    for directive in directives:
        if not isinstance(directive, dict):
            continue
        op = directive.get("op")
        if op == "place":
            pp_idx = _filter_value(directive.get("filter"), "PP")
            devices = directive.get("devices", directive.get("device"))
            if pp_idx is None or not isinstance(devices, list) or not devices:
                raise ValueError(f"place directive must include PP filter and non-empty devices: {directive}")
            pp_to_devices[int(pp_idx)] = [int(d) for d in devices]
        elif op == "split":
            n = int(directive.get("num_microbatches", 0))
            if n <= 0:
                raise ValueError(f"split directive requires num_microbatches > 0: {directive}")
            if num_microbatches is not None and num_microbatches != n:
                raise ValueError(
                    f"multiple split directives disagree on num_microbatches: "
                    f"{num_microbatches} vs {n}"
                )
            num_microbatches = n

    if not pp_to_devices:
        raise ValueError("schedule JSON must include place directives with PP filters")
    pp_indices = sorted(pp_to_devices)
    expected_pp = list(range(len(pp_indices)))
    if pp_indices != expected_pp:
        raise ValueError(f"PP indices must be contiguous from 0, got {pp_indices}")
    device_counts = {len(devices) for devices in pp_to_devices.values()}
    if len(device_counts) != 1:
        raise ValueError(f"all PP place directives must use the same device count, got {pp_to_devices}")
    device_keys = sorted({tuple(devices) for devices in pp_to_devices.values()})
    if num_microbatches is None:
        raise ValueError("schedule JSON must include a split directive with num_microbatches")

    return {
        "name": os.path.splitext(os.path.basename(schedule_path))[0],
        "path": schedule_path,
        "num_stages": len(pp_indices),
        "pp_degree": len(device_keys),
        "dp_degree": next(iter(device_counts)),
        "num_microbatches": num_microbatches,
    }


def _validate_directive_shape(directive: dict, idx: int) -> None:
    op = directive.get("op")
    if op in {"place", "replicate", "shard", "split"}:
        if not isinstance(directive.get("filter"), dict):
            raise ValueError(f"{op} directive[{idx}] requires object field 'filter': {directive}")
        if "filters" in directive:
            raise ValueError(f"{op} directive[{idx}] does not accept field 'filters': {directive}")
    elif op == "order":
        filters = directive.get("filters")
        if not isinstance(filters, list) or not filters:
            raise ValueError(f"order directive[{idx}] requires non-empty list field 'filters': {directive}")
        for group_idx, group in enumerate(filters):
            if not isinstance(group, list) or not group:
                raise ValueError(
                    f"order directive[{idx}] group[{group_idx}] must be a non-empty list: {directive}"
                )
            for filter_idx, flt in enumerate(group):
                if not isinstance(flt, dict):
                    raise ValueError(
                        f"order directive[{idx}] group[{group_idx}][{filter_idx}] "
                        f"must be a filter object: {directive}"
                    )
    else:
        raise ValueError(f"schedule directive[{idx}] has unsupported op: {directive}")


def _filter_value(filter_spec, key: str):
    if isinstance(filter_spec, dict):
        return filter_spec.get(key)
    return None
