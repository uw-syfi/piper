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
    _validate_device_set_consistency(directives)
    return directives


def load_schedule_info(path: str) -> dict[str, Any]:
    return derive_schedule_info(load_schedule_directives(path), path)


def derive_schedule_info(directives: list[dict], schedule_path: str) -> dict[str, Any]:
    pp_indices: set[int] = set()
    actor_group_devices: list[tuple[int, ...]] = []
    seen_actor_groups: set[tuple[int, ...]] = set()
    num_microbatches = None

    for directive in directives:
        if not isinstance(directive, dict):
            continue
        op = directive.get("op")
        if op == "place":
            devices = directive.get("devices", directive.get("device"))
            if not isinstance(devices, list) or not devices:
                raise ValueError(f"place directive must include non-empty devices: {directive}")
            device_key = tuple(sorted(int(d) for d in devices))
            if len(device_key) != len(set(device_key)):
                raise ValueError(f"place directive devices must not contain duplicates: {directive}")
            if any(device < 0 for device in device_key):
                raise ValueError(f"place directive devices must be non-negative: {directive}")
            if device_key not in seen_actor_groups:
                actor_group_devices.append(device_key)
                seen_actor_groups.add(device_key)
            pp_idx = _filter_value(directive.get("filter"), "PP")
            if isinstance(pp_idx, int) and not isinstance(pp_idx, bool):
                if pp_idx < 0:
                    raise ValueError(f"PP filter indices must be non-negative: {directive}")
                pp_indices.add(pp_idx)
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

    if not actor_group_devices:
        raise ValueError("schedule JSON must include at least one place directive")

    global_ranks = sorted({rank for group in actor_group_devices for rank in group})
    expected_ranks = list(range(len(global_ranks)))
    if global_ranks != expected_ranks:
        raise ValueError(
            "place directives must cover contiguous global actor ranks starting at 0; "
            f"got {global_ranks}"
        )

    actor_groups = [
        {
            "id": group_id,
            "devices": list(devices),
            "size": len(devices),
        }
        for group_id, devices in enumerate(actor_group_devices)
    ]
    global_rank_to_actor_group: dict[int, dict[str, int]] = {}
    for group in actor_groups:
        for local_rank, global_rank in enumerate(group["devices"]):
            global_rank_to_actor_group[int(global_rank)] = {
                "actor_group_id": int(group["id"]),
                "local_rank": local_rank,
            }

    group_sizes = [len(devices) for devices in actor_group_devices]
    # ``num_stages`` describes the model's PP annotation domain, not how many
    # distinct actor groups own the resulting DAG fragments.  AFD schedules
    # may only mention endpoint PP tags (for example PP0 and PP3).
    num_stages = max(pp_indices) + 1 if pp_indices else len(actor_groups)

    return {
        "name": os.path.splitext(os.path.basename(schedule_path))[0],
        "path": schedule_path,
        "num_stages": num_stages,
        "actor_groups": actor_groups,
        "actor_group_count": len(actor_groups),
        "actor_group_sizes": group_sizes,
        "global_ranks": global_ranks,
        "global_rank_to_actor_group": global_rank_to_actor_group,
        "world_size": len(global_ranks),
        # PP remains the model-stage/DAG count used by current compiler APIs.
        # Replica topology is represented exclusively by the explicit device
        # membership of each ActorGroup, including for equal-width groups.
        "pp_degree": len(actor_groups),
        "num_microbatches": num_microbatches if num_microbatches is not None else 1,
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


def _validate_device_set_consistency(directives: list[dict]) -> None:
    """Require schedule device sets to form disjoint SPMD groups.

    Place, replicate, and shard directives may reuse the same device set, but
    two distinct sets may not overlap. This prevents a schedule from treating
    a subset (or any other partial overlap) as a different SPMD group.
    """
    device_sets: list[tuple[int, str, frozenset[int]]] = []
    place_sets: set[frozenset[int]] = set()
    communication_sets: list[tuple[int, str, frozenset[int]]] = []
    for idx, directive in enumerate(directives):
        op = directive.get("op")
        if op not in {"place", "replicate", "shard"}:
            continue
        devices = directive.get("devices", directive.get("device"))
        if not isinstance(devices, list) or not devices:
            raise ValueError(
                f"{op} directive[{idx}] requires a non-empty devices list"
            )
        if any(
            isinstance(device, bool) or not isinstance(device, int)
            for device in devices
        ):
            raise ValueError(
                f"{op} directive[{idx}] devices must be integer actor ranks: "
                f"{devices}"
            )
        if len(devices) != len(set(devices)):
            raise ValueError(
                f"{op} directive[{idx}] devices must not contain duplicates: "
                f"{devices}"
            )
        if any(device < 0 for device in devices):
            raise ValueError(
                f"{op} directive[{idx}] devices must be non-negative: {devices}"
            )
        device_set = frozenset(devices)
        for previous_idx, previous_op, previous_set in device_sets:
            if device_set != previous_set and device_set & previous_set:
                raise ValueError(
                    "schedule device sets must be identical or disjoint; "
                    f"directive[{previous_idx}] {previous_op} uses "
                    f"devices={sorted(previous_set)}, while directive[{idx}] "
                    f"{op} uses overlapping devices={sorted(device_set)}"
                )
        device_sets.append((idx, str(op), device_set))
        if op == "place":
            place_sets.add(device_set)
        else:
            communication_sets.append((idx, str(op), device_set))

    for idx, op, device_set in communication_sets:
        if device_set not in place_sets:
            raise ValueError(
                f"{op} directive[{idx}] devices={sorted(device_set)} must "
                "exactly match a place ActorGroup"
            )


def _filter_value(filter_spec, key: str):
    if isinstance(filter_spec, dict):
        return filter_spec.get(key)
    return None
