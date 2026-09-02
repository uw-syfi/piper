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

"""Build JSON order directives for common pipeline schedules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

Pass = Literal["F", "B", "BI", "BW"]


@dataclass(frozen=True)
class _Op:
    pp: int
    mb: int
    pass_: Pass


# A "slot" is one timestep in a rank's row. A slot holds one op normally; when
# multiple ops are nested in the same order-directive filter group they share a
# slot and visualize as a vertically-stacked cell, indicating that they run
# interleaved within the same scheduling block.
_Slot = list[_Op]


def _filter(pp: int, mb: int, pass_: Pass) -> dict[str, int | str]:
    return {"PP": pp, "MB": mb, "PASS": pass_}


def _order_directive(ops: list[_Op]) -> dict:
    return _order_directive_from_slots([[op] for op in ops])


def _order_directive_from_slots(slots: list[_Slot]) -> dict:
    return {
        "op": "order",
        "filters": [
            [_filter(op.pp, op.mb, op.pass_) for op in slot]
            for slot in slots
        ],
    }


def build_1f1b_schedule(n_ranks: int, n_mbs: int) -> list[dict]:
    """Return one 1F1B order directive per rank.

    This preserves the old PipelineSchedule builder's non-None per-rank op order:
    rank r warms up with ``n_ranks - 1 - r`` forwards, then alternates forward
    and fused backward until all microbatches have completed.
    """
    _validate_positive("n_ranks", n_ranks)
    _validate_positive("n_mbs", n_mbs)

    rows: list[list[_Op]] = [[] for _ in range(n_ranks)]
    fwd_mb = [0] * n_ranks
    bwd_mb = [0] * n_ranks

    for rank in range(n_ranks):
        warmup = n_ranks - 1 - rank
        for _ in range(min(warmup, n_mbs)):
            rows[rank].append(_Op(pp=rank, mb=fwd_mb[rank], pass_="F"))
            fwd_mb[rank] += 1

    for rank in range(n_ranks):
        while bwd_mb[rank] < n_mbs:
            if fwd_mb[rank] < n_mbs:
                rows[rank].append(_Op(pp=rank, mb=fwd_mb[rank], pass_="F"))
                fwd_mb[rank] += 1
            rows[rank].append(_Op(pp=rank, mb=bwd_mb[rank], pass_="B"))
            bwd_mb[rank] += 1

    return [_order_directive(row) for row in rows]


def build_zerobubble_schedule(n_ranks: int, n_mbs: int) -> list[dict]:
    """Return one ZeroBubble order directive per rank.

    This is the old ZB-1 schedule builder expressed as order directives. Each
    rank owns one pipeline stage, fused backward is split into BI/BW, and BW is
    deferred by ``rank`` backward-input steps.
    """
    _validate_positive("n_ranks", n_ranks)
    _validate_positive("n_mbs", n_mbs)

    rows: list[list[_Op]] = []
    for rank in range(n_ranks):
        warmup = min(n_ranks - 1 - rank, n_mbs)
        fwd_bwd_ops = n_mbs - warmup
        cooldown_ops = n_mbs - fwd_bwd_ops
        deferred_bw_count = rank

        fwd_mb = 0
        bwd_mb = 0
        bw_mb = 0
        bwdi_count = 0
        emitted_bw_count = 0
        ops: list[_Op] = []

        for op_idx in range(warmup + fwd_bwd_ops + cooldown_ops):
            if op_idx < warmup:
                ops.append(_Op(pp=rank, mb=fwd_mb, pass_="F"))
                fwd_mb += 1
                continue

            if op_idx < warmup + fwd_bwd_ops:
                ops.append(_Op(pp=rank, mb=fwd_mb, pass_="F"))
                fwd_mb += 1

            ops.append(_Op(pp=rank, mb=bwd_mb, pass_="BI"))
            bwd_mb += 1
            bwdi_count += 1

            if bwdi_count > deferred_bw_count:
                ops.append(_Op(pp=rank, mb=bw_mb, pass_="BW"))
                bw_mb += 1
                emitted_bw_count += 1

        while emitted_bw_count < bwdi_count:
            ops.append(_Op(pp=rank, mb=bw_mb, pass_="BW"))
            bw_mb += 1
            emitted_bw_count += 1

        rows.append(ops)

    return [_order_directive(row) for row in rows]


def build_interleaved_1f1b_schedule(
    n_ranks: int,
    n_mbs: int,
    n_virtual_stages: int,
) -> list[dict]:
    """Return one interleaved 1F1B order directive per physical rank.

    Rank ``r`` owns virtual pipeline stages ``r + k * n_ranks``. The op order is
    the old interleaved PipelineSchedule order with the placeholder ``None`` slots
    omitted.
    """
    _validate_positive("n_ranks", n_ranks)
    _validate_positive("n_mbs", n_mbs)
    _validate_positive("n_virtual_stages", n_virtual_stages)

    microbatches_per_round = n_ranks

    def warmup_ops(rank: int) -> int:
        return min(
            (n_virtual_stages - 1) * n_ranks + 2 * (n_ranks - 1 - rank),
            n_mbs * n_virtual_stages,
        )

    def forward_stage_index(step: int, rank: int) -> int:
        local_index = (step // microbatches_per_round) % n_virtual_stages
        return local_index * n_ranks + rank

    def backward_stage_index(step: int, rank: int, warmup: int) -> int:
        local_index = (
            n_virtual_stages
            - 1
            - ((step - warmup) // microbatches_per_round) % n_virtual_stages
        )
        return local_index * n_ranks + rank

    rows: list[list[_Op]] = []
    for rank in range(n_ranks):
        warmup = warmup_ops(rank)
        microbatch_ops = n_virtual_stages * n_mbs
        fwd_bwd_ops = microbatch_ops - warmup
        cooldown_ops = microbatch_ops - fwd_bwd_ops

        fwd_mb: dict[int, int] = defaultdict(int)
        bwd_mb: dict[int, int] = defaultdict(int)
        ops: list[_Op] = []

        for op_idx in range(warmup + fwd_bwd_ops + cooldown_ops):
            if op_idx < warmup:
                stage = forward_stage_index(op_idx, rank)
                mb = fwd_mb[stage]
                fwd_mb[stage] += 1
                ops.append(_Op(pp=stage, mb=mb, pass_="F"))
            elif op_idx < warmup + fwd_bwd_ops:
                stage = forward_stage_index(op_idx, rank)
                mb = fwd_mb[stage]
                fwd_mb[stage] += 1
                ops.append(_Op(pp=stage, mb=mb, pass_="F"))

                stage = backward_stage_index(op_idx, rank, warmup)
                mb = bwd_mb[stage]
                bwd_mb[stage] += 1
                ops.append(_Op(pp=stage, mb=mb, pass_="B"))
            else:
                stage = backward_stage_index(op_idx, rank, warmup)
                mb = bwd_mb[stage]
                bwd_mb[stage] += 1
                ops.append(_Op(pp=stage, mb=mb, pass_="B"))

        rows.append(ops)

    return [_order_directive(row) for row in rows]


def build_dualpipev_schedule(n_ranks: int, n_mbs: int) -> list[dict]:
    """Return DualPipeV order directives with overlapped FWD/BWD slots.

    Rank ``r`` owns V-layout stages ``r`` and ``2 * n_ranks - 1 - r``. The
    old ``FWD_BWD`` chunk is represented as one nested order filter group
    containing the corresponding FWD filter and fused BWD filter.
    """
    _validate_positive("n_ranks", n_ranks)
    _validate_positive("n_mbs", n_mbs)
    if n_mbs < 2 * n_ranks:
        raise ValueError(
            f"dualpipev requires n_mbs >= 2 * n_ranks, got n_mbs={n_mbs}, "
            f"n_ranks={n_ranks}"
        )

    rows: list[list[_Slot]] = [[] for _ in range(n_ranks)]

    for rank in range(n_ranks):
        s0 = rank
        s1 = 2 * n_ranks - 1 - rank
        slots = rows[rank]
        counts: dict[tuple[int, str], int] = {}
        weight_queue: list[tuple[int, int]] = []

        def count(stage: int, key: str) -> int:
            return counts.get((stage, key), 0)

        def inc(stage: int, key: str) -> None:
            counts[(stage, key)] = count(stage, key) + 1

        def inc_bwd(stage: int) -> None:
            inc(stage, "i")
            inc(stage, "w")

        def append_op(stage: int, mb: int, pass_: Pass) -> None:
            slots.append([_Op(pp=stage, mb=mb, pass_=pass_)])

        def fwd(stage: int) -> None:
            mb = count(stage, "f")
            append_op(stage, mb, "F")
            inc(stage, "f")

        def bwdi(stage: int) -> None:
            mb = count(stage, "i")
            # Virtual stage 0 consumes model inputs (for example integer token
            # IDs), not a differentiable activation received from another
            # stage.  It therefore has no activation-input boundary at which
            # BI can stop and BW can resume.
            if stage == 0:
                append_op(stage, mb, "B")
                inc_bwd(stage)
                return
            append_op(stage, mb, "BI")
            weight_queue.append((stage, mb))
            inc(stage, "i")

        def drain_w() -> None:
            if not weight_queue:
                return
            stage, mb = weight_queue.pop(0)
            append_op(stage, mb, "BW")
            inc(stage, "w")

        def full_bwd(stage: int) -> None:
            mb = count(stage, "i")
            append_op(stage, mb, "B")
            inc_bwd(stage)

        def overlap_fb(fwd_stage: int, bwd_stage: int) -> None:
            fwd_mb = count(fwd_stage, "f")
            bwd_mb = count(bwd_stage, "i")
            slots.append([
                _Op(pp=fwd_stage, mb=fwd_mb, pass_="F"),
                _Op(pp=bwd_stage, mb=bwd_mb, pass_="B"),
            ])
            inc(fwd_stage, "f")
            inc_bwd(bwd_stage)

        # Phase 1: F0 warmup.
        for _ in range((n_ranks - rank - 1) * 2):
            fwd(s0)

        # Phase 2: F0F1.
        for _ in range(rank + 1):
            fwd(s0)
            fwd(s1)

        # Phase 3: I1 W1 F1.
        for _ in range(n_ranks - rank - 1):
            bwdi(s1)
            drain_w()
            fwd(s1)

        # Phase 4: Main overlapped F0B1 + F1B0.
        for i in range(n_mbs - n_ranks * 2 + rank + 1):
            if i == 0 and rank == n_ranks - 1:
                fwd(s0)
                full_bwd(s1)
            else:
                overlap_fb(s0, s1)
            overlap_fb(s1, s0)

        # Phase 5: B1 + F1B0.
        for _ in range(n_ranks - rank - 1):
            full_bwd(s1)
            overlap_fb(s1, s0)

        # Phase 6: B1B0, switching the second half to split backward.
        enable_zb = False
        for i in range(rank + 1):
            if i == (rank + 1) // 2 and rank % 2 == 1:
                enable_zb = True
            (bwdi if enable_zb else full_bwd)(s1)
            if i == (rank + 1) // 2 and rank % 2 == 0:
                enable_zb = True
            (bwdi if enable_zb else full_bwd)(s0)

        # Phase 7: W0 B0.
        for _ in range(n_ranks - rank - 1):
            drain_w()
            (bwdi if enable_zb else full_bwd)(s0)

        # Phase 8: W0.
        for _ in range(rank + 1):
            drain_w()

    return [_order_directive_from_slots(row) for row in rows]


@dataclass(frozen=True)
class _AFPipeRegion:
    """One ordered model region in the disaggregated A/F pipeline."""

    tag: dict[str, int]
    order_filter: dict[str, int | str]
    devices: tuple[int, ...]


def build_afpipe_baseline_schedule(
    place_directives: Sequence[Mapping],
    n_mbs: int,
    n_layers: int,
    *,
    expected_actor_groups: int | None = None,
) -> list[dict]:
    """Build a safe actor-local AF-Pipe baseline order.

    The logical forward path is::

        input, ATTN0, FFN0, ..., ATTN_L, output

    where ``ATTN_L`` is the dense tail after the final expert block.  Adjacent
    regions placed on the same device set form one local pipeline chunk.  Each
    chunk runs all of its microbatches in the forward direction, then chunks
    are visited in reverse for backward.  This deliberately conservative
    baseline provides a valid, non-overlapping order and connects all chunks
    owned by an actor group.  More aggressive AF-Pipe 1F1B policies can replace
    the ordering without changing placement or communication lowering.

    ``place_directives`` is used as the source of truth for actor ownership;
    neither ATTN nor FFN group counts are inferred from GPU ordinals.
    """
    _validate_positive("n_mbs", n_mbs)
    _validate_positive("n_layers", n_layers)
    if expected_actor_groups is not None:
        _validate_positive("expected_actor_groups", expected_actor_groups)

    places = [directive for directive in place_directives if directive.get("op") == "place"]
    if not places:
        raise ValueError("afpipe requires at least one place directive")

    n_pp_stages = _num_pp_stages(places)

    def make_region(
        tag: dict[str, int],
        order_filter: dict[str, int | str],
    ) -> _AFPipeRegion:
        return _AFPipeRegion(
            tag=tag,
            order_filter=order_filter,
            devices=_resolve_region_devices(tag, places),
        )

    regions: list[_AFPipeRegion] = [
        make_region(
            {"PP": 0},
            {"PP": 0, "ATTN": "-", "FFN": "-"},
        )
    ]
    for layer in range(n_layers):
        pp = _layer_pp_stage(layer, n_layers, n_pp_stages)
        regions.append(make_region({"PP": pp, "ATTN": layer}, {"ATTN": layer}))
        regions.append(make_region({"PP": pp, "FFN": layer}, {"FFN": layer}))

    final_pp = _layer_pp_stage(n_layers - 1, n_layers, n_pp_stages)
    regions.append(
        make_region(
            {"PP": final_pp, "ATTN": n_layers},
            {"ATTN": n_layers},
        )
    )
    regions.append(
        make_region(
            {"PP": n_pp_stages - 1},
            {
                "PP": n_pp_stages - 1,
                "ATTN": "-",
                "FFN": "-",
            },
        )
    )

    # Co-located consecutive regions are one local pipeline chunk.  The same
    # actor group may own multiple non-consecutive chunks.
    chunks: list[list[_AFPipeRegion]] = []
    for region in regions:
        if chunks and chunks[-1][-1].devices == region.devices:
            chunks[-1].append(region)
        else:
            chunks.append([region])

    actor_order: list[tuple[int, ...]] = []
    rows: dict[tuple[int, ...], list[dict[str, int | str]]] = {}
    for region in regions:
        if region.devices not in rows:
            actor_order.append(region.devices)
            rows[region.devices] = []

    if expected_actor_groups is not None and len(rows) != expected_actor_groups:
        raise ValueError(
            "AF-Pipe placement actor-group count does not match --ranks: "
            f"placement has {len(rows)} distinct device sets {actor_order}, "
            f"but --ranks={expected_actor_groups}"
        )

    def append_op(region: _AFPipeRegion, mb: int, pass_: Pass) -> None:
        rows[region.devices].append(
            {**region.order_filter, "MB": mb, "PASS": pass_}
        )

    for chunk in chunks:
        for mb in range(n_mbs):
            for region in chunk:
                append_op(region, mb, "F")

    for chunk in reversed(chunks):
        for mb in range(n_mbs):
            for region in reversed(chunk):
                append_op(region, mb, "B")

    return [
        {"op": "order", "filters": [[flt] for flt in rows[devices]]}
        for devices in actor_order
    ]


def infer_afpipe_num_layers(
    place_directives: Sequence[Mapping],
    *,
    fallback: int | None = None,
) -> int:
    """Infer the AFD layer count from exact placement filters when possible.

    AFD emits ``FFN0 .. FFN_(L-1)`` and ``ATTN0 .. ATTN_L``.  Wildcard-only
    placements do not encode ``L`` and therefore require ``fallback``.
    """
    exact_attn: list[int] = []
    exact_ffn: list[int] = []
    for directive in place_directives:
        if directive.get("op") != "place":
            continue
        spec = directive.get("filter")
        if not isinstance(spec, Mapping):
            continue
        if isinstance(spec.get("ATTN"), int):
            exact_attn.append(int(spec["ATTN"]))
        if isinstance(spec.get("FFN"), int):
            exact_ffn.append(int(spec["FFN"]))

    candidates: list[int] = []
    if exact_attn:
        candidates.append(max(exact_attn))
    if exact_ffn:
        candidates.append(max(exact_ffn) + 1)
    if candidates and any(candidate != candidates[0] for candidate in candidates):
        raise ValueError(
            "AF-Pipe placement encodes inconsistent layer counts: "
            f"ATTN indices={sorted(set(exact_attn))}, "
            f"FFN indices={sorted(set(exact_ffn))}"
        )
    if candidates:
        _validate_positive("inferred n_layers", candidates[0])
        return candidates[0]
    if fallback is None:
        raise ValueError(
            "cannot infer AF-Pipe layer count from wildcard placement; "
            "provide n_layers"
        )
    _validate_positive("fallback n_layers", fallback)
    return fallback


def _num_pp_stages(place_directives: Sequence[Mapping]) -> int:
    pp_indices: set[int] = set()
    for directive in place_directives:
        spec = directive.get("filter")
        if isinstance(spec, Mapping) and isinstance(spec.get("PP"), int):
            pp_indices.add(int(spec["PP"]))
    if not pp_indices:
        raise ValueError("AF-Pipe placement must contain exact PP endpoint filters")
    if min(pp_indices) != 0:
        raise ValueError(f"AF-Pipe PP indices must start at 0, got {sorted(pp_indices)}")
    return max(pp_indices) + 1


def _layer_pp_stage(layer: int, n_layers: int, n_pp_stages: int) -> int:
    for stage in range(n_pp_stages):
        if stage * n_layers // n_pp_stages <= layer < (stage + 1) * n_layers // n_pp_stages:
            return stage
    raise ValueError(
        f"layer {layer} is not owned by any of {n_pp_stages} PP stages "
        f"for a {n_layers}-layer model"
    )


def _resolve_region_devices(
    tag: Mapping[str, int],
    place_directives: Sequence[Mapping],
) -> tuple[int, ...]:
    matches: list[tuple[int, ...]] = []
    for directive in place_directives:
        spec = directive.get("filter")
        if not isinstance(spec, Mapping) or not _matches_place_filter(tag, spec):
            continue
        devices = directive.get("devices", directive.get("device"))
        if not isinstance(devices, list) or not devices:
            raise ValueError(f"place directive requires non-empty devices: {directive}")
        matches.append(tuple(sorted(int(device) for device in devices)))
    if len(matches) != 1:
        raise ValueError(
            f"AF-Pipe region {dict(tag)} must match exactly one place directive, "
            f"matched {len(matches)} device sets: {matches}"
        )
    return matches[0]


def _matches_place_filter(tag: Mapping[str, int], spec: Mapping) -> bool:
    for key, expected in spec.items():
        if expected == "*":
            if key not in tag:
                return False
        elif expected == "-":
            if key in tag:
                return False
        elif tag.get(key) != expected:
            return False
    return True


def _validate_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def main() -> None:
    from run_harness import main as harness_main

    harness_main()


if __name__ == "__main__":
    main()
