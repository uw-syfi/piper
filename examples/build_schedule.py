"""Build JSON order directives for common pipeline schedules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

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


def _validate_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def main() -> None:
    from test_harness import main as harness_main

    harness_main()


if __name__ == "__main__":
    main()
