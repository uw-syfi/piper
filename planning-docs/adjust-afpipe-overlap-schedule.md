# Adjust AF-Pipe overlap schedule

## Task summary

Replace the current interleaved-1F1B-derived order directives in `afd_interleaved_1f1b_pp2_v4_mbs4.json` with a dependency-valid AF-Pipe ordering for four microbatches, then regenerate its schedule visualization for inspection before any Ray execution.

## Objectives

- Keep the existing AFD placement, replication, sharding, and four-microbatch split unchanged.
- Time-multiplex microbatches across the disaggregated attention and FFN actor groups so independent groups can make progress concurrently.
- Admit `PP0/MB1` at the earliest legal point: immediately after `ATTN0/PP0/MB0` on its actor group.
- Preserve forward/backward data dependencies and serialize chunks sharing a device group.
- Produce a persistent visualization that makes actor-group overlap and bubbles auditable.

## Non-goals

- Do not change Piper's scheduler, DAG builder, runtime, visualizer, model, or placement topology.
- Do not reproduce DisagMoE's separate communication streams in the visualization; Piper's routed communication nodes remain dependency-inferred rather than explicit order-directive entries.
- Do not execute the schedule on the new Ray session until the user approves the regenerated visualization.
- Do not tune the ordering from measured kernel or communication durations.

## Correctness criteria

- Every forward and backward model chunk for all four microbatches is ordered exactly once on its owning device group.
- All explicit order edges are acyclic and consistent with model data dependencies.
- No two compute chunks assigned to the same device group can overlap.
- `PP0/MB1` immediately follows `ATTN0/PP0/MB0` in the first attention actor group's order.
- Work whose dependencies are satisfied is not unnecessarily serialized behind work on a different actor group.
- Piper accepts and visualizes the resulting schedule without errors.

## Subtasks

1. Derive an AF-Pipe order from the model DAG, current placement, and four-microbatch constraint.
2. Replace only the schedule's order directives.
3. Run schedule parsing/DAG validation and regenerate the PNG visualization.

## Assumptions / open questions

- DisagMoE Figure 8 is an illustrative schedule with a different layer/microbatch scale; this task adapts its dependency-driven time-multiplexing principle to Piper's four virtual stages and four microbatches.
- "Immediately after" means adjacency in the owning actor group's compute order; communication on independent streams is not represented as an intervening order entry.
- The current placement and device-group topology remain the intended physical mapping.

## Implementation plan

1. Load the persistent schedule and retain its `place`, `replicate`, `shard`, and four-microbatch `split` directives unchanged.
2. Replace its four current interleaved-1F1B-derived `order` directives with the actor-local rows produced by Piper's existing `build_afpipe_baseline_schedule` for four layers and four microbatches. This generator already batches ready microbatches at each disaggregated component boundary, including the required `input/MB1` then `ATTN0/MB1` sequence immediately after `ATTN0/MB0`.
3. Write the generated rows back to the same schedule JSON without changing scheduling/runtime code.
4. Render the updated JSON with `visualize_order_directives` to `out/afd_interleaved_1f1b_pp2_v4_mbs4.png`.

Compatibility risk: the filename retains its historical `interleaved_1f1b` name even though the corrected order is AF-Pipe-specific. Renaming it would break the persistent path already requested and is outside this change.

## Evaluation plan

- **Exact coverage and ownership serialization:** compare the generated rows against the schedule's placed model regions and assert every forward/backward region–microbatch combination appears exactly once in its owning row.
- **Required early admission:** inspect/assert that the first attention row starts `input MB0`, `ATTN0 MB0`, `input MB1`, `ATTN0 MB1`.
- **Dependency validity and acyclicity:** apply the completed directives through the existing AFD full-lowering path on the Qwen graph without dispatching GPU compute.
- **Generator regression:** run the focused AF-Pipe schedule-builder tests.
- **Visualization:** render the PNG successfully and inspect it for four grouped device rows and the intended cross-group overlap. No Ray/GPU execution will be started before user review.

## Implementation log

- The prior schedule used standard interleaved-1F1B actor rows; on devices `[0, 4, 8, 10]`, it placed `ATTN2/PP1/MB0` between `ATTN0/PP0/MB0` and admission of `PP0/MB1`, creating avoidable serialization across disaggregated actor groups.
- DisagMoE's published figure uses a larger illustrative configuration. The proposed order uses Piper's existing AF-Pipe adaptation rather than copying figure coordinates literally.
