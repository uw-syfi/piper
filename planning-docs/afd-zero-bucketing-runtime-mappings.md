# Routed ZeRO bucketing runtime mappings

## Task summary

Make parameter bucketing preserve Piper's structured runtime-value ABI when a
bucket rewrite replaces a compute node, so schedules can combine dense
parameter/gradient sharding, bucketing, and routed boundaries. The motivating
failures cover both cross-group AFD routing and same-group DualPipeV ragged
all-to-all.

## Objectives

- Preserve every externally visible activation and gradient mapping across a
  bucketed compute region.
- Keep routed payload/count markers and their communication lowering valid when
  adjacent dense regions are bucketed.
- Allow routed AF-Pipe and DualPipeV schedules to execute with dense-region
  parameter and gradient sharding plus bucketing.

## Non-goals

- Change bucket partitioning, routed transport semantics, placement, or
  pipeline ordering.
- Add bucketing to regions governed by a `shard` directive.
- Change ZeRO numerical or gradient-reduction semantics.

## Correctness criteria

- Replacing a compute node with a bucket chain does not lose, duplicate, or
  misaddress any structured runtime input or output value.
- Backward gradients follow the same logical producer/consumer relationships as
  before bucketing.
- Routed marker metadata remains attached to the correct bucket-chain boundary.
- Existing unbucketed and non-routed schedules retain their current DAG and
  execution behavior.
- Routed workloads using either ragged A2A or cross-group M-to-N/N-to-M lower
  and execute without missing runtime mappings, input slots, or invalid ZeRO
  lifetime chains.

## Assumptions / open questions

- External forward inputs belong on the first bucket and external forward
  outputs on the last bucket; the backward boundary is the reverse chain.
- UID-keyed mappings on adjacent nodes must be rewritten to the corresponding
  bucket-chain boundary rather than copied unchanged.
- The output embedding remains unbucketed because schedule tags do not expose
  internal bucket indices needed by its direct ZeRO lifetime order chain.
- The reproduced DualPipeV failure is
  `s0.seg0.bucket1 -> s0.seg1`: the final dense bucket retained the routed
  marker but lost `runtime_outputs_by_destination`, so A2A insertion rejected
  the boundary before execution.
