# Transaction-aware interference-free serialization

## Task summary

Replace interference-free profiling's actor-local dispatch-order chain with one deterministic, transaction-aware static order that serializes work on each rank without introducing cross-rank communication cycles.

## Objectives

- Build one global ordering of logical execution units from finalized ActorGroup DAGs, grouping all endpoints of each P2P or collective transaction into the same unit.
- Project that order onto every actor so all ranks issue shared communication transactions consistently while preserving each node's declared physical stream.
- Serialize consecutive projected operations per rank, relying on stream FIFO on the same physical stream and CUDA events only across physical streams.
- Reject an interference-free profile before execution when the transaction-collapsed dependency graph is cyclic or communication endpoints cannot be grouped unambiguously.
- Keep the policy deterministic and general over communication kind, ActorGroup width, placement, and schedule tags.

## Non-goals

- Changing the finalized training DAG, schedule directives, normal execution order, simulator replay dependencies, communicator selection, or communication kernels.
- Adding dynamic ready-queue execution, relaxing dependencies, or recovering from an invalid schedule at runtime.
- Globally serializing operations on disjoint devices; serialization is per physical rank, while peers of one communication transaction necessarily execute together.
- Supporting distributed work hidden inside compute nodes. Distributed operations must remain explicit Piper communication nodes.
- Addressing CUPTI overhead, host issue latency, or unrelated profile-and-simulate accuracy differences.

## Correctness criteria

- Every interference-free actor order is a topological order of that actor's finalized DAG.
- Matching P2P endpoints and all endpoints of a collective have one shared position in the global logical order; every actor's projection orders shared transactions consistently.
- Interference-free execution has at most one Piper node executing at a time on each rank, except that one node's internal kernels remain an atomic measured region.
- Same-physical-stream serialization uses FIFO without an additional event wait; cross-stream serialization uses an event dependency.
- The profiling policy introduces no communication wait cycle when the transaction-collapsed dependency graph is acyclic, including schedules with opposite forward/backward P2P directions.
- A cyclic or malformed transaction graph is rejected deterministically with the involved units and actor groups identified.
- Normal execution and both simulator replay modes retain the original finalized DAG and dispatch semantics.
- The conventional PP2/DP2/EP2 regression that exposed `RECV(F, MB1)` versus `SEND(B, MB0)` completes interference-free profiling, and focused AFD profiling remains deadlock-free.

## Subtasks

1. Construct and validate the global logical-unit graph and deterministic topological order.
2. Pass each ActorGroup's projected order into actor setup and use it only for interference-free prepared execution.
3. Make serialization waits physical-stream-aware and add focused unit/GPU regression coverage.

## Assumptions / open questions

- Existing communication identity and participant validation is authoritative: P2P uses `comm_id` across SEND/RECV endpoints, and collectives use `(node_kind, comm_id)` across their participants.
- A source endpoint may complete before a directional destination, but consistent transaction issue order is still required when transactions share ranks or communicators.
- If collapsing transaction endpoints creates a dependency cycle, no fully serialized execution consistent with the finalized DAG exists; rejection is preferable to a runtime heuristic.
- The global order is profiling-control metadata only. It must not be serialized as new model DAG dependencies or used by simulator replay.

## Implementation plan

1. **Build the global profiling order once from finalized ActorGroup DAGs.**
   - Add a private multi-DAG helper in `src/piper.py`, alongside the existing cross-actor communication validation.
   - Map matching SEND/RECV endpoints to one P2P unit and matching collective endpoints to one unit using communication kind, `comm_id`, and declared participants. Map every non-communication node to an actor-local unit.
   - Project every finalized DAG edge onto the unit graph, ignore edges internal to one unit, and compute a stable topological order. Existing actor dispatch position is only a deterministic tie-break among ready units and does not add precedence.
   - Reject missing/ambiguous transaction identity, duplicate local endpoints, incomplete participant representation, or a cycle in the collapsed graph. Return one exact UID order per ActorGroup without mutating any DAG or node metadata.

2. **Pass the projected order through actor setup.**
   - In `src/compile.py`, compute the orders from the final RPC-safe ActorGroup DAGs and pass each order with the corresponding DAG to every rank in that ActorGroup.
   - Extend the internal `PiperActor.load_training_dag` interface in `src/actor.py` to validate that the supplied order covers every node exactly once and respects every DAG edge.
   - Keep the ordinary prepared execution unchanged. Build the interference-free prepared execution from the supplied order while retaining the existing declared-stream mapping. Do not add DAG edges or persist the profiling order in `ProfiledTrainingPlan`.

3. **Make the existing serialization chain stream-aware.**
   - In `src/executors.py`, retain one completion event after each serialized node so a following node on another stream can wait for it.
   - Skip the explicit wait when consecutive nodes use the same physical stream; CUDA FIFO already supplies that dependency.
   - Keep every finalized execution-plan predecessor wait unchanged. The profiling chain adds ordering only between consecutive nodes in the approved static projection and never relaxes an encoded dependency.

### Compatibility and risks

- `load_training_dag` is an internal actor RPC; its call sites and focused fakes must be updated together. Normal `run_dag` behavior remains byte-for-byte based on `_dispatch_order`.
- A collapsed-graph cycle means no transaction-consistent, fully serialized policy exists without relaxing a finalized dependency. The implementation will fail during actor loading rather than try another runtime schedule.
- The helper assumes all distributed work is represented by Piper communication nodes with valid identity/participant metadata. A synchronous host read inside a communication node may delay host issue, but it cannot create a cycle when every actor follows the common transaction order.
- The stable order may differ from normal issue order for otherwise independent nodes. This is intentional measurement control; replay still uses the original DAG and measured service samples.

## Evaluation plan

1. **Global order construction** — covers consistent transaction ordering and rejection.
   - Add focused tests near the existing multi-DAG ordering tests using generic two-actor DAGs with two opposite-direction P2P transactions whose local normal dispatch orders disagree. Assert both projections use one common transaction order and every original DAG edge is respected.
   - Add a topology-neutral collapsed-cycle case and assert an early diagnostic names the cyclic units. Do not encode the exact DualPipeV or PP2 schedule.
   - Cover a multi-endpoint collective and disjoint local compute to verify endpoint grouping without globally serializing disjoint ranks.

2. **Actor preparation and executor serialization** — covers the profiling-only data path.
   - Extend actor/compile setup tests to verify every group member receives the same projected UID order and that normal preparation retains `_dispatch_order` while interference-free preparation uses the projection.
   - Extend `test/test_execution_plan.py` so serialized consecutive nodes on one physical stream use FIFO with no event wait, while consecutive nodes on distinct streams perform exactly one serialization wait before the second node.
   - Assert all ordinary DAG dependency waits remain present and neither prepared execution mutates the finalized DAG.

3. **Focused CPU regression**
   - Run the new ordering, actor-group topology, actor profiling, execution-plan, profile-stream, DAG-builder-equivalence, and simulator tests.
   - Run Python compilation for modified modules and `git diff --check`.

4. **Focused GPU regression**
   - On the available H100 node, rerun the small `pp2_dp2_ep2_zero2` profile-and-simulate workload that previously formed the cross-direction P2P cycle. Use the existing ten-minute guard and verify both interference-free and normal captures complete.
   - Rerun the small unequal-group AFD profile-and-simulate workload and verify capture, replay, and attribution still complete.
   - Do not run the full unit or E2E suites; these two workloads exercise the changed profiling order and the existing directional routed path.

## Implementation log

- Added one global logical-unit ordering pass over finalized ActorGroup DAGs.
  P2P SEND/RECV endpoints share a unit, non-P2P communication endpoints share
  a unit by `(node_kind, comm_id)`, and non-communication nodes remain local.
  The pass projects only existing DAG edges, uses the ordinary local dispatch
  position only as a deterministic tie-break, and does not mutate a DAG.
- The projected order is computed before actor loading and sent identically to
  every member of an ActorGroup. Actors validate exact node coverage and every
  original edge, retain `_dispatch_order` for normal execution, and prepare a
  separate order only for interference-free profiling.
- Interference-free serialization now skips an event wait for consecutive work
  on the same cached physical CUDA stream; an event handoff remains for a
  cross-stream transition. Every ordinary execution-plan predecessor wait is
  unchanged.
- The cycle diagnostic reports both cyclic logical units and their ActorGroup
  DAG-index/UID members. This was added after review so a rejected schedule can
  be located without reproducing it on GPUs.
- The global stable order also contains actor-local units, but their relative
  placement is only projected locally: no cross-actor edge, barrier, or runtime
  synchronization is created for disjoint local work.
- The implementation assumes every distributed operation is an explicit
  `*_COMM` node. Non-P2P communication nodes must provide a unique `comm_id`,
  sorted unique `participants`, and endpoint `device` sets whose union equals
  those participants. Hidden distributed work inside compute remains outside
  the policy by design.
- No material deviation from the approved scope was required.

## Evaluation results

- `cpu-venv/bin/python -m py_compile src/piper.py src/compile.py src/actor.py src/executors.py test/test_interference_free_serialization.py` — passed.
- Focused ordering, ActorGroup, actor-profile, execution-plan, profile-stream,
  DAG-equivalence, and simulator tests — 100 passed.
- Focused AFD routed lowering, full lowering, order-directive, profiling, and
  profile-validation tests — 114 passed (14 upstream Torch deprecation
  warnings).
- `git diff --check` — passed.
- Conventional Qwen 9M `pp2_dp2_ep2_zero2`, four ranks, two microbatches,
  `1/2` profile warmup/active iterations — completed both normal and
  interference-free capture instead of reproducing the prior P2P cycle.
  Kineto validation covered 348 actor/mode nodes, 696 occurrences, and 63,613
  launches with exact activity equality; normal and IF replay had the same 174
  events. Artifact: `out/20260818_083311/step_time_profiles.json`.
- Unequal-group AFD Qwen 9M, ActorGroups `[0,1]` and `[2]`, two microbatches,
  `1/2` profile warmup/active iterations — completed capture and replay.
  Kineto validation covered 430 actor/mode nodes, 860 occurrences, and 53,536
  launches with exact activity equality; normal and IF replay had the same 215
  events. Artifact: `out/20260818_083520/step_time_profiles.json`.
- The full unit and E2E suites were intentionally not run; the approved focused
  tests cover the changed construction, actor setup, executor, routed, and
  replay paths.

## Remaining known limitations

- A cycle in the transaction-collapsed graph is rejected because no fully
  serialized transaction-consistent execution can preserve all of its DAG
  edges; Piper does not relax the schedule or use a dynamic runtime workaround.
- Distributed operations hidden inside compute nodes cannot be grouped or
  validated by this policy.
- The GPU integrations used reduced focused sampling rather than the default
  `5/30`; sampling configuration is unrelated to the ordering change.

## Commit message

`Serialize interference-free profiles by communication transaction`
