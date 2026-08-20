# Global communication ordering

## Phase 1: Scope

### Task summary

Replace Piper's independent routed-collective and directional-P2P ordering
analyses with one global communication-transaction analysis. The builder must
account for all DAG-visible communication when selecting actor-local host issue
and physical-stream FIFO orders, reject mandatory cycles before execution, and
finalize a deterministic deadlock-free order when the modeled constraints are
acyclic.

### Objectives

- Represent pipeline and dense-state P2P, routed A2A/M-to-N/N-to-M, and
  replica/ZeRO collectives using explicit transaction identity, participants,
  endpoints, communicator, and physical-stream metadata rather than model tags.
- Project actor-DAG reachability, including paths through compute, onto a single
  global communication graph.
- Treat a matched P2P pair or all endpoints of one collective as one logical
  transaction for global ordering, while retaining its actor-local endpoint.
- Detect and diagnose cycles involving any mixture of communication kinds and
  actor groups, including cycles that the current per-domain P2P analysis
  cannot see.
- Select one deterministic global linear extension of the mandatory
  transaction constraints, then project it into every actor's communication
  issue sequence and each physical stream's FIFO sequence.
- Preserve overlap: cross-stream issue order must not become a cross-stream
  completion dependency, and unrelated communication or compute must not be
  serialized merely because the builder chose a host launch order.
- Make the executor and simulator consume the same finalized DAG, stream FIFO,
  and host issue order without runtime dependency relaxation or deadlock
  heuristics.

### Non-goals

- Changing process-group membership, directional P2P streams, schedule-declared
  routed streams, collective algorithms, or communication payload formats.
- Inferring communication from tags, model structure, tensor values, or
  schedule-specific names.
- Guaranteeing progress in the presence of malformed/mismatched NCCL calls,
  implicit communication hidden inside compute nodes, transport failures, or
  GPU/NCCL implementation bugs outside the declared transaction model.
- Adding a dynamic ready queue, receive-last runtime policy, dependency
  relaxation, or executor-only synchronization.
- Serializing completion across different physical streams solely to impose a
  deterministic host launch order.

### Correctness criteria

1. Every DAG-visible communication operation belongs to exactly one validated
   logical transaction with consistent endpoints, participants, communicator,
   and physical-stream metadata; malformed or incomplete transactions are
   rejected before execution.
2. If any endpoint of transaction A reaches an endpoint of transaction B in an
   actor DAG, including through compute or another communication kind, the
   global graph contains the mandatory precedence A before B.
3. A mandatory communication cycle involving any supported mixture of P2P,
   routed, replica, or ZeRO communication is rejected with a diagnostic that
   identifies the transaction IDs, kinds, actor groups, streams, and constraint
   responsible for each cycle edge.
4. For an acyclic input, every actor's finalized host communication issue
   sequence is a projection of one deterministic global transaction order and
   still respects every actor-DAG dependency.
5. Every physical stream's finalized FIFO sequence is consistent with that
   global order at all endpoints and cannot introduce a cycle when projected
   back into the global transaction graph.
6. Cross-stream host issue ordering does not create a completion edge or event
   wait. Only original DAG dependencies and same-physical-stream FIFO constrain
   completion, preserving overlap between otherwise-incomparable work.
7. The finalized executor and normal simulator issue communication in the same
   actor-local order and observe the same DAG dependencies and physical-stream
   FIFO relationships; neither adds or relaxes ordering independently.
8. The motivating PP4 communication pattern is either assigned an order that
   removes its dense-state-P2P/routed-collective wait cycle or is rejected
   before GPU execution with the exact cycle, rather than hanging at runtime.

### Subtasks

- Define and validate a transport-neutral global transaction representation for
  all current communication node kinds.
- Replace the separate routed and P2P ordering passes with global constraint
  construction, cycle reporting, deterministic ordering, and actor projection.
- Finalize stream FIFO edges and host issue metadata without conflating issue
  order with completion dependencies.
- Align executor, simulator, diagnostics, and focused tests with the finalized
  global order.

### Assumptions / open questions

- The progress guarantee is relative to a blocking-transaction model: once an
  actor issues a communication transaction, its host thread may be unable to
  issue a later transaction until the current transaction's matching endpoints
  have issued. A common global order prevents a circular wait under this model.
- A communication DAG node may execute several fixed, contiguous backend calls
  (for example a metadata exchange followed by payload exchange). It may be
  treated as one transaction only when the calls cannot interleave with another
  DAG node and their endpoint/participant ordering is internally consistent.
  Any internal phase with a materially different matching or participant set
  must be represented as an ordered subtransaction or rejected as unanalyzable.
- Non-communication DAG nodes must not issue hidden distributed operations or
  host-block waiting for a future communication transaction. Such behavior
  would be outside the graph and invalidate static deadlock analysis.
- One global total transaction order is intentionally stronger than the
  minimum partial order but does not itself serialize CUDA execution: actors
  project away transactions in which they do not participate, and distinct
  streams retain concurrency after launch.

## Phase 2: Implementation and evaluation plan

### Implementation plan

1. **Normalize every communication transaction in `src/piper.py`.** Replace
   the separate routed and P2P collectors with one internal transaction record
   shared by normal ordering and interference-free serialization. A record
   contains a typed identity, `comm_id`, kind, participants, communicator
   channel(s), and `(actor_dag_index, endpoint_uid, physical_stream)` members.
   Pair `SEND_COMM` with `RECV_COMM`; group collective endpoints by kind and
   `comm_id`; account for one ActorGroup DAG node representing all SPMD ranks
   in that group. Validate endpoint coverage, roles, participants, channels,
   streams, and globally incompatible reuse of a textual `comm_id` before
   ordering.

2. **Finalize non-communication constraints before global analysis.** Keep
   default-stream compute ordering after ActorGroup splitting, then prune ZeRO
   lifetimes and materialize their inter-chain constraints before selecting a
   communication order. This makes schedule, data, compute, dynamic-shape, and
   ZeRO dependencies visible to the global graph. Existing schedule JSON and
   directive semantics remain unchanged.

3. **Build the mandatory global transaction graph.** Collapse every logical
   transaction's actor-local endpoints into one vertex. For every actor DAG,
   add `A -> B` when an endpoint of A reaches an endpoint of B through the DAG;
   retain provenance identifying the actor, endpoint path, and constraint
   kind. Detect strongly connected components before making any optional
   ordering choice. Report an explicit cycle with transaction kind/ID,
   participants, actor groups, streams, and edge provenance.

4. **Choose a deterministic global order.** Compute one topological linear
   extension of the mandatory graph. Among ready transactions, prefer their
   existing graph-derived dispatch positions so communication remains close to
   producers and consumers; use stable typed identity only as the final
   tie-break. Do not inspect model tags or names. Persist the resulting global
   communication index on every endpoint.

5. **Project completion FIFO only within physical streams.** For each actor and
   physical stream, take the local communication subsequence of the global
   order and add temporal edges only between consecutive endpoints on that
   stream. Then materialize the rest of that stream's compute/communication
   FIFO in an order constrained by those edges. No edge is added between
   endpoints on different physical streams merely because their global indices
   differ. Rebuild the transaction graph after stream finalization and verify
   that the selected global order remains a linear extension.

6. **Represent cross-stream host order as an explicit DAG edge.** Extend
   `TrainingDAGEdge.dep_kind` with `issue`. After all physical-stream FIFO
   completion edges are present, project the global transaction order onto each
   actor's communication endpoints. For each consecutive pair, add an `issue`
   edge only when existing data, temporal, FIFO, or previously-added issue
   reachability does not already order the pair. Same-stream pairs therefore
   need no issue edge. This is the transitive reduction of each actor's local
   total communication order and is the fewest issue edges that can guarantee
   every adjacent relation. `_dispatch_order` and diagnostic dispatch levels
   consume all three edge kinds, but `issue` never becomes a completion
   predecessor.

7. **Keep normal CUDA execution dependency-faithful.** `src/executors.py` keeps
   its single static dispatch loop. The loop order itself satisfies issue-only
   predecessors; it performs no CUDA event wait, `Work.wait`, stream sync, or
   device sync for them. Existing DAG completion predecessors retain current
   behavior: same-stream relations use FIFO and cross-stream relations use
   events. No runtime ready queue or transport-kind heuristic is added.

8. **Replay the same issue semantics in `src/simulator.py`.** Lower an `issue`
   DAG edge as a zero-duration edge from the predecessor
   communication endpoint's arrival vertex to the successor's arrival vertex,
   rather than from completion to arrival. This prevents replay from issuing a
   later communication first while retaining cross-stream GPU overlap after
   both operations have arrived. Collective/P2P rendezvous and endpoint service
   timing remain otherwise unchanged.

9. **Keep profiling and diagnostics consistent.** Make
   `_interference_free_serial_orders` consume the same transaction identities
   and respect the finalized issue order while adding its profiling-only full
   serialization. Update `src/visualization.py` and textual order diagnostics
   to show issue-only relations distinctly from data/temporal completion edges.
   Normal profiling continues to use the ordinary finalized dispatch order.

### Compatibility risks and limitations

- The earlier collective-aware implementation used collectives only as
  constraints when selecting directional-P2P FIFO; it still allowed each actor
  to invent a different cross-stream host order. This plan supersedes that
  boundary by finalizing a common host transaction order without adding
  cross-stream completion edges.
- `TrainingDAG.succs`/`preds` and topological ordering include `issue` edges for
  legal host dispatch. Components that reason specifically about completion
  (`build_execution_plan`, actor dependency lists, simulator completion edges,
  and stream-FIFO materialization) must filter edge kinds explicitly so an
  issue-only path is never mistaken for completed CUDA work.
- A single global order can change which ready communication the host launches
  first and therefore performance, but it does not add GPU completion
  serialization across streams. Disjoint actors project away transactions in
  which they do not participate.
- Persisted dispatch-order metadata changes DAG artifact digests. Any digest
  update will be reviewed for unchanged nodes/data dependencies and only the
  intended FIFO/issue-order differences; schedules will not be edited to match
  the implementation.
- Node-level atomicity is valid for current P2P header/payload and routed
  metadata/payload sequences only because the backend issues each sequence
  contiguously before dispatching another DAG node and all phases use validated
  matching participants. If code inspection or tests find a phase with a
  different participant set that can create an independent wait, implementation
  stops to revise the transaction representation rather than hiding it.
- The static guarantee assumes every distributed operation is represented by a
  communication node. Communication launched implicitly from model compute or
  optimizer code remains unsupported and must be rejected if detectable.
- The guarantee is deadlock freedom under the documented blocking-transaction
  model, not a guarantee against NCCL, CUDA, network, or hardware failures.

### Evaluation plan

1. **Transaction schema and normalization — new focused unit coverage.** Add a
   focused communication-ordering test module (or a clearly separated section
   of `test/test_afd_routed_directives.py`) covering P2P, A2A, M-to-N, N-to-M,
   all-reduce, all-gather, and reduce-scatter; ActorGroup-expanded participant
   coverage; malformed pairs/endpoints; channel or stream mismatch; and
   incompatible `comm_id` reuse.

2. **Global reachability and cycle diagnostics.** Build schedule-neutral
   synthetic actor DAGs in which paths through compute and different
   communication kinds establish transaction precedence. Verify a mixed
   P2P/collective cycle is rejected with edge provenance. The regression will
   exercise the structural failure without matching exact AFD, DualPipeV,
   model tags, or node names.

3. **Acyclic ordering and overlap preservation.** Build a mixed graph with
   multiple valid orders and verify determinism, matching actor projections,
   and an acyclic finalized graph. Verify same-stream communication gains FIFO
   temporal edges, while cross-stream consecutive transactions gain only an
   `issue` edge and no completion predecessor/event wait. Verify redundant
   issue edges are omitted when an existing path already orders adjacent local
   transactions.

4. **Dispatch and execution-plan fidelity.** Extend
   `test/test_execution_plan.py` and `test/test_order_directive.py` to verify
   finalized dispatch order respects DAG and issue-only constraints; invalid
   persisted order is rejected; send/receive priority cannot override the
   global order; and the executor's mocked streams wait only for encoded
   cross-stream completion dependencies.

5. **Simulator fidelity.** Extend `test/test_simulator.py` with unit-duration
   profiles to verify communication arrival follows issue order, completion
   still follows rendezvous/service, and unrelated streams overlap. Compare the
   executor execution plan and simulator lowering for identical issue and
   completion predecessor sets.

6. **Interference-free profiling compatibility.** Extend
   `test/test_interference_free_serialization.py` so profiling serialization
   respects the normal global transaction order, remains consistent at all
   endpoints, and adds no normal-mode completion dependency.

7. **Existing AFD PP4 CPU integration.** Lower the existing
   `afd_placement_pp4.json` with the repository AF-Pipe schedule builder, assign
   finalized stream and dispatch order, and inspect the derived communication
   graph. Verify the former dense-state-P2P/M-to-N/N-to-M triangle is absent
   from the finalized issue graph and CPU replay completes. Do not specialize
   implementation logic to this schedule.

8. **Focused CPU regression.** Run only the relevant files:

   ```bash
   cpu-venv/bin/python -m pytest -q \
     test/test_global_communication_ordering.py \
     test/test_execution_plan.py \
     test/test_order_directive.py \
     test/test_afd_routed_directives.py \
     test/test_afd_full_lowering.py \
     test/test_interference_free_serialization.py \
     test/test_simulator.py \
     test/test_dag_builder_equivalence.py \
     test/test_visualization.py
   ```

   If the focused tests remain in existing files rather than a new module, omit
   the nonexistent path. Review rather than mechanically accept DAG digest
   changes.

9. **Focused GPU regression.** With an available Ray session, first rerun the
   PP4 AF-Pipe deadlock smoke for one warmup and one timed iteration with a
   two-minute execution timeout. Then run the existing small AFD PP2, AFD PP4,
   and DualPipeV parameter-parity selectors. A remaining stall is investigated
   from the finalized global/actor issue graphs; no runtime workaround is added.

### Correctness-criterion mapping

| Criterion | Verification |
| --- | --- |
| 1 | Evaluation 1 validates all supported transaction kinds and metadata. |
| 2 | Evaluation 2 verifies DAG paths through compute and mixed communication become global edges. |
| 3 | Evaluation 2 verifies mixed-cycle rejection and diagnostic provenance. |
| 4 | Evaluations 3 and 4 verify deterministic actor projections and DAG-valid host issue order. |
| 5 | Evaluations 3 and 7 verify stream FIFO projection and finalized global acyclicity. |
| 6 | Evaluations 3–5 verify issue-only relations create no completion wait and retain overlap. |
| 7 | Evaluations 4–6 compare executor, simulator, and profiling consumption of the finalized order. |
| 8 | Evaluations 7 and 9 cover the motivating PP4 pattern before and during GPU execution. |

### Implementation log

- Phase 1 approved.
- Phase 2 approved with one architectural adjustment: cross-stream host order
  is an explicit `issue` DAG edge rather than execution-plan-only metadata.
  `issue` constrains dispatch/arrival, never CUDA completion. The user further
  required the minimal transitive-reduced set of issue edges.
- The production lowering path now finalizes default-stream and ZeRO lifetime
  constraints before normalizing communication transactions. One global
  linear extension covers P2P, routed A2A/M-to-N/N-to-M, reduce/all-reduce,
  all-gather, and reduce-scatter nodes.
- Same-physical-stream projections use temporal FIFO edges. Cross-stream actor
  projections use an `issue` edge only between consecutive local
  transactions when no existing path already orders them. This omits every
  redundant issue edge while preserving the selected local total order.
- Execution plans expose issue predecessors separately from completion
  predecessors. Actor runtime adjacency ignores issue edges for CUDA waits;
  simulator replay lowers them from arrival to arrival. Diagnostics render
  them distinctly.
- Existing routed and directional-P2P validators remain as focused metadata
  checks after the unified pass; they no longer select production ordering.
- The node-atomicity assumption was retained: the fixed header/payload or
  metadata/payload calls inside one communication DAG node execute contiguously
  with one validated participant set. Hidden communication inside compute
  remains outside the static guarantee.
- Existing DAG equivalence digests changed because communication endpoints now
  persist a global index and some mixed schedules gain intended FIFO/issue
  edges. Review confirmed no nodes or data edges were removed. The PP4 AFD DAG
  lowers and replays successfully on CPU with synthetic aligned profiles.
- One broader control-plane workload test fails before the changed ordering
  path: `test_real_397b_meta_trace_bucketing_increases_finalized_dag_size`
  reaches A2A insertion with no runtime value mapping on
  `s0.seg0.bucket5 -> s0.seg1`. This is a pre-existing ZeRO-bucketing/A2A
  lowering issue and was not modified.
- A two-node PP4 GPU smoke was attempted against the discovered 16-GPU Ray
  cluster. The Ray GCS at `10.52.51.203:6379` became unreachable during
  `ray.init`; no actor was admitted and no GPU DAG execution occurred, so this
  first attempt neither validated nor invalidated the scheduling change. A
  fresh session at the same address was reachable outside the sandbox and the
  exact 12-rank regression subsequently completed.
- A fresh local Ray session on one 8xH100 node then completed the small AFD PP2
  and AFD PP4 parameter-parity workloads and the mixed P2P/collective
  DualPipeV parameter-parity workload. No dynamic scheduling or runtime
  dependency relaxation was required.

## Evaluation results

- `cpu-venv/bin/python -m pytest -q test/test_global_communication_ordering.py`
  — 11 passed.
- Planned focused suite (`test_global_communication_ordering`,
  `test_execution_plan`, `test_order_directive`, `test_afd_routed_directives`,
  `test_afd_full_lowering`, `test_interference_free_serialization`,
  `test_simulator`, `test_dag_builder_equivalence`, and `test_visualization`)
  — 179 passed.
- `test/test_executors.py test/test_actor_profile_timing.py
  test/test_profile_streams.py` — 57 passed.
- `test/test_control_plane_benchmark.py test/test_qwen_control_plane_workloads.py`
  — 7 passed, 1 unrelated pre-ordering failure described above.
- `py_compile` for every changed Python module/test and `git diff --check` —
  passed.
- GPU parameter parity, Qwen 9M, two optimizer steps: AFD PP2 (`afd_e2e_pp2`),
  AFD PP4 (`afd_e2e_pp4`), and DualPipeV
  (`pp4_dp2_ep2_v_placement`) — all exited successfully within two minutes.
  DualPipeV reported 51 parameters and maximum absolute error 0.00012207.
- Exact two-node, 12-rank PP4 AF-Pipe smoke (`afd_placement_pp4`, four
  microbatches, one warmup and one timed iteration) — passed in 79.4 seconds.
  Actor groups were `[0,4,8,10]` and `[3,7]` on `10.52.51.203`, with `[1,5]`
  and `[2,6,9,11]` on `10.52.54.102`. The timed step was 86.162 ms for 256
  tokens (2,971.13 tokens/s). Artifact: `out/20260820_131949/results.csv`.
- Black and Ruff were not available in `cpu-venv`; formatting was checked by
  compilation, focused tests, and diff whitespace validation.

## Remaining limitations

- Static deadlock freedom covers only explicit supported communication nodes
  and the documented blocking-transaction model. Hidden distributed work,
  mismatched backend calls, or transport failures remain outside the proof.
- A communication node with internally interleavable calls or changing
  participant sets must be split into explicit transactions; current nodes are
  valid only because their internal calls are contiguous and participant
  consistent.
- The deterministic global linear extension may alter which otherwise-ready
  communication is host-issued first. It adds no cross-stream GPU completion
  serialization, but the selected issue order can still affect performance.
- The global ordering guarantee retains the explicit-model limitations above;
  no known schedule-level deadlock remains in the tested 12-rank PP4 pattern.

## Commit message

```text
Build one global order for DAG communication transactions

Normalize P2P and collective endpoints into a shared transaction graph,
reject mandatory cycles with dependency provenance, and project one
deterministic order into same-stream FIFO and minimal cross-stream issue-only
edges. Keep executor waits completion-faithful and replay issue ordering at
endpoint arrival so independent streams retain overlap.
```
