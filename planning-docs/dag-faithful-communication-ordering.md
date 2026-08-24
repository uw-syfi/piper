# DAG-faithful communication ordering

## Phase 1: Scope

### Task summary

Make the finalized actor-local DAG authoritative for inter-node ordering in normal execution and normal simulation. Piper must honor every encoded dependency and must not add, relax, or bypass dependencies at runtime. Routed communication uses the stream declared by the schedule; pipeline P2P remains the explicit exception and continues to use its directional low-to-high and high-to-low communicators and physical streams.

### Objectives

- Preserve every data and temporal dependency in the finalized DAG through execution-plan lowering, CUDA execution, and simulator replay.
- Ensure all inter-node ordering required for correctness or deadlock freedom is constructed in the DAG and validated before execution.
- Use each DAG node's declared stream for `A2A_COMM`, `M_TO_N_COMM`, and `N_TO_M_COMM`, without direction- or kind-specific runtime stream overrides.
- Retain directional physical streams and communicators for pipeline `SEND_COMM` and `RECV_COMM`, while still enforcing every DAG dependency crossing or residing on those streams.
- Establish one deterministic, consistent issue order for all routed transactions sharing a schedule-declared stream across all participating actor groups, including within-group A2A and cross-group M-to-N/N-to-M.
- Preserve overlap between incomparable work. In particular, after `ATTN(mb0)`, its routed dispatch and `ATTN(mb1)` may be at the same topological level; communication-first dispatch priority may launch the routed operation first without creating a dependency between them.
- Make node completion truthful: a node's completion signal must cover all work semantically owned by that node, so successors never depend on implicit tensor waits, deferred communication drains, or unrelated device-wide synchronization.
- Keep normal simulator replay behaviorally aligned with the normal executor's DAG and stream semantics.

### Non-goals

- Replacing the directional pipeline P2P communicator or physical-stream design.
- Changing the meaning or syntax of schedule stream fields, placement, split, or order directives.
- Changing routed tensor annotations, routing layouts, collective algorithms, or model-side routing semantics except where required to make completion observable.
- Optimizing communication kernels or selecting a different routed transport solely for performance.
- Changing interference-free profiling, whose explicit serialization is a profiling counterfactual rather than normal DAG execution.
- Removing synchronization used only outside normal DAG execution, such as setup, teardown, model export, standalone communication benchmarks, or profiler capture boundaries.

### Correctness criteria

1. Every finalized DAG data or temporal edge is enforced as a completion dependency by both the executor and simulator; no node kind, communication direction, stream assignment, or dependency reason can relax it.
2. For two DAG nodes that are incomparable and execute on distinct physical streams, normal execution and normal simulation introduce no event wait, host wait, device synchronization, or synthetic dependency between them.
3. Any unavoidable same-physical-stream FIFO order is reflected in the finalized DAG before execution, rather than being invented independently by the executor or simulator.
4. Routed A2A, M-to-N, and N-to-M nodes execute on their DAG-declared stream in normal execution and both normal and interference-free profiling stream mappings.
5. Pipeline P2P nodes continue to map to the appropriate low-to-high or high-to-low physical stream and communicator, and every DAG dependency involving those nodes remains enforced across the mapped streams.
6. Every routed transaction has matching endpoint stream metadata, and all transactions sharing a routed stream have one acyclic, deterministic issue order consistent across participating actor groups. Invalid or cyclic orders are rejected before execution.
7. Collective issue-order edges connect only communication endpoints. Schedule order directives connect model-compute boundaries and do not make an outgoing routed communication a prerequisite of otherwise-ready next-microbatch compute.
8. When a communication node is reported complete, its transport work is complete for dependency purposes. Normal execution does not rely on implicit `AsyncCollectiveTensor` consumption, end-of-DAG routed-result drains, next-iteration send drains, or whole-device synchronization to satisfy communication dependencies.
9. For the motivating overlap, the finalized DAG contains `ATTN(mb0) -> M_TO_N(mb0)` and `ATTN(mb0) -> ATTN(mb1)`, but no path from `M_TO_N(mb0)` to `ATTN(mb1)` unless the schedule explicitly requires one. The dispatcher prioritizes the ready communication without converting that priority into a dependency.
10. Given identical DAGs, stream mappings, and node durations, normal simulator replay preserves the executor's dependency, FIFO, communication-rendezvous, and overlap relationships.

### Subtasks

- Make DAG construction own compute order, per-stream order, routed collective issue order, update prerequisites, and any other currently executor-synthesized inter-node constraints.
- Remove routed stream remapping and all dependency-relaxation paths while retaining directional P2P stream mapping.
- Make asynchronous routed and P2P completion observable through node completion without hidden cross-node or cross-iteration waits.
- Make simulator lowering consume the same finalized dependencies, physical stream mapping, and communication completion semantics as the executor.
- Update schedule validation and focused regression coverage for consistent routed ordering, full dependency enforcement, and the motivating overlap.

### Assumptions / open questions

- “No additional synchronization” applies to ordering between normal DAG nodes. Synchronization internal to a node is permitted only when necessary to implement that node and must complete before the node's completion signal; it may not constrain unrelated nodes.
- CUDA FIFO behavior is an unavoidable property of assigning multiple nodes to one physical stream. Piper will make that order explicit in the finalized DAG so the simulator and executor share one source of truth.
- Buffer retention without waiting is permitted for transport lifetime safety; any wait that can delay another DAG node is an ordering constraint and must be represented in the DAG or execution plan derived directly from it.
- Directional P2P mapping may place two DAG nodes that declare one logical stream onto different physical streams. DAG dependencies, rather than their former logical-stream co-location, remain authoritative after that mapping.

### Phase 1 amendment: collective-aware P2P ordering

#### Task summary

Replace the P2P-only ordering analysis with a global communication transaction
model that includes collective rendezvous when choosing the FIFO order of P2P
transactions on each directional physical stream. This supersedes the earlier
P2P-only graph boundary. The builder must select an order compatible with the
entire communication graph or reject the schedule; it must not use a
receive-last runtime heuristic.

#### Objectives

- Represent every DAG communication transaction relevant to rendezvous:
  pipeline and dense-state SEND/RECV pairs, routed A2A/M-to-N/N-to-M, and
  replica/ZeRO reduce, all-gather, and reduce-scatter collectives.
- Identify transactions from explicit `comm_id`, participant, endpoint, and
  actor-group metadata rather than tags or model-specific node names.
- Project all actor-local DAG reachability constraints onto one global
  communication graph. Paths may pass through compute or other communication
  nodes.
- Use the full graph to choose the order of P2P transactions independently for
  each directional physical P2P stream. Collective order and collective stream
  assignments remain constraints, not choices made by this pass.
- Materialize only the selected same-P2P-stream FIFO edges in actor DAGs. Do
  not add cross-stream completion edges between a P2P operation and a
  collective merely because one was considered first by the analysis.
- Reject malformed transaction metadata, collective endpoint/participant
  disagreement, cyclic input constraints, or a schedule for which no acyclic
  per-stream P2P ordering can be selected.
- Keep executor and simulator dependency-faithful; neither may relax edges or
  add a receive-late scheduling policy to compensate for an invalid graph.

#### Non-goals

- Reordering collectives, changing their process groups, or changing their
  schedule-declared streams.
- Serializing all NCCL operations on a rank or imposing a total order across
  independent communication streams.
- Adding model-, tag-, schedule-, DualPipeV-, AFD-, or expert-specific rules.
- Treating a deterministic tie-break as permission to override a graph
  dependency or collective rendezvous requirement.
- Changing P2P or collective transport algorithms as part of this amendment.

#### Correctness criteria

1. Every P2P and collective transaction has one unambiguous identity,
   participant set, and set of actor-local endpoints derived from explicit DAG
   metadata; malformed or inconsistent transactions are rejected.
2. Every communication precedence implied by an actor-local DAG path appears
   in the global transaction graph, including paths whose intermediate nodes
   contain compute, P2P, routed communication, or replica/ZeRO collectives.
3. The selected FIFO order for each directional P2P stream is consistent at
   paired endpoints and cannot create a cycle in the full transaction graph
   containing collective rendezvous.
4. A cycle that is visible only when a collective transaction connects its
   participant actors is rejected before execution with the participating
   communication IDs and actor groups in the diagnostic.
5. Collective order, stream assignment, and DAG dependencies are unchanged by
   P2P ordering; only otherwise-incomparable P2P endpoints on the same physical
   P2P stream receive new temporal edges.
6. Valid schedules remain deterministic without relying on lexicographic IDs,
   model tags, a receive-last heuristic, dependency relaxation, or runtime
   synchronization absent from the DAG.
7. Executor and simulator accept every valid finalized graph and observe the
   same P2P FIFO and collective rendezvous constraints.

#### Assumptions / open questions

- `comm_id` and `participants` already exist on the relevant communication
  nodes. The implementation plan must verify that their current identities are
  globally unique at the transaction granularity, especially for replicated
  ActorGroup DAGs and aligned P2P lanes.
- Routed A2A has peer-progress semantics in the simulator rather than one
  all-rank completion barrier. It still participates as a communication
  transaction for issue/FIFO cycle analysis; the implementation plan must
  preserve its existing completion semantics.
- This amendment chooses P2P FIFO edges using collective rendezvous
  constraints. It does not by itself create an issue-before relationship
  between a first P2P receive and a collective on another stream. If the
  motivating GPU stall requires such a cross-stream launch constraint after
  full-graph ordering, that is a distinct semantic extension and must be
  reviewed rather than encoded as a hidden completion edge.

## Phase 2: Implementation and evaluation plan

### Implementation plan

#### 1. Finalize all ordering in actor-local DAGs

- In `src/directives.py`, make order directives consistently connect compute-only region boundaries. Remove the cross-group special case that promotes routed and dense-state communication endpoints to model-order sinks/sources.
- In `src/ordering.py`, define one pure execution-stream-domain rule used during DAG finalization:
  - P2P `SEND_COMM`/`RECV_COMM` use their declared low-to-high, high-to-low, or dense-state directional domain.
  - Every other node, including A2A, M-to-N, and N-to-M, uses `node.stream`.
- Refactor per-stream ordering so the finalized DAG has explicit temporal paths between consecutive operations in each physical stream domain. Opposite P2P directions remain independent domains.
- Add a cross-actor communication-ordering pass in `src/piper.py` after actor-DAG splitting and before final per-stream ordering. For each routed schedule stream:
  1. Pair endpoints by `comm_id`, including within-group A2A and cross-group M-to-N/N-to-M.
  2. Validate matching endpoint streams and metadata.
  3. Project all existing actor-local reachability constraints onto a transaction graph.
  4. Reject cycles, then choose a deterministic common topological order using stable transaction identity only to break unconstrained ties.
  5. Add temporal edges only between consecutive communication endpoints in each actor's participating subsequence.
- Apply the same construction principle to directional P2P domains: preserve their physical stream split, construct consistent endpoint issue order where multiple transactions share a domain, and reject an inconsistent or cyclic schedule instead of relying on runtime relaxation.
- Keep existing update and gradient-synchronization prerequisites in the DAG. Verify they are complete after communication lowering; add missing explicit edges at construction time rather than synthesizing update waits later.
- Add a final validation that every consecutive operation assigned to one physical stream domain is already ordered by the DAG and every paired communication transaction has a consistent endpoint order.

#### 2. Make the execution plan a mechanical DAG lowering

- In `src/execution_plan.py`, remove the relaxable-dependency predicates and direction-specific routed stream mapping.
- Build `completion_predecessors` only from finalized DAG data and temporal edges. Remove independently synthesized stream FIFO, receive-consumer guards, and update dependencies; construction-time ordering and validation replace them.
- Preserve the P2P physical-domain mapping solely for determining whether an encoded dependency crosses physical streams. Routed locality is determined from `node.stream`.
- Keep same-physical-stream edges in the plan for simulator fidelity and diagnostics. The CUDA executor satisfies them through FIFO and must not insert an event wait; event waits are used only for encoded dependencies that cross physical streams.

#### 3. Preserve declared runtime streams

- In `src/runtime.py`, remove private M-to-N/N-to-M CUDA streams and their normal/profile overrides. `execution_stream_for_task` and `profile_stream_for_task` return the DAG-declared stream for routed nodes.
- Retain directional P2P stream and communicator selection unchanged.
- In `src/actor.py`, remove routed-direction stream initialization and profiling metadata while retaining directional P2P metadata.
- Retain the existing routed process-group topology for this change; communicator redesign is unnecessary to honor schedule-declared streams because the canonical DAG order covers collectives sharing one stream.

#### 4. Make communication-node completion truthful

- In `src/executors.py`, replace functional `all_to_all_single_autograd` payload operations with preallocated, stream-ordered c10d all-to-all operations. Piper already represents reverse transport with explicit backward communication nodes, so transport outputs need ordinary leaf-gradient boundaries rather than autograd through the collective itself.
- Launch each payload collective inside the selected node stream and record completion only after that stream-ordered operation. Eliminate `AsyncCollectiveTensor`-dependent consumption and `pending_routed_source_results`/end-of-DAG drains.
- Keep metadata waits inside the routed node only where the CPU must read dynamic split information. Source-side metadata work must remain stream-ordered and must not escape as hidden state that later iterations drain.
- Change dynamic dense-state P2P sends to the same stream-ordered send semantics as ordinary P2P, retain buffers only as long as required by the operation, and remove pending-send waits from the next iteration.
- Remove the device-wide synchronization from normal UPD execution. Its DAG predecessors and completion event must provide the required ordering.
- Preserve profiling-boundary synchronization and interference-free serialization as explicitly non-normal profiling behavior.

#### 5. Make simulation consume the same semantics

- In `src/simulator.py`, remove dependency-relaxation and nonblocking dense-send special cases.
- Build replay edges from every execution-plan predecessor without filtering. Use the shared physical stream-domain rule so routed nodes follow `node.stream` and P2P follows its directional stream.
- Preserve communication rendezvous modeling, but do not let rendezvous modeling add a dependency between otherwise incomparable DAG nodes.
- Keep the prologue as an explicit replay boundary; it is not a replacement for any DAG-node dependency.

#### 6. Dispatch policy and compatibility cleanup

- In `src/ordering.py`, include M-to-N and N-to-M in the existing communication-first priority for nodes at the same topological level. Priority changes launch order only and does not add an edge.
- Update comments, profiling stream metadata, validators, and tests that describe independent M-to-N/N-to-M streams or relaxable dependencies.
- Do not modify schedule JSON files. DAG digest fixtures may change because previously implicit or incorrect ordering becomes explicit; update a digest only after confirming the semantic DAG change is intended.

### Compatibility risks and constraints

- Schedules whose communication orders are cyclic or only runnable because the executor ignored dependencies will now fail during DAG construction. This is intentional.
- Replacing functional autograd collectives relies on Piper's explicit backward routed nodes. Construction/runtime validation must reject any routed boundary that lacks the matching backward transport rather than silently losing gradients.
- Canonical per-stream ordering may expose conflicts between previously independent M-to-N and N-to-M sequences. It must reject conflicts rather than choose a locally valid but globally inconsistent order.
- P2P retains its private directional physical streams even when its DAG node carries another logical stream. This remains the only stream override.
- Interference-free profiling intentionally adds serialization and therefore is not used to assert normal-execution overlap.
- If a focused E2E workload deadlocks, stop after collecting enough evidence to identify the schedule-level wait cycle. Do not add schedule changes, dependency relaxation, stream remapping, or runtime launch heuristics until the cause and intended solution have been reviewed with the user.

### Phase 2 amendment: global P2P transaction ordering

The DualPipeV deadlock requires replacing the P2P portion of the per-domain
communication tie-break. This amendment is intentionally limited to
`SEND_COMM`/`RECV_COMM` transactions, including ordinary pipeline and direct
dense-state P2P. A2A, M-to-N, N-to-M, and other collectives are not vertices
in this graph; their existing ordering remains separate. Actor-local paths
between P2P endpoints may traverse those operations, so their encoded DAG
dependencies still constrain P2P transaction order, but collective
rendezvous are not collapsed into this P2P-only graph.

Minimal implementation:

1. Add explicit `source_compute_uid` and `destination_compute_uid` metadata to
   every SEND/RECV pair. Ordinary pipeline P2P uses the compute endpoints it
   already lowers; dense-state forward uses state-owner/combine-consumer
   forward nodes, while dense-state backward uses the actual consumer/owner
   backward nodes. Validate that each anchor exists as local COMPUTE on the
   appropriate actor DAG. Do not infer anchors from successor list position.
2. Keep routed-collective ordering separate and run it first. Replace the P2P
   portion of `_order_communication_transactions` with one global graph whose
   vertex is a paired P2P `comm_id`. Add a hard edge `T1 -> T2` whenever an
   actor-local finalized dependency path connects the local endpoint of T1 to
   the local endpoint of T2, regardless of directional stream.
3. Topologically sort that global graph with Kahn's algorithm. Among currently
   ready transactions, use destination-compute topological index, then
   source-compute topological index, then `comm_id` as a deterministic
   priority. The compute positions are scheduling preferences only; they
   never override a hard graph dependency.
4. Project the resulting single transaction order onto each actor-local
   physical P2P domain (`pp_lo_hi`, `pp_hi_lo`,
   `dense_state_lo_hi`, `dense_state_hi_lo`) and add temporal edges only
   between consecutive local endpoints in that projection. No edge is added
   between different physical streams.
5. Explicitly reject a cyclic global P2P transaction graph and report the
   participating `comm_id`s and compute anchors. Rebuild and validate the same
   P2P-only transaction graph after final stream/ZeRO edges are materialized,
   so later DAG finalization cannot silently reintroduce a cycle.

This guarantees that every inserted P2P FIFO edge points forward in one global
transaction order. It does not claim to detect a cycle whose essential
rendezvous vertex is a non-P2P collective; the simulator remains the broader
final communication-graph validator.

Focused evaluation for the amendment:

- Add generalized synthetic regressions in which destination-compute order
  conflicts with lexicographic communication IDs: the earlier consumer's P2P
  transaction must lead the shared physical stream, paired endpoints must
  agree, and the finalized CPU simulator replay must be acyclic. Do not
  hard-code the DualPipeV schedule into the test.
- Add synthetic rejection tests for a cyclic global P2P transaction graph
  spanning opposite directional streams and for malformed/missing compute
  anchors.
- Add dense-state forward/backward tests proving their explicit anchors refer
  to the actual producing and consuming compute nodes.
- Rerun the existing focused communication/order/executor/simulator/DAG
  equivalence CPU files, followed by the 9M DualPipeV and focused AFD GPU
  parameter-parity cases. A GPU deadlock remains a stop-and-review condition.

The Phase 2 amendment was approved with the explicit requirement that its
regression tests remain schedule-general rather than encoding the exact
DualPipeV case. P2P analysis runs after actor-group DAG splitting, while
consuming all actor-group DAGs together; anchor metadata is created during
global communication lowering before the split.

Implementation update:

- Routed collective ordering remains its own per-declared-stream pass. The new
  global graph contains only paired DAG-level SEND/RECV transactions.
- Generic and dense-state lowering now records explicit source/destination
  compute anchors; dense-state backward anchors are the actual backward
  producer and consumer.
- Global P2P dependencies are collected across every actor and directional
  stream, topologically sorted with consumer/source compute-index priority,
  projected back onto actor-local physical streams, and validated again after
  final DAG edges.
- General synthetic tests cover compute-priority versus adversarial comm IDs,
  paired stream agreement, simulator acyclicity, explicit global-cycle
  rejection, missing anchors, and dense-state backward anchors. No test
  encodes the DualPipeV schedule.
- The focused CPU suite (`test_execution_plan`, `test_order_directive`,
  `test_afd_routed_directives`, `test_afd_full_lowering`,
  `test_profile_streams`, `test_executors`, `test_simulator`, and
  `test_dag_builder_equivalence`) passes: 174 tests. All nine schedule-matrix
  DAGs also complete CPU simulator replay with unit node durations.
- The post-change DualPipeV GPU run still times out, but not in communicator
  setup: a standalone two-rank reproduction successfully initializes and uses
  all four Piper P2P channels, and a PP2/EP2 1F1B correctness workload using
  the same routed A2A and P2P mechanisms passes on four GPUs.
- Per-rank dispatch diagnostics originally suggested interference between an
  early P2P receive and routed A2A. Later host-call timing disproved that
  hypothesis: the actor blocks inside synchronous `dist.recv` before it can
  issue the later A2A or P2P send. The stall therefore requires no
  cross-communicator NCCL progress interference.

### Phase 2 amendment: full communication graph for P2P FIFO selection

This amendment supersedes the P2P-only transaction graph in the preceding
implementation. It does not replace the already-approved routed-stream
canonicalization or dependency-faithful executor/simulator behavior.

#### Implementation plan

1. **Normalize communication transactions after ActorGroup splitting.** In
   `src/piper.py`, introduce one internal transaction record populated only
   from explicit DAG metadata. It will cover:
   - paired pipeline and dense-state `SEND_COMM`/`RECV_COMM`;
   - routed `A2A_COMM`, `M_TO_N_COMM`, and `N_TO_M_COMM`;
   - `REDUCE_COMM`/`ALL_REDUCE_COMM`, `ALL_GATHER_COMM`, and
     `REDUCE_SCATTER_COMM`.

   Each record contains a typed transaction identity, communication kind,
   participant ranks, actor-local endpoint UIDs, and physical stream domains.
   P2P retains its source/destination compute anchors and one directional
   stream. Collective records validate identical `comm_id`/participants across
   all logical endpoints and that endpoint ActorGroups exactly cover the
   declared participants. A same text `comm_id` describing incompatible kinds
   or participant sets is rejected rather than silently merged.

2. **Make all fixed constraints visible before choosing P2P FIFO.** Preserve
   the existing common routed-collective ordering first. Refactor
   `src/ordering.py` so physical FIFO can be materialized for all non-P2P
   domains while leaving the four directional P2P domains unresolved. Apply
   ZeRO lifetime/inter-chain constraints after that fixed stream order and
   before P2P selection. This prevents a later collective or ZeRO edge from
   invalidating an ordering chosen from an incomplete graph.

3. **Build one global communication transaction DAG.** For every actor-local
   DAG, use reachability between all local communication endpoints to add hard
   transaction precedences. Collapsing all endpoints of the same collective
   transaction into one vertex carries precedence across its participant
   ActorGroups; paths may contain arbitrary compute and communication nodes.
   Reject a cycle in these fixed constraints with a diagnostic listing each
   transaction's kind, `comm_id`, participants, local endpoints, and the cycle
   path.

4. **Select P2P FIFO from a full-graph linear extension.** Topologically sort
   the full communication DAG. Existing P2P source/destination compute
   positions remain deterministic preferences among ready vertices; stable
   typed transaction identity is only the final tie-break. Project that one
   linear extension onto the P2P transactions of each directional physical
   stream, then add temporal edges only between consecutive local P2P
   endpoints in the projection. Because every new edge points forward in the
   full linear extension, it cannot create a transaction cycle. No collective
   edge or cross-stream P2P/collective edge is added by this projection.

5. **Finalize and revalidate.** Run the normal final physical-stream pass,
   which should find every directional P2P domain already totally ordered.
   Rebuild the full communication graph from the finalized actor DAGs and
   reject any cycle or endpoint inconsistency. Retain the existing routed,
   dense-state, and paired-P2P sequence validators as focused schema/channel
   diagnostics rather than relying only on the global check.

6. **Keep simulator semantics unchanged except for shared validation
   coverage.** The simulator already forms rendezvous vertices for P2P,
   replica/ZeRO collectives, and cross-group M/N while preserving ragged-A2A
   peer-progress semantics. Reuse transaction identity/participant validation
   helpers where practical, but do not turn A2A into an all-rank completion
   barrier and do not add replay dependencies absent from the finalized DAG.

#### Compatibility risks and limitations

- Reordering finalization so ZeRO edges precede P2P selection must preserve the
  same allocation/free lifetimes. If `_add_inter_chain_temporal_edges` cannot
  run from non-P2P-finalized topology without changing its semantics, split
  it into constraint derivation and edge application rather than approximating
  or omitting its edges.
- ActorGroup DAG nodes represent SPMD execution on every device in the group.
  Transaction validation must reason about those represented ranks and must
  not incorrectly require one actor-local node per global rank.
- A2A remains peer-progress communication in replay even though it is a vertex
  in construction-time transaction ordering.
- The algorithm guarantees that chosen P2P FIFO edges do not create a cycle in
  the modeled DAG/rendezvous graph. As recorded in Phase 1, it does not invent
  a cross-stream issue-before edge for two otherwise-incomparable operations.
  The focused DualPipeV GPU run will determine whether the observed unmatched
  receive stall remains outside this model.
- Existing schedule JSON files are not modified. Digest expectations change
  only after reviewing the semantic edge difference.

#### Evaluation plan

1. **Transaction normalization and rejection:** extend generalized synthetic
   cases in `test/test_afd_routed_directives.py` (or a focused transaction
   ordering test file if separation is clearer) to cover all supported
   communication kinds, ActorGroup-expanded participants, duplicate/conflicting
   identities, missing participants, and mismatched collective endpoints.
2. **Collective-mediated ordering:** add a schedule-neutral three-ActorGroup
   graph where P2P-only reachability leaves two transactions incomparable but
   a cross-group collective carries precedence between them. Verify the
   selected same-stream P2P order follows that precedence and no collective or
   cross-stream edge is materialized.
3. **Collective-mediated cycle rejection:** add a generalized graph whose only
   cycle closes by collapsing collective endpoints across ActorGroups. Verify
   construction rejects it and reports transaction kinds, IDs, participants,
   and the cycle. Do not encode the exact DualPipeV schedule.
4. **Finalization ordering:** add focused ZeRO/all-gather/reduce-scatter cases
   proving their fixed edges are visible before P2P selection and their
   lifetime semantics are unchanged. Verify the final P2P domains are total
   and rebuilding the full graph is acyclic.
5. **Simulator fidelity:** run the generalized valid graph through CPU replay
   with unit profiles and verify rendezvous completion. Retain explicit tests
   that ragged A2A does not become an all-rank barrier.
6. **Focused CPU regression:** run
   `test_execution_plan.py`, `test_order_directive.py`,
   `test_afd_routed_directives.py`, `test_afd_full_lowering.py`,
   `test_simulator.py`, and `test_dag_builder_equivalence.py`. Add
   `test_executors.py` only if a shared validation interface requires runtime
   changes; no runtime change is otherwise planned.
7. **Focused GPU regression:** run the existing small PP2/EP2 1F1B correctness
   workload, the existing small AFD PP2 correctness workload, and the 9M
   DualPipeV workload. The first two guard ordinary collective/P2P behavior;
   DualPipeV tests whether the collective-aware FIFO model resolves the
   motivating execution. If it still stalls, stop with the finalized graph and
   dispatch evidence rather than adding a receive-last or runtime workaround.

Correctness-criterion mapping for this amendment:

| Criterion | Verification |
| --- | --- |
| 1 | Evaluation 1 validates identity, participants, represented ranks, and endpoints for every supported kind. |
| 2 | Evaluations 2 and 4 prove local paths and fixed finalization edges appear in the global graph. |
| 3 | Evaluations 2, 4, and 5 prove projected P2P FIFO is paired, acyclic, and replayable. |
| 4 | Evaluation 3 proves a collective-only bridge closes and diagnoses the otherwise-hidden cycle. |
| 5 | Evaluations 2 and 4 compare collective nodes/edges/streams before and after P2P selection. |
| 6 | Evaluations 2, 3, and 6 use schedule-neutral identities and deterministic non-lexicographic constraints. |
| 7 | Evaluations 5 and 7 cover finalized-graph replay and focused GPU execution. |

### Evaluation plan

#### CPU-only focused verification

1. **Final DAG and order-directive behavior** — extend `test/test_order_directive.py`, `test/test_afd_routed_directives.py`, and `test/test_afd_full_lowering.py` to verify:
   - model order connects compute boundaries only;
   - the motivating M-to-N and next-microbatch ATTN nodes share a topological level with no path between them;
   - routed endpoints on one declared stream receive a consistent communication-only order;
   - mixed A2A/M-to-N/N-to-M and multi-actor-pair orders are canonicalized;
   - stream mismatch and cyclic transaction constraints are rejected.
2. **Execution-plan fidelity** — update `test/test_execution_plan.py` so explicit data/temporal edges are never relaxable, routed nodes retain their declared stream, directional P2P domains remain distinct, and the plan has no predecessor absent from the finalized DAG.
3. **Runtime stream mapping and completion** — update `test/test_profile_streams.py` and focused cases in `test/test_executors.py` to verify routed normal/profile execution preserves the supplied stream, P2P still selects directional streams, mocked routed payload transport returns ordinary tensors, and no routed-result or pending-send drain remains.
4. **Executor dependency behavior** — use mocked streams/events in `test/test_execution_plan.py` or `test/test_executors.py` to prove every cross-stream DAG predecessor causes a wait, same-stream FIFO needs no redundant wait, and incomparable nodes receive no wait or host/device synchronization.
5. **Simulator fidelity** — update focused cases in `test/test_simulator.py` to verify every DAG edge is replayed, incomparable streams overlap, same-domain nodes follow explicit DAG order, directional P2P remains separated, dense sends use ordinary rendezvous completion, and opposite communication directions are never made safe by dropping an edge.
6. **General schedule regressions** — run `test/test_dag_builder_equivalence.py`, reviewing any digest changes, plus the focused files above. No schedule fixture will be changed to make a failing ordering test pass.

Planned CPU command set:

```bash
cpu-venv/bin/python -m pytest \
  test/test_execution_plan.py \
  test/test_order_directive.py \
  test/test_afd_routed_directives.py \
  test/test_afd_full_lowering.py \
  test/test_profile_streams.py \
  test/test_executors.py \
  test/test_simulator.py \
  test/test_dag_builder_equivalence.py
```

#### Focused GPU verification

1. Run the existing Qwen parameter-parity cases for `afd_pp2_single_node`, `afd_pp4_single_node`, and `dualpipev` from `test/test_qwen_e2e_correctness.py`. These cover routed stream execution, unequal routed/dense scaling, and bidirectional directional P2P without adding new schedule fixtures.
2. Run one small AFD PP2 profile-and-simulate workload in normal mode. Confirm it completes without deadlock, CUPTI attributes routed nodes to the schedule-declared stream, and normal replay consumes the same finalized dependency plan.
3. Generate one normal-mode profiler trace for the motivating AFD overlap and verify that M-to-N and next-microbatch ATTN are independently ready after the prior ATTN; observed overlap is permitted but not required because hardware contention may serialize execution.

The existing focused E2E selector will be used rather than the entire E2E suite:

```bash
gpu-venv/bin/python -m pytest test/test_qwen_e2e_correctness.py \
  -k 'afd_pp2_single_node or afd_pp4_single_node or dualpipev'
```

#### Correctness-criterion mapping

| Criteria | Verification |
| --- | --- |
| 1 | CPU steps 2, 4, and 5: explicit routed and P2P temporal/data edges are always enforced by executor and replay. |
| 2 | CPU steps 1, 4, and 5: forked incomparable work has no wait and overlaps in simulation. |
| 3 | CPU steps 1, 2, and 6: finalized DAGs contain physical-domain temporal paths and plans add no synthetic predecessors. |
| 4 | CPU step 3 and GPU steps 1–2: routed operations preserve the schedule-declared stream in runtime and profiling. |
| 5 | CPU steps 2–5 and the DualPipeV E2E case: directional P2P mapping remains while dependencies are fully enforced. |
| 6 | CPU step 1: endpoint mismatch, multi-group ordering, mixed routed kinds, and cycles are covered. |
| 7 | CPU step 1 and GPU step 3: only communication endpoints receive issue-order edges; model order no longer captures outgoing communication. |
| 8 | CPU steps 3–4 and AFD E2E cases: no implicit collective wait, end-of-DAG drain, next-step send drain, or update-wide synchronization is required. |
| 9 | CPU step 1 and GPU step 3: the motivating fork topology and communication-first dispatch are directly inspected. |
| 10 | CPU step 5 and GPU step 2: simulator and executor consume the same dependencies, stream domains, and rendezvous semantics. |

### Implementation log

- The collective-aware P2P-ordering Phase 2 amendment was approved with the
  explicit boundary that only same-directional-P2P FIFO edges may change;
  collectives are read-only constraints and newly invalid schedules may be
  rejected earlier.
- Phase 2 approved. Same-physical-stream plan edges are diagnostic/simulator dependencies satisfied by FIFO without an event wait; only cross-stream encoded dependencies receive event waits.
- E2E deadlocks are a stop-and-review condition: investigate the schedule-level cycle and obtain user confirmation before any schedule or runtime mitigation.
- Removed routed-direction stream overrides and dependency relaxation. Routed nodes now retain their declared stream; directional P2P remains the sole physical-stream override.
- The legacy non-default-stream anchor pass conflicted with the common cross-actor communication order. Replaced it with deterministic DAG-compatible FIFO materialization per physical stream domain; this makes unavoidable FIFO explicit without reversing an encoded dependency.
- Routed payloads now use preallocated stream-ordered c10d operations and Piper's explicit backward transport nodes. Functional async collective tensors, end-of-DAG routed drains, next-iteration sized-send drains, and normal-update device synchronization were removed.
- Digest review found that splitting an internal bucket backward into BI/BW could leave BW outside every update prerequisite. The split rewrite now connects every BWD_W to update; the focused bucket regression covers this invariant.
- DAG digest changes were reviewed rather than accepted mechanically: nodes are unchanged; former cross-direction logical-P2P edges are replaced by independent directional FIFO chains, and routed FIFO is explicit. The missing-update edge above was fixed before refreshing expectations.
- CPU checks completed so far: 40 order-directive tests, 39 routed-directive/full-AFD lowering tests, 85 execution-plan/profile-stream/executor/simulator tests, and 4 DAG-builder equivalence tests pass.
- Before the common communication-order pass, each actor now materializes its established default-compute constraints so transaction ordering sees the complete model schedule; final physical-stream FIFO materialization remains after the common pass.
- Added a three-ActorGroup regression for a communication wait cycle spanning three different actor pairs; construction rejects it rather than relying on executor behavior.
- GPU visibility differs by execution sandbox: the default filesystem sandbox hides `/dev/nvidia*`, while an approved host execution sees all eight H100s and `torch.cuda.is_available()` reports eight devices. GPU verification must therefore run outside the filesystem sandbox.
- The first DualPipeV GPU run reached DAG execution and failed because a backward RECV allocated its dynamic buffer from the corresponding forward's runtime output shape without an encoded dependency on that forward. The builder now records an explicit receive kind and forward-shape UID, adds `forward -> backward RECV`, and the executor consumes that metadata instead of choosing `data_succs[0]`. This keeps general dynamic-shape support and makes the prerequisite visible to execution and simulation.
- The user confirmed the dynamic-shape design: retain the corresponding forward's runtime shape metadata and its explicit `forward -> backward RECV` dependency rather than imposing static boundary shapes or sending a shape header.
- An earlier pre-amendment DualPipeV DAG produced a simulator-visible
  cross-direction communication cycle. The collective-aware finalized DAG no
  longer does: unit-duration CPU replay completes in `93.0` time units, while
  the GPU workload still stalls. That discrepancy is evidence that the
  simulator and transaction validator omit a host-dispatch constraint rather
  than evidence of cross-communicator GPU blocking.
- A second issue exists inside each domain: schedule `order` directives constrain compute boundaries only, while an inserted P2P receive has only `RECV -> destination compute`. The common pass compares reachability between communication endpoints, not the scheduled order of their producer/consumer compute nodes. Incomparable transactions are selected by lexicographic `comm_id`, and that stable implementation tie-break is then materialized as real FIFO edges. In DualPipeV this places PP0 backward MB0/1/2 receives before PP3 forward MB0 even though PP3 forward MB0 is the first scheduled consumer; a far-future receive can consequently block ready traffic on the same stream. This must be addressed as part of the agreed construction-time ordering design, not by runtime relaxation.
- The approved collective-aware amendment normalizes P2P, routed, replica,
  and ZeRO communication from explicit `comm_id`, participants, ActorGroup
  placement, endpoints, and stream metadata. One actor-DAG collective node
  represents every SPMD rank in that ActorGroup; endpoint ActorGroups must
  exactly cover the declared participant ranks.
- Fixed routed ordering, non-directional physical-stream FIFO, and ZeRO
  lifetime edges are now materialized before selecting directional P2P FIFO.
  The full transaction graph collapses collective endpoints across
  ActorGroups, but projects its linear extension only onto same-directional
  P2P endpoints. It adds no P2P-to-collective, collective-to-P2P, cross-stream,
  or collective-reordering edge.
- Generalized three-ActorGroup tests demonstrate both outcomes that the old
  P2P-only graph could not express: a collective-only bridge constrains a
  shared P2P FIFO, and collapsing that bridge exposes a global cycle. Schema
  tests cover local A2A, reduce/all-reduce, all-gather, reduce-scatter,
  participant coverage, and incompatible `comm_id` reuse.
- An initial PP2/EP2 parity invocation used zero numerical tolerance and
  reported expected BF16 accumulation-order differences. Re-running with the
  existing E2E contract (`rtol=0`, `atol=1e-7`, two dense BF16 ULPs, and
  `expert_atol=2e-4`) passed; no correctness threshold was changed.
- The collective-aware ordering did not resolve the focused DualPipeV GPU
  stall: the 9M run again reached actor execution and timed out at 120 seconds.
  This matches the documented boundary of the model: it prevents a P2P FIFO
  choice from creating a modeled communication-transaction cycle, but does
  not invent an issue-before dependency between otherwise-incomparable
  operations on different streams/communicators. No receive-last policy,
  dependency relaxation, schedule edit, or runtime synchronization was added.
- The current stall is a two-transaction P2P issue-order cycle. ActorGroup 0
  dispatches `recv.1` (`PP3/F/MB0`) before `send.0` (`PP0/F/MB0`), while
  ActorGroup 1 dispatches `recv.0` (`PP1/F/MB0`) before `send.1`
  (`PP2/F/MB0`). `recv.1` waits for `send.1`, whose actor is blocked in
  `recv.0`; `recv.0` waits for `send.0`, whose actor is blocked in `recv.1`.
- Piper calls synchronous `dist.send`/`dist.recv` from one sequential Python
  dispatch loop per actor. A two-rank NCCL probe delayed the matching send by
  three seconds and measured the receive call returning only after `3.362`
  seconds, confirming that the unmatched receive prevents that actor from
  issuing later DAG nodes even when those nodes use different CUDA streams or
  communicators.
- The global communication graph models DAG completion reachability and
  same-physical-stream FIFO, but not the actor-local host issue sequence across
  different directional P2P streams. Consequently, its acyclicity does not
  guarantee freedom from deadlocks caused by blocking P2P calls. The CPU
  simulator likewise allows later-stream work to be issued while a receive is
  pending and therefore does not reproduce this host-dispatch cycle.

### Evaluation results to date

- `cpu-venv/bin/python -m pytest -q test/test_execution_plan.py test/test_order_directive.py test/test_afd_routed_directives.py test/test_afd_full_lowering.py test/test_profile_streams.py test/test_executors.py test/test_simulator.py test/test_dag_builder_equivalence.py --maxfail=20`: **171 passed**, 14 pre-existing TorchScript deprecation warnings.
- `git diff --check`: **passed**.
- `python -m py_compile` for all changed source modules: **passed**.
- Initial focused DualPipeV GPU parameter-parity attempt: **failed** with missing `shape_ref` runtime state; resolved by the explicit forward-shape dependency above.
- Corrected focused DualPipeV GPU retry with a 120-second timeout: **timed out during actor execution**. The earlier pre-amendment CPU schedule exposed a communication cycle; the current collective-aware finalized schedule instead completes CPU replay in `93.0` unit-duration time units because host-blocking P2P issue order is not modeled.
- Collective-aware amendment CPU verification:
  `test_execution_plan.py`, `test_order_directive.py`,
  `test_afd_routed_directives.py`, `test_afd_full_lowering.py`,
  `test_simulator.py`, and `test_dag_builder_equivalence.py`: **143 passed**.
- Focused local-H100 PP2/EP2 1F1B parameter parity with the repository's
  standard BF16 tolerances: **passed**, 51 parameters, maximum absolute error
  `3.05176e-05`.
- Focused local-H100 AFD PP2 parameter parity with the same contract:
  **passed**, 51 parameters, maximum absolute error `4.76837e-07`.
- Focused local-H100 9M DualPipeV parameter-parity run: **timed out after 120
  seconds during actor execution**; per the approved stop condition, no
  mitigation was attempted.
- The task-local Ray session was stopped after verification, including actors
  left by the timed-out DualPipeV run.

### Remaining known limitations

- The current DualPipeV finalized schedule has a confirmed cross-direction
  blocking-P2P host issue cycle, despite having an acyclic DAG completion graph
  and matching per-direction endpoint sequences.
- Per the approved constraint, any GPU E2E deadlock will be diagnosed as a schedule-level wait cycle and reported before changing schedule construction or runtime launch behavior.

### Proposed commit message

`Make communication execution faithful to finalized DAG ordering`

## Phase 1 amendment: P2P-only nonblocking directional issue

### Task summary

Revert the collective-aware extension of the communication transaction model.
The ordering analysis will cover only paired pipeline and dense-state P2P
SEND/RECV operations and their directional physical streams/communicators.
Restore the intended independence of opposite P2P directions by making P2P
launch nonblocking with respect to the single Python dispatch thread, while
keeping completion dependencies faithful to the DAG.

### Objectives

- Remove A2A, M-to-N, N-to-M, reduce/all-reduce, all-gather, and
  reduce-scatter transactions from P2P normalization, ordering selection, and
  global P2P validation.
- Retain explicit pairing and validation of every P2P SEND/RECV transaction by
  `comm_id`, participants, endpoints, compute anchors, and directional stream.
- Preserve independent host issue/FIFO domains for low-to-high, high-to-low,
  dense-state low-to-high, and dense-state high-to-low P2P.
- Replace synchronous P2P calls with stream-ordered nonblocking launches so an
  unmatched operation in one direction cannot prevent the actor from issuing
  ready work in another direction.
- Keep P2P work and buffers alive until their stream-visible completion, and
  expose that completion through the existing DAG event dependencies rather
  than a hidden wait or end-of-step drain.
- Select and validate deterministic matching endpoint order independently per
  directional P2P domain, consistent with encoded DAG reachability.
- Reject malformed pairs or cyclic constraints within a directional P2P
  domain before execution.

### Non-goals

- Reordering, validating, or otherwise modeling collective transactions as
  part of P2P ordering selection.
- Changing routed, replica, or ZeRO collective streams, communicators, or
  rendezvous semantics.
- Serializing opposite P2P directions on one actor-wide host issue domain.
- Adding receive-last, schedule-specific, model-specific, or runtime deadlock
  heuristics.
- Changing schedule JSON files or directional P2P stream assignment.

### Correctness criteria

1. P2P ordering and global P2P validation inspect only `SEND_COMM` and
   `RECV_COMM`; adding, removing, or reordering a non-P2P collective cannot
   change the selected P2P order unless an encoded DAG path through it creates
   ordinary reachability between P2P endpoints.
2. Every P2P transaction has exactly one SEND and one RECV with matching
   `comm_id`, participants, compute anchors, and directional physical stream;
   malformed pairs are rejected.
3. An unmatched P2P operation returns control to the actor after being issued;
   it cannot prevent later ready nodes on another directional stream from
   being issued.
4. Every P2P node's completion event covers its transport operation, so DAG
   successors observe completed data without host waits, deferred drains, or
   implicit tensor synchronization.
5. Paired endpoints have one deterministic matching order within each
   directional P2P stream/communicator; invalid or cyclic per-direction orders
   are rejected with communication IDs and endpoints.
6. Opposite directional P2P operations receive no synthetic ordering edge,
   host wait, CUDA event wait, or shared FIFO solely because one actor dispatch
   thread issued them.
7. P2P ordering changes do not create collective edges, change collective
   order, or inspect collective transaction metadata.
8. The generalized bidirectional case that previously blocked in two opposing
   receives completes because both receives can be issued before either
   matching send becomes ready.

### Assumptions / open questions

- Piper's current `dist.send` and `dist.recv` calls are host-blocking until the
  peer enters the matching operation. A second two-GPU probe warmed the
  communicator first and still measured the delayed receive returning only
  after `3.219` seconds, ruling out first-use initialization as the cause.
- Separate CUDA streams/communicators provide the intended cross-direction
  independence only after operations are enqueued. The runtime must therefore
  use a nonblocking c10d P2P launch contract.
- The implementation plan must verify the lifetime and completion semantics of
  the selected nonblocking API. A returned Work handle may not be dropped while
  its buffers are live, and waiting on it from the dispatch thread would
  recreate the same deadlock.
- Ordinary pipeline receives already know their buffer shapes, so they can be
  issued nonblockingly. Dense-state sized receives currently read dynamic shape
  headers on the CPU before allocating payload buffers; fully nonblocking
  dense-state P2P would require a two-phase continuation or a stronger static
  shape contract and must be addressed explicitly in the implementation plan.

## Phase 2 amendment: P2P-only nonblocking directional execution

This amendment supersedes the previously approved collective-aware transaction
ordering implementation. It restores the P2P-only ordering model and fixes the
observed DualPipeV stall at its actual source: host-blocking pipeline P2P calls
on the actor's single dispatch thread.

### Implementation plan

1. **Restore P2P-only construction and validation.** In `src/piper.py`, replace
   the generalized communication-transaction collector and precedence graph
   with paired SEND/RECV transactions only. Validate each pair's `comm_id`,
   participants, endpoints, compute anchors, and directional physical domain.
   Select and project a deterministic linear extension independently for each
   directional P2P domain. Non-P2P collectives remain ordinary DAG nodes: an
   existing DAG path through one may establish reachability between two P2P
   endpoints, but collective metadata is not normalized, collapsed, or used as
   an additional communication model.

2. **Restore normal stream finalization.** Remove the partial-stream
   `excluded_domains` facility added for collective-aware P2P selection from
   `src/ordering.py`. Materialize routed communication order, directional P2P
   order, and final per-physical-stream FIFO through the normal builder flow.
   Do not add actor-wide host issue edges or cross-direction P2P edges.

3. **Make fixed-shape pipeline P2P host-nonblocking.** In
   `src/executors.py`, replace normal DAG `dist.send`/`dist.recv` calls with
   `dist.isend`/`dist.irecv` on the existing direction-specific process group
   and CUDA stream. Immediately invoke each returned `Work.wait()` while that
   CUDA stream is current. With Piper's NCCL configuration this enqueues the
   device-side completion dependency and returns control to Python; the node's
   existing completion event is then recorded after the dependency on the same
   stream. Preserve the deterministic matched P2P warm-up in `src/actor.py`,
   because the first async P2P operation can otherwise perform blocking lazy
   communicator initialization.

4. **Retain asynchronous state without hidden synchronization.** Introduce a
   small executor-internal launch record containing all Work handles and source
   or destination tensor references for one P2P node. After the DAG executor
   records that node's completion event, associate the record with that event.
   Reap records only when `event.query()` reports completion; never call a host
   wait, synchronize a stream/device, or drain outstanding P2P at a step
   boundary. Record source tensors on the launch stream so allocator reuse
   cannot race the send.

5. **Fail early for incompatible blocking-wait configuration.** Validate that
   the normal NCCL runtime is not configured with
   `TORCH_NCCL_BLOCKING_WAIT=1`, because that changes `Work.wait()` into a host
   wait and would recreate the single-threaded issue deadlock. This is a runtime
   compatibility requirement, not a scheduling heuristic.

6. **Keep simulator and collective execution unchanged.** The simulator
   continues to consume the finalized DAG and physical stream domains. Same-
   stream edges are satisfied by FIFO; encoded cross-stream dependencies use
   events. Routed, replica, and ZeRO collective launch/completion behavior is
   outside this amendment.

7. **Explicit dynamic-sized dense-state boundary.** Ordinary pipeline P2P,
   including dynamic backward receive shapes obtained from the corresponding
   forward's recorded runtime metadata, is covered because its receive buffer
   shape is known before launch. The separate `send_sized_tensors` /
   `recv_sized_tensors` dense-state envelope is not converted in this narrow
   amendment: a genuinely dynamic receive must obtain a shape header on the CPU
   before allocating its payload, and making that path host-nonblocking requires
   a DAG-visible two-phase receive or a resumable executor. It remains a known
   blocking limitation; no hidden thread, polling scheduler, static-shape
   restriction, or implicit drain will be introduced here. This is the sole
   intentional deviation from Phase 1 correctness criterion 3's otherwise
   general wording.

### Compatibility risks and assumptions

- The nonblocking completion mechanism relies on NCCL c10d's default
  `Work.wait()` behavior: when called with the operation's CUDA stream current,
  it inserts a stream dependency without waiting on the CPU. A two-rank H100
  probe after matched warm-up measured immediate Python return and an event
  recorded afterward becoming ready only when the delayed peer sent.
- Dropping Work or source-buffer references before the recorded completion
  event is unsafe; lifetime retention is part of correctness, not an optional
  optimization.
- The existing four direction-specific P2P process groups and stream mapping
  remain authoritative. This change does not infer topology from device
  ordinals or merge opposite directions.
- If a focused E2E workload still stalls, implementation stops for inspection
  of its finalized DAG and endpoint issue sequence. No receive-last policy,
  dependency relaxation, or schedule/runtime deadlock heuristic will be added
  without a new approved plan.

### Evaluation plan

#### Focused CPU tests

1. In `test/test_afd_routed_directives.py`, restore and extend generalized
   P2P-only cases: valid pairing, independent directional sequences,
   deterministic selection from compute-anchor reachability, malformed
   endpoint metadata, and rejection of a cyclic per-direction transaction
   graph. Verify that inserting or changing unrelated collective nodes does not
   change a P2P order except through ordinary DAG reachability.
2. In `test/test_executors.py`, use mocked streams, Work objects, and events to
   verify that fixed-shape sends/receives call `isend`/`irecv` on the correct
   process group; call `Work.wait()` with the declared stream current; record
   completion afterward; retain all Works and tensors until `event.query()`;
   reap without a host wait; and protect source tensors with `record_stream`.
   Cover multiple tensors in one communication node.
3. In `test/test_execution_plan.py` and `test/test_profile_streams.py`, confirm
   that opposite P2P directions retain distinct physical streams, receive no
   synthetic cross-direction dependency, and that all encoded same- and cross-
   stream dependencies remain present in the plan.
4. In `test/test_simulator.py`, confirm that independent opposite-direction
   P2P can make progress while same-direction FIFO and explicit DAG edges remain
   enforced.
5. Run the smallest combined regression set for the changed construction and
   runtime paths:

   ```bash
   cpu-venv/bin/python -m pytest -q \
     test/test_afd_routed_directives.py \
     test/test_execution_plan.py \
     test/test_profile_streams.py \
     test/test_executors.py \
     test/test_simulator.py \
     test/test_dag_builder_equivalence.py
   ```

#### Focused GPU verification

1. Run a two-rank delayed-peer probe after Piper's matched warm-up and verify:
   Python returns from `isend`/`irecv` plus stream-local `Work.wait()` before
   the peer completes; the completion event does not report ready until the
   transfer completes; and retained buffers remain valid.
2. Run the existing small Qwen PP2/EP2 1F1B and AFD PP2 parameter-parity cases
   with their established BF16 tolerances to detect regressions in ordinary
   pipeline and routed execution.
3. Run the existing small DualPipeV parameter-parity case with a two-minute
   timeout. It must pass without changing the schedule or relaxing a DAG edge.
   On a stall, capture the actor-local P2P endpoint issue sequence and stop for
   review.

The full unit and E2E suites are intentionally excluded; these focused tests
cover the modified builder, plan, executor, simulator, and motivating
bidirectional workload.

#### Correctness-criterion mapping

| Criterion | Verification |
| --- | --- |
| 1, 7 | CPU tests 1 and 5 verify a P2P-only model and collective independence. |
| 2, 5 | CPU test 1 verifies complete pairing, directional ordering, diagnostics, and cycle rejection. |
| 3, 6, 8 | CPU tests 2–4 and GPU tests 1 and 3 verify nonblocking cross-direction progress without synthetic ordering. |
| 4 | CPU test 2 and GPU test 1 verify stream-visible completion and lifetime without hidden synchronization. |

### Planned implementation log entries

- Record the removal of collective normalization and any resulting DAG digest
  changes only after confirming node sets and intended P2P dependencies.
- Record the dynamic-sized dense-state limitation above and any runtime
  configuration incompatibility discovered during GPU verification.
- Record every focused command actually run, its result, and any environmental
  limitation; do not generalize focused results to the full suite.

### Phase 3 implementation log

- Removed collective transaction normalization/collapse from P2P ordering and
  validation. P2P precedence and cycle detection now operate independently per
  directional physical domain; ordinary DAG reachability, including a path
  through a collective, remains visible without interpreting collective
  metadata.
- Restored one normal physical-stream finalization pass and removed the
  temporary `excluded_domains` interface. Opposite directional P2P domains
  receive no synthetic ordering edges.
- Normal fixed-shape pipeline P2P now uses `isend`/`irecv`, followed by
  stream-local `Work.wait()`. Work handles and tensors are retained behind the
  node's DAG completion event and reaped only with nonblocking `event.query()`;
  there is no step-boundary drain or host synchronization.
- Added an early incompatibility error for `TORCH_NCCL_BLOCKING_WAIT=1` and
  retained deterministic matched communicator warm-up. Dynamic sized
  dense-state envelopes remain on their existing synchronous header/payload
  path, as explicitly approved in Phase 2.
- Focused construction/executor tests:
  `test_afd_routed_directives.py test_executors.py`: **77 passed**.
- Planned focused CPU suite:
  `test_afd_routed_directives.py test_execution_plan.py
  test_profile_streams.py test_executors.py test_simulator.py
  test_dag_builder_equivalence.py`: **136 passed**.
- Focused 9M DualPipeV parameter parity: **timed out after 120 seconds**.
  Per the approved stop condition, no ordering heuristic, dependency
  relaxation, or collective/runtime workaround was added.
- Temporary node-level diagnostics (removed after capture) showed that all four
  initial `irecv` calls and their stream-local `Work.wait()` calls returned to
  Python immediately. ActorGroup 0 then stalled in `a2a.0` after independently
  posting `recv.1`; ActorGroup 1 stalled in `s1.seg3` after posting its
  data-predecessor `recv.0`. No matching SEND had yet been issued.
- The finalized synthetic DualPipeV DAG confirms the relevant local issue
  order. ActorGroup 0 begins `s0.seg0`, `recv.1`, `a2a.0`, while `send.0`
  follows the remainder of PP0 forward compute. ActorGroup 1 begins `recv.0`
  and PP1/PP2 forward compute, while `send.1` follows PP2 forward compute.
  `recv.1` and `a2a.0` have no DAG edge, so the observed A2A stall is not an
  executor-inserted dependency or a failure of `irecv` to return.
- A controlled four-rank probe subsequently refuted the broad hypothesis that
  an unmatched receive necessarily blocks a collective on another
  stream/communicator: the independent collective completed before either
  delayed matching send. A second probe showed default-stream compute also
  completes while an unmatched receive is pending.
- Fine-grained diagnostics localized the real workload's first blocking point
  to the CPU scalar read in A2A route-count validation, before any A2A
  collective call. The route counts were produced by `s0.seg0` but had not
  completed when the old dispatcher posted `recv.1` and then tried to consume
  them.
- The root scheduling defect was the fixed-level serializer. It chose all
  level-0 nodes before any level-1 node, so its communication priority was
  considered only among nodes assigned the same static level. After issuing
  `s0.seg0`, newly-ready `a2a.0` could not overtake unrelated level-0
  `recv.1` even though the two nodes had no DAG path between them.
- Per the user's explicit approval bypass for the final debug pass, execution
  dispatch now uses a priority-aware Kahn ready queue. It reconsiders priority
  after every issued node while preserving every DAG edge. DAG construction,
  same-stream FIFO edges, directional P2P streams, and communicator selection
  are unchanged. Training-DAG order diagnostics now print this actual runtime
  issue order.

### Phase 3 evaluation status

- `cpu-venv/bin/python -m pytest -q test/test_afd_routed_directives.py
  test/test_execution_plan.py test/test_profile_streams.py
  test/test_executors.py test/test_simulator.py
  test/test_dag_builder_equivalence.py --maxfail=20`: **136 passed**.
- `gpu-venv/bin/python -m pytest -q test/test_qwen_e2e_correctness.py -k
  "9M and (zero2_1f1b or afd_pp2_single_node or dualpipev)" --maxfail=1 -s`:
  **stopped on DualPipeV timeout**; later selected cases were not run because
  the plan required stopping on this failure.
- One-step diagnostic DualPipeV reruns at 75 and 60 seconds reproduced the
  stall and established the node/launch sequence above. Diagnostic logging was
  removed afterward.
- Post-fix focused CPU verification (`test_execution_plan.py`,
  `test_visualization.py`, `test_afd_routed_directives.py`,
  `test_profile_streams.py`, `test_executors.py`, `test_simulator.py`, and
  `test_dag_builder_equivalence.py`): **143 passed**.
- Post-fix 9M DualPipeV parameter-parity run on the local H100 node: **passed**
  in `59.93` seconds with the original schedule and established correctness
  tolerances.
- Post-fix 9M AFD PP2 parameter-parity regression on the same node: **passed**
  in `52.52` seconds.

### Remaining known limitations

- Dynamic-sized dense-state P2P still uses its existing synchronous
  shape-header/payload envelope; making that path host-nonblocking requires a
  DAG-visible two-phase receive or resumable execution.
- Ready-node priority affects only otherwise-valid linear extensions of the
  DAG. It does not prove that an arbitrary schedule's explicit dependency and
  same-stream FIFO graph is communication-deadlock-free; cyclic P2P transaction
  constraints continue to be rejected by construction-time validation.

### Updated commit message

`Make DAG dispatch dependency-faithful and communication-aware`

## Static late-receive investigation

- The initial proposal to add every other consumer predecessor as a receive
  predecessor is withdrawn. Those temporal edges would impose completion
  serialization where only a static host issue placement is requested.
- Current P2P ordering uses source/destination compute positions only to break
  ties between otherwise-unconstrained P2P transactions on the same
  directional domain. It then adds FIFO edges only between P2P endpoints; it
  does not anchor a receive to its destination compute or assign the receive a
  scheduling level.
- In the motivating DAG, `recv.1` is the first `pp_hi_lo` endpoint. It has no
  incoming data or temporal edge, so its derived topological level is zero.
  Its only data edge is outgoing to `s3.seg9`; the separate scheduled compute
  path into `s3.seg9` does not constrain `recv.1`.
- A revised solution must distinguish a reproducible static host dispatch
  position from a DAG completion dependency. It may assign or persist a
  receive dispatch slot immediately before its compute consumer only when the
  resulting total order respects all existing data, temporal, and directional
  FIFO edges; it must not encode the placement as an extra completion edge.

## Dispatch-level amendment

### Scope and correctness criteria

- Persist one integer `dispatch_level` per finalized actor-DAG node. Non-P2P
  nodes retain their graph topological level.
- Move each SEND earlier toward `source_compute + 1`, stopping at the greatest
  dispatch level required by any encoded predecessor. Move each RECV later
  toward `destination_compute - 1`, stopping at the least dispatch level
  allowed by any encoded successor.
- Do not add, remove, or relax a data or temporal edge. Validate that every
  encoded edge points from a strictly smaller to a strictly larger dispatch
  level.
- Use one static order in executor, simulator, profiling, sequence validation,
  control-plane replay, and diagnostics. Within a dispatch level order SEND,
  then every other non-RECV communication node, then compute/other nodes, then
  RECV.
- Remove priority-ready-queue execution. Preserve communicator transaction
  ordering and physical-stream FIFO established during DAG construction.
- The DualPipeV regression must complete without changing its schedule; the
  focused AFD routed regression must retain parameter parity.

### Implementation and evaluation plan

1. Add dispatch-level assignment and static ordering helpers in
   `src/ordering.py`. Assign only after all actor-local dependency, P2P FIFO,
   routed FIFO, and ZeRO edges are finalized.
2. Persist levels in node metadata and make runtime execution, execution-plan
   defaults, simulation, relevant sequence validators, control-plane replay,
   and DAG order diagnostics consume the static order.
3. Replace ready-queue tests with generalized tests for SEND movement, RECV
   movement, predecessor/successor clamping, edge validation, tie priority,
   and deterministic replay order.
4. Run the focused ordering/executor/simulator/DAG-builder CPU suite, then the
   existing 9M DualPipeV and AFD PP2 GPU parameter-parity cases.

### Assumptions and limitations

- `source_compute_uid` and `destination_compute_uid` are the explicit anchors
  for pipeline and dense-state P2P. Missing/non-local/non-compute anchors are
  construction errors in finalized actor DAGs.
- Dispatch levels encode static host issue placement, not completion
  dependencies. Graph topological levels remain available for dependency
  depth and cycle diagnostics.
- The heuristic improves endpoint locality while respecting the already
  selected communicator order. It does not claim globally optimal slack or
  prove deadlock freedom beyond the existing transaction and replay checks.

### Dispatch-level implementation log and evaluation

- Implemented a two-pass monotone heuristic. SEND levels move only earlier in
  graph-topological order and are clamped by finalized predecessor levels;
  RECV levels move only later in reverse graph-topological order and are
  clamped by finalized successor levels. A final strict edge-level check and
  total-order check reject inconsistent persisted metadata.
- The priority-ready-queue order was removed. Actor execution, execution-plan
  defaults, simulator replay, profile execution, communication-sequence
  validation, control-plane replay, and DAG diagnostics now consume the same
  static dispatch order. Diagnostics display both dependency topo level and
  dispatch level.
- DAG equivalence hashes changed because every finalized node now persists
  `dispatch_level`; reviewed node/edge construction is otherwise unchanged.
- The model-free control-plane harness needed a no-op P2P lifetime reap hook to
  match the current real executor interface; no runtime behavior changed.
- Focused CPU evaluation: **194 passed** across execution-plan,
  visualization, routed/full AFD lowering, order directives, profile streams,
  executors, simulator, DAG-builder equivalence, and control-plane tests.
- Focused 9M DualPipeV parameter parity with the unchanged schedule: **passed**
  in `63.28` seconds. In ActorGroup 0, `recv.1` moves from dependency topo 0
  to dispatch level 14, immediately before its level-15 PP3 consumer.
- Focused 9M AFD PP2 parameter parity: **passed** in `53.18` seconds.
- Additional deadlock regression selected 9M interleaved 1F1B, AFD PP2, and
  AFD PP4 sequentially. All three workloads completed within their two-minute
  per-case timeout, confirming no scheduling deadlock. Interleaved 1F1B and
  AFD PP2 passed parameter parity. AFD PP4 completed but failed numerical
  parity for two dense tensors: maximum reported mismatches were `28.75`
  reference ULPs / `2.14204e-7` absolute for `tok_embeddings.weight` and `3`
  reference ULPs / `1.78814e-7` absolute for
  `layers.0.attention_norm.weight`, versus the existing two-ULP and `1e-7`
  dense tolerances. This is not recorded as a correctness pass.
- `py_compile` for changed source modules and `git diff --check`: **passed**.

### Dispatch-level commit message

`Schedule P2P endpoints with static anchor-aware dispatch levels`

### Dense-state P2P follow-up

- The PP4/world12 AF-Pipe regression exposed a host-dispatch cycle at dense
  state sends because that path still used blocking `dist.send`/`dist.recv`
  while ordinary pipeline P2P used `isend`/`irecv` launch objects.
- Dense-state shape headers and payloads now use the ordinary nonblocking P2P
  launch/lifetime path in both directions. Dynamic receives still perform the
  required host-visible shape read before allocating each payload; the launch
  retains headers, payloads, and work handles until the node completion event.
- Focused executor evaluation: **34 passed** in `test/test_executors.py`;
  `py_compile` and `git diff --check` passed for the changed files.
- A warmed two-rank NCCL probe confirmed two unmatched header/payload sends
  return immediately from `isend` and `Work.wait`; without communicator warmup,
  the first `isend` blocks during lazy P2P initialization, validating Piper's
  existing eager directional-P2P warmup requirement.
- Before envelope batching, the unchanged PP4/world12 8K regression still
  stalled. NCCL traces showed all
  AG0 lanes enqueue MB1 dense-state `send.2` after the MB0 M-to-N transaction,
  AG2 has completed only MB0 `recv.0`, and AG1 reaches the next M-to-N metadata
  exchange. The unmatched MB1 dense-state transfer remains resident while the
  prior routed transaction cannot finish. Avoiding that cross-communicator
  schedule-level stall requires a broader static dispatch/order or transport
  decision and was intentionally not hidden in the narrow executor fix.
- A call-boundary trace isolated the remaining host block: after posting the
  first payload of MB1 `send.2`, the sender synchronously constructed the next
  CUDA shape header on the blocked dense-state stream. Dense-state transport
  now sends one combined shape header and posts all heterogeneous payloads in
  one `batch_isend_irecv` call, leaving the model envelope and DAG unchanged.
  The combined header is staged from pinned host memory so subsequent headers
  do not require a synchronous CUDA value construction on that stream.
- The payload-shape descriptor is produced on the caller stream and consumed
  from the dense-state receive stream. Explicit CUDA events now order those
  cross-stream uses; this fixed a transient `[0, 0]` descriptor read without a
  device-wide synchronization.
- On the unchanged PP4/world12 8K regression, all AG0 ranks advanced through
  the final `send.31`, and all AG2 ranks completed the first backward
  `recv.25`. The original `send.2` host-dispatch block is therefore fixed.
  The step still timed out after two minutes: AG2 next waits to execute its
  first backward compute behind routed `n_to_m.51`. This is a later posted-GPU
  communication/dependency cycle and is not addressed by the narrow envelope
  batching change.
