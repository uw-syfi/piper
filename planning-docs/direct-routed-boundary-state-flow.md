# Direct routed-boundary state flow

## Phase 1: Scope

### Task summary

Replace Piper's routed-state-owner relay with direct lowering of each original
model boundary. A replicated-to-sharded boundary has one M-to-N communication;
a sharded-to-replicated boundary has one N-to-M communication terminating at
the actual downstream group. Runtime model values that bypass the routed region
remain on their original dense-to-dense FX dataflow edge: they are transferred
directly between disjoint dense groups or retained locally when those groups are
the same. Piper-owned routing layout state follows the same direct path without
appearing in the model interface.

This revises the earlier routed contract that permitted bypass values but did
not transport them: a routed marker selects the tensor slot that receives
ragged routing semantics; it does not make every other cross-region model value
illegal or require model-specific annotations for passthrough state.

### Objectives

- Lower every original compute-to-compute boundary to at most one logical
  communication kind, selected from its endpoint placements and routing role.
- Lower `replicate(M) -> shard(N)` directly as M-to-N and
  `shard(N) -> replicate(K)` directly as N-to-M, with the latter terminating at
  the actual downstream replicated ActorGroup.
- Preserve every runtime model value bypassing a routed region on a generic
  dense-to-dense path inferred from FX identity, without sending it through the
  routed workers.
- Infer passthrough model state from graph dataflow and segment interfaces;
  require no knowledge of residual connections, experts, Qwen, tag names, or
  particular operators.
- Keep route counts, participant mappings, split vectors, row-layout identity,
  and transaction state private to Piper. Transfer the required state directly
  between equal-width dense groups when they are disjoint, and retain it locally
  when they are the same group.
- Preserve forward and backward/autograd semantics for routed payloads and all
  passthrough model values.
- Retain existing within-group ragged-A2A behavior and ordinary non-routed P2P
  behavior.

### Non-goals

- Adding model annotations for residuals or other named passthrough values.
- Inspecting model operators, expert kernels, parameter names, or annotation
  tag spelling to determine communication behavior.
- Changing placement, replicate, shard, split, or order directive syntax.
- Supporting multiple independently routed payload/count/combine marker
  triplets on one routed boundary.
- Changing expert ownership, route partitioning, true data parallelism, or
  parameter/gradient sharding semantics.
- Optimizing or compressing passthrough traffic beyond what is required for a
  clean and correct generic implementation.

### Correctness criteria

1. For a graph fragment `A0 -> F0 -> A1`, where A0 and A1 are replicated on
   equal-width groups disjoint from sharded F0, the lowered forward DAG contains
   exactly an M-to-N routed boundary from A0 to F0, an N-to-M routed boundary
   from F0 to A1, and one logical dense-state transfer directly from A0 to A1.
   The direct transfer may overlap the routed path. It has no routed-state-owner
   N-to-M, relay through F0, or temporal surrogate chain. If A0 and A1 use the
   same ActorGroup, dense state is retained locally and no dense-state
   communication is inserted.
2. Communication classification depends only on the original graph boundary,
   marker role, and exact endpoint placements. Nodes are not redirected to an
   earlier state owner and do not inherit an unrelated region's identity.
3. Every tensor-valued runtime model input that bypasses the routed region is
   available with the same logical value, shape, dtype, and ordering at the
   downstream dense region, without becoming an input or output of routed
   compute.
4. Values requiring gradients preserve the eager graph's gradient flow across
   the routed path and the direct dense-state path; bypass gradients are neither
   lost, duplicated, nor accumulated through an unintended route.
5. The marked dispatch payload alone receives variable-row routing semantics.
   Other model-interface values are transported generically according to their
   captured boundary schemas and are not mistaken for expert rows.
6. Piper transfers or reconstructs all backend-owned layout information needed
   by the direct N-to-M operation without exposing it to model code or relying
   on model-visible passthrough values.
7. Forward and backward routed transactions remain pairable and deadlock-free
   when the dispatch union is `A0 union F0` and the combine union is
   `F0 union A1`.
8. Within-group ragged A2A and non-routed equal-width pipeline boundaries retain
   their existing observable behavior.
9. Invalid routed interfaces or topologies are rejected before execution with
   an error describing the unsupported boundary or missing model/transport
   state.

### Subtasks

- Retain FX identities and schemas for values that bypass the routed region,
  without adding them to the routed model interface.
- Lower each routed boundary directly to its actual endpoint ActorGroups and
  remove state-owner relay construction.
- Hand Piper-owned dispatch layout directly from each dispatch dense lane to
  its equal-ordinal combine dense lane.
- Generalize forward/backward buffer handling for routed and dense-state slots.
- Update contract documentation and focused verification for DAG structure,
  runtime values, gradients, and communicator sequencing.

### Assumptions / open questions

- The initial design continues to use one marked dispatch payload, one count
  descriptor, and one marked combine payload per routed region. Additional
  lifecycle values are generic bypass model state, not additional routed
  tensors or routed-region interface slots.
- Dispatch and combine dense groups must have equal widths. Piper will reject
  an unequal-width lifecycle during schedule lowering, and canonical
  device-list order defines the one-to-one dispatch-lane/combine-lane mapping.
- Tensor passthroughs may have unrelated shapes and dtypes. They use the
  structured dense-state transfer specified in Phase 2, without receiving
  routed-row splits or being assigned to routed-worker carriers.
- Runtime symbolic/scalar values should be reconstructed when derivable from
  transported tensors or static schemas; any genuinely dynamic non-tensor
  boundary value must either be transported generically or rejected explicitly.
- F0 does not carry values it does not consume. Piper preserves their original
  A0-to-A1 FX identity without requiring artificial identity operations in the
  routed model GraphModule.

### Pre-implementation DisagMoE research

- The AF-Pipe paper says that M2N/N2M transfers hidden states between A and F
  stages and that N2M fuses conventional combine with inter-stage P2P, but it
  does not describe residual/bypass tensors, routing metadata, or an A-to-A side
  channel. The released arXiv source contains no additional contract for them.
- The paper and arXiv entry do not link a public implementation of the stated
  Megatron-LM training system. The public `USC-NSL/DisagMoE` repository is a
  separate vLLM-based inference project: its placement graph connects
  `Attention_i -> Expert_i -> Attention_{i+1}`, has no direct attention-to-
  attention edge, and its attention executor does not return the residual value
  produced by its fused layer norm. It cannot validate AF-Pipe training residual
  semantics.
- Piper will therefore preserve explicit model dataflow: routed values use
  D-to-R-to-C M2N/N2M, while bypass values and Piper-owned combine layout state
  use a direct D-to-C path. This is a Piper design choice, not a claimed
  reproduction of undocumented DisagMoE code.

## Phase 2: Implementation and evaluation plan

### Implementation plan

#### 1. Describe bypass values from FX identity

Update `src/fx.py` and `src/dag.py` so each segment boundary retains stable
identities and schemas for its ordered runtime values. For one routed lifecycle:

- marker metadata identifies the dispatch payload, route-count, and combine
  payload slots;
- bypass slots are original FX values produced in D and consumed in C without
  being consumed by R;
- R consumes only its actual routed inputs and produces only its actual routed
  outputs; bypass values are not added to R's interface; and
- lifted parameters, buffers, rematerialized shape expressions, and static
  constants remain outside this runtime-value mapping.

Store an explicit D-to-C bypass slot map and tensor schema rather
than inferring identity from placeholder names, slot prefixes, Qwen structure,
or residual conventions. Keep genuinely dynamic non-tensor values unsupported
unless the existing splitter can reconstruct them; reject them with a boundary
contract error.

Revise the public annotation documentation in `src/piper.py` and `README.md`:
the markers select the one tensor with routed-row semantics and its counts;
other graph values may bypass the routed region automatically, but model authors
remain responsible for the documented packed-row identity of the marked
payload.

#### 2. Lower each original boundary directly

Refactor cross-group lowering in `src/directives.py` around one routed
lifecycle topology:

```text
dispatch dense group D -> routed group R -> combine dense group C
```

Validate before edge replacement that D and C have equal widths, that each
dense/routed pair is identical or disjoint as required by the existing routing
mode, and that ordered dense lanes pair by canonical local ordinal. Record D,
R, C, their lane map, routed slot indices, bypass slot map, and schemas on
all lifecycle communication nodes.

Replace the original forward edges directly:

- `D -> R` becomes one paired-endpoint `M_TO_N_COMM` transaction;
- `R -> C` becomes one paired-endpoint `N_TO_M_COMM` transaction; and
- a real FX bypass edge `D -> C`, together with Piper-owned combine layout
  state, becomes one logical dense-state transaction when D and C are disjoint,
  or a local retained-state dependency when D and C are identical; and
- backward edges become the exact reverse transactions.

Do not redirect the routed N-to-M destination from C back to D, route bypass
state through R, or serialize the direct dense-state transfer after the routed
path. Remove the old state-owner relay builder and its specialized sequence
validator; retain generic P2P validation for genuinely non-routed equal-width
pipeline boundaries. Update routed lifecycle validation in `src/piper.py` to
check the D/R/C topology, forward/backward reversal, slot schemas, endpoint
pairs, and per-channel call order.

#### 3. Carry a generic structured dense-state payload directly from D to C

Generalize buffer handling in `src/executors.py`. Routed boundary payloads
contain only the marked routed tensor, with route-count-derived variable splits,
and the marked count descriptor on dispatch.

The separate dense-state payload is an ordered internal envelope containing the
FX-derived bypass tensor slots plus Piper-owned routing layout metadata. Pair D
and C lanes by canonical local ordinal. If D and C are disjoint, transfer each
lane's envelope directly with paired P2P operations represented by one logical
D-to-C DAG transaction; launch it after D compute independently of M-to-N and
make C wait for both it and N-to-M. If D and C are identical, retain the same
envelope in per-microbatch local state and insert only its real data dependency.
Tensor slots may have different shapes, dtypes, and differentiability; execute
their transfers in stable slot order and use captured schemas plus internal size
metadata for dynamic dimensions.

On backward, transfer only differentiable bypass slots directly from C to D.
Merge those gradients with routed-path gradients by explicit FX slot identity,
without routing them through R or duplicating accumulation.

Keep this mechanism generic: no tensor names, model types, residual detection,
or expert-kernel inspection may influence which values are carried.

#### 4. Bridge Piper-owned transport state directly between dense groups

Replace `_RoutedCrossGroupState`'s assumption that M-to-N and N-to-M share one
participant union. Represent the complete D/R/C lifecycle, with separate routed
groups for `D union R` and `R union C` plus the direct paired D/C lane map.

At dispatch, each D lane computes the route-count-derived split information
needed by its paired C lane. That private state travels in the direct dense-state
envelope; C uses it to size N-to-M receives. Each R lane derives its N-to-M send
splits from the counts it received during M-to-N. Thus model code never sees
Piper's transport state and R does not relay metadata solely for C. Equal D/C
widths make source ownership map directly by dense local ordinal.

Use the routed forward identity plus microbatch identity for lifecycle state,
as today, and preserve distinct directional process groups. Ensure every rank
in each union issues metadata and tensor collectives in the same statically
validated slot order. Remove retained dense forward/backward buffers and the
old state-owner relay paths once no callers remain; retain only the new
same-group local envelope and direct D/C transfer paths.

#### 5. Compatibility and migration

- Keep the three public marker APIs and schedule JSON syntax unchanged.
- Preserve within-group ragged A2A and same-dense-group cross-group routing.
- Reject unequal D/C widths early instead of synthesizing a relay or silently
  repartitioning model state.
- Update tests that currently assert state-owner return and dense relay nodes;
  do not change checked-in schedule semantics to accommodate the old lowering.
- Treat the changed cross-group routed buffer ABI as internal; serialized DAG
  artifacts produced by the old metadata shape must be regenerated.

### Evaluation plan

Evaluation proceeds CPU-first, then uses focused GPU tests. It does not run the
entire unit or end-to-end suite.

#### CPU/meta verification

| Correctness criteria | Verification |
|---|---|
| 1, 2 | Extend `test/test_afd_routed_directives.py` with distinct equal-width D and C groups. Assert exactly one M-to-N, one N-to-M, and one direct D-to-C dense-state transaction per forward lifecycle, their backward reversals, and absence of any relay through R or surrogate temporal chain. Add a same-D/C case that retains state locally and inserts no dense-state communication. |
| 3, 5 | Extend `test/test_graph_transform.py` with multiple bypass tensors of different shapes/dtypes. Assert stable FX-derived D-to-C bypass slot mapping, marker-slot isolation, and that bypass values do not become R inputs or outputs. |
| 3, 4, 6 | Replace retained-buffer/relay cases in `test/test_executors.py` with dense-state envelope tests covering direct lane pairing, different tensor schemas, nondifferentiable slots, direct reconstruction on C, same-group local retention, exact backward gradients, zero routed rows, and independent microbatch state. Mock communication and assert that C receives the layout derived by D while R derives send splits from its dispatch counts. |
| 7, 9 | Add routed lifecycle and sequence-validation cases for mismatched endpoint metadata, collective order, missing passthrough schemas, and unequal D/C widths. The unequal-width schedule must fail before communication nodes are inserted. |
| 1, 7, 8 | Update `test/test_afd_full_lowering.py` to lower the checked-in PP2 and PP4 AFD placements on meta tensors and verify connected per-ActorGroup DAGs with direct routed boundaries and direct/local dense state, without an R-mediated relay. Run the existing routed cases in `test/test_order_directive.py` and the existing within-group A2A lowering cases as regressions. |

Run the focused CPU files after their narrow cases pass:

```bash
cpu-venv/bin/python -m pytest -q \
  test/test_graph_transform.py \
  test/test_afd_routed_directives.py \
  test/test_afd_full_lowering.py \
  test/test_executors.py \
  test/test_order_directive.py
```

Also generate the four per-ActorGroup DAG visualizations for the motivating
AFD schedule and inspect that each routed edge has one communication, each
disjoint dense pair has one direct state path, and downstream ATTN regions wait
on N-to-M plus the direct state path rather than an R-mediated relay.

#### Focused GPU verification

After CPU/meta verification is green:

- run the existing Qwen 9M AFD PP2 and PP4 parameter/gradient parity cases;
- run the existing Qwen 9M DualPipeV parity case to guard within-group ragged
  A2A behavior;
- exercise passthrough tensors and backward gradients with at least one
  distinct-D/R/C topology where `len(D) == len(C)` and `len(R)` differs; and
- rerun the motivating four-ActorGroup Qwen30B L4 schedule, then capture the
  third profiler iteration after two profiler warmups and confirm the trace has
  one direct dense-state transfer and no R-mediated relay chain.

Record exact commands, results, trace/DAG artifact paths, environmental
limitations, and remaining topology restrictions in this document.

### Implementation log

- 2026-08-15: Pre-implementation research found no public AF-Pipe training code
  and no residual/state-path description in the paper or arXiv source. The only
  public repository named DisagMoE is a distinct inference project and cannot
  validate the paper's training semantics. Revised Phase 2 from routed-worker
  carriers to a direct D-to-C dense-state envelope; implementation remains
  paused pending approval of this material architecture change.
- 2026-08-15: User approved the revised design and requested an intermediate
  stop after CPU DAG verification and PP4 per-ActorGroup visualization. Runtime
  executor changes and GPU evaluation are intentionally deferred to the next
  implementation milestone.
- 2026-08-15: The existing FX splitter kept a long-lived value in every
  intervening segment ABI. Replaced that representation with stable runtime
  value descriptors and actual producer/consumer mappings: adjacent routed
  marker values form D-to-R and R-to-C dependencies, while bypass values form
  an explicit D-to-C dependency and do not enter R's GraphModule interface.
- 2026-08-15: Microbatch splitting initially copied producer/consumer mapping
  keys without remapping their UIDs, leaving copied lifecycles with empty
  mappings. Split lowering now remaps both lifecycle UID fields and structured
  runtime mapping keys for each microbatch.
- 2026-08-15: Removed the obsolete cross-group requirement that dispatch
  counts occupy the final dense output slot. Explicit marker metadata and the
  FX slot map now define payload/count ownership; focused tests cover the
  reversed count/payload ordering.
- 2026-08-15: Direct dense-state P2P nodes initially inherited the routed
  collective stream, which serialized the independent D-to-C path behind
  M-to-N/N-to-M. They now use `pp_stream`; the rendered DAG places each direct
  state transfer and routed collective at the same topological level after the
  producing dense compute.
- 2026-08-16: Runtime state is now one explicit D/R/C lifecycle with separate
  `D union R` and `R union C` communicator/split fields. D gathers dispatch
  counts, R retains only its source-major local-route counts, and C receives
  its paired D lane's counts through the direct envelope; no model value or
  backend state is retained on R solely to relay it to C.
- 2026-08-16: The direct envelope sends a fixed-rank shape header followed by
  each tensor. This supports symbolic/dynamic dimensions and heterogeneous
  dtypes without inspecting values. Backward sends only differentiable bypass
  slots and reconstructs D output-gradient slots from the FX mapping.
- 2026-08-16: Compute input and output-gradient assembly previously selected
  one predecessor. It now merges all explicit predecessor buffers by recorded
  FX slots and sums genuine fan-out gradient contributions. This is required
  for C to consume N-to-M plus direct bypass state and for D to consume their
  two reversed gradient paths.
- 2026-08-16: Direct bypass edges make a dispatch segment fan out to both R
  and C. Routed-marker ownership therefore continues to select the unique
  shard-matched successor; if none exists, validation reports the missing
  shard directive rather than rejecting the valid D-to-C edge.
- 2026-08-16: The current execution shell is the allocated `eos0321` Slurm
  node, but its container namespace has no `/dev/nvidia*` entries even though
  `/proc/driver/nvidia` lists eight GPUs and `NVIDIA_VISIBLE_DEVICES=all`.
  The command sandbox mounts a second `/dev` after the container's NVIDIA
  mounts; unsandboxed task execution exposes all eight H100 80GB devices and
  CUDA-enabled PyTorch. GPU evaluation uses that context.
- 2026-08-16: Initial GPU execution exposed two ABI gaps hidden by mocked
  collectives. Same-group A2A still returned an inherited full buffer, so it
  did not fill the routed slot when a second predecessor supplied bypass slots;
  A2A nodes now carry the same explicit FX mapping in forward and backward.
  Cross-group destinations also tried to infer zero-row payload storage from a
  local routed bucket, which a disjoint endpoint need not own. D now exchanges
  the runtime rank-2 payload shape with R and carries it privately to its paired
  C lane; destinations allocate from this backend state and the serialized
  dtype, with no model-visible metadata or stage-lookup heuristic.
- 2026-08-16: The first twelve-rank, two-node PP4 run exposed a host-dispatch
  cycle absent from the single-node parity tests: D posted a blocking direct
  state SEND before joining M-to-N, R waited for D in routed-layout exchange,
  and C could not reach the matching state RECV until the routed path advanced.
  Direct state now uses dedicated `dense_state_{lo_hi,hi_lo}` communicators and
  physical streams rather than ordinary PP traffic, and its heterogeneous
  sends are posted asynchronously with work/tensor lifetimes retained through
  an end-of-step drain. This preserves the approved independent D-to-C path;
  no schedule-specific ordering edge was added.
- 2026-08-16: Thirty-sample profile replay exposed a simulator-only cycle:
  it modeled every P2P SEND completion as waiting for peer rendezvous, while
  direct dense-state SEND uses nonblocking `isend` and lets local successors
  proceed once the operation is posted. The simulator now preserves the SEND
  arrival as a rendezvous prerequisite but makes its local completion depend
  only on posting; blocking sends and all receives retain rendezvous-completion
  semantics. A two-rank regression covers the formerly cyclic ordering.

### Intermediate CPU/DAG evaluation

- `cpu-venv/bin/python -m pytest -q test/test_graph_transform.py`: **26
  passed**. This includes two different-shape/dtype bypass tensors and verifies
  that neither becomes a routed-region runtime input or output.
- `cpu-venv/bin/python -m pytest -q test/test_afd_routed_directives.py -k
  "cross_group or pp4 or full_afd or unequal or split_remaps or hybrid or
  partially_overlapping"`: **13 passed, 20 deselected**.
- `cpu-venv/bin/python -m pytest -q test/test_afd_full_lowering.py -k
  "afd_placement_pp4"`: **1 passed, 3 deselected**. The traced Qwen DAG check
  verifies actual N-to-M destinations, direct D-to-C endpoints, nonempty model
  bypass state, and distinct local/direct state modes.
- `cpu-venv/bin/python -m pytest -q test/test_visualization.py`: **4 passed**.
- Regenerated and inspected four PP4 ActorGroup DAG PNGs plus text execution
  orders under `out/afd_pp4_direct_state_dags/`. Dense compute fans out to the
  M-to-N source and direct state SEND in parallel; the next dense compute has
  the actual N-to-M destination and direct state RECV as its two data
  predecessors.

### Runtime CPU evaluation

- `cpu-venv/bin/python -m pytest -q test/test_executors.py`: **27 passed**,
  including D/R/C split derivation, reverse collectives, direct forward and
  backward envelopes, and fixed-rank dynamic-shape transport.
- `cpu-venv/bin/python -m pytest -q test/test_graph_transform.py
  test/test_afd_routed_directives.py test/test_afd_full_lowering.py
  test/test_executors.py test/test_order_directive.py
  test/test_visualization.py`: **128 passed, 14 warnings**. The warnings are
  pre-existing TorchScript deprecations.

The runtime implementation no longer uses the old `dense_relay` metadata or
retained routed buffers.

### Focused GPU evaluation

- `gpu-venv/bin/python -m pytest -q -m gpu
  test/test_qwen_e2e_correctness.py -k "9M and (afd_pp2_single_node or
  dualpipev)"`: **2 passed, 16 deselected, 14 warnings** in 102.36s.
- `gpu-venv/bin/python -m pytest -q -m gpu
  test/test_qwen_e2e_correctness.py -k "9M and afd_pp4_single_node"`: **1
  passed, 17 deselected, 14 warnings** in 80.66s.
- The warnings are the same pre-existing TorchScript deprecations as the CPU
  suite. AFD PP4 exercises disjoint equal-width dense groups around routed
  singleton groups on six GPUs; DualPipeV covers the within-group ragged-A2A
  regression.

### Multi-node profiler evaluation

- Focused isolation regressions after adding dedicated dense-state streams,
  communicators, and asynchronous sends: `cpu-venv/bin/python -m pytest -q
  test/test_executors.py test/test_profile_streams.py
  test/test_actor_group_topology.py test/test_execution_plan.py
  test/test_afd_routed_directives.py test/test_afd_full_lowering.py`: **103
  passed, 14 warnings**.
- The old twelve-rank PP4 placement ran successfully on Ray
  `10.52.50.69:6379` with Qwen3-30B-A3B L4, batch 4, sequence length 1024,
  four microbatches, one ordinary warmup, and one timed iteration. Placement
  was D0 `[0,4,8,10]`, R0 `[1,5]`, D1 `[2,6,9,11]`, R1 `[3,7]`, split 6/6
  GPUs across the two nodes. The timed sample was **0.123780 s**.
- PyTorch profiling ran three additional iterations, warmed iterations 1-2,
  and retained only iteration 3. All 12 actor traces were combined into
  `out/20260816_113048/pytorch_profile_custom_pp2_world12_mbsX.json`
  (**75,107,682 bytes; 320,630 trace events**).
- Profile-and-simulate then collected 30 measured iterations for each of the
  normal and interference-free policies after five warmups per policy. After
  correcting the simulator's nonblocking-SEND model, dependency replay gave a
  median normal step of **161.667 ms** and interference-free step of **169.391
  ms**, or **-4.778% estimated savings** (a slowdown). Direct instrumented
  envelopes similarly measured **123.863 ms** normal and **131.045 ms**
  interference-free, or **-5.798%**. Treat the replay estimate as low
  confidence: both modes reported 249 negative communication residuals and
  replay overpredicted their instrumented medians by 30.521% and 29.262%,
  respectively. The reusable profile plan is
  `out/20260816_113645/step_time_profiles.json`.

### Completion

- Final CPU command: `cpu-venv/bin/python -m pytest -q
  test/test_graph_transform.py test/test_afd_routed_directives.py
  test/test_afd_full_lowering.py test/test_executors.py
  test/test_order_directive.py test/test_visualization.py
  test/test_piper_schedule.py`: **144 passed, 14 warnings**.
- Static checks: `py_compile` for `src/fx.py`, `src/dag.py`,
  `src/directives.py`, `src/piper.py`, and `src/executors.py`: **passed**;
  `git diff --check`: **passed**.
- Focused GPU result: **3 passed** across Qwen 9M AFD PP2, AFD PP4, and
  DualPipeV parameter/gradient parity.
- Multi-node Qwen30B L4 profiler workload: **passed** on twelve ranks across
  two nodes; combined third-iteration trace recorded above.
- Remaining topology limits are intentional approved scope: one routed marker
  triplet per lifecycle, rank-2 routed payloads, equal contiguous route
  ownership, and equal-width identical-or-disjoint D/C groups. Piper validates
  shapes/counts and transport ordering but cannot prove semantic routed-row
  identity from tensor values.

Commit message: `Generalize routed state flow across direct D/R/C boundaries`
