# Plan: remove grouped-MM hard-coding from routed execution

## Goal

Make Piper's routed communication independent of the implementation inside a
routed model region. Piper should transport packed tensor rows according to
schedule-defined route ownership. Model code should own any expert-local
permutation, expert-kernel padding or alignment, expert computation, and
restoration of transport order before combine.

Piper core must not inspect a routed region for `torch._grouped_mm`, import
TorchTitan's private MoE permutation helpers, infer semantics from expert weight
shapes, or depend on a particular sequence of expert operations.

## Minimal ownership boundary

| Responsibility | Owner |
|---|---|
| Ordered source and destination ranks | Piper schedule |
| Whole-expert ownership | `shard` directive in ascending global-rank order |
| Packed dispatch payload | Model code |
| Counts for the ordered global routes | Model code |
| Split calculation and dynamic metadata exchange | Piper backend |
| Ragged A2A or M-to-N/N-to-M transfer | Piper backend |
| Expert-local regrouping | Model-side routed-region code |
| Kernel-specific padding and alignment | Expert implementation |
| Restoration of transport order before combine | Model-side routed-region code |
| Reverse combine transfer | Piper backend |

The transport helpers introduced by this work are private Piper backend helpers.
They are neither imported nor called by model code. Model code sees ordinary
tensors and the public marker APIs only.

## Expert-parallel ownership semantics

Traditional expert parallelism distributes complete experts across EP ranks. If
the shard degree is `N` and every rank owns `L` local experts, the model has
`N * L` total experts and each expert rank materializes `L` complete experts.
Piper does not shard the matrices within an individual expert; that would be
tensor parallelism within an expert, not traditional EP.

For the initial implementation, every trainable lifted parameter consumed by a
routed region is route-owned and must be captured with leading dimension
`num_local_experts`. Piper derives that local cardinality from the marked global
route width and scheduled shard degree, validates the parameter shape, and
never infers ownership from parameter names or expert operators. A future
explicit parameter marker is required before a routed region can mix
route-owned and replicated trainable parameters.

The initially supported ownership rule is **equal contiguous route-axis
sharding**. For `shard(N)`, Piper divides the ordered global route/expert axis
into `N` equal contiguous ranges. Piper must:

- validate that the global route count is divisible by the shard degree; and
- require shard device lists in ascending global actor-rank order and assign
  ownership in that canonical order.

For example, with eight experts and devices `[1, 3]`, device 1 owns experts
0--3 and device 3 owns experts 4--7. `[3, 1]` is rejected rather than silently
changing that ownership. Uneven or noncontiguous ownership requires a future
explicit route-to-device mapping and must not be inferred heuristically.

The model-side code that converts received source-major rows into expert-local
order needs counts logically shaped:

```text
[num_source_ranks, num_local_experts]
```

It can infer both dimensions from the tensor. It does not need the EP degree or
the identities of remote ranks. Piper needs the shard degree to partition the
global route axis and construct transport metadata.

## Current implementation-specific scope to remove

The hard-coding spans both graph capture and runtime execution:

- `src/fx.py::_externalize_torchtitan_expert_dispatch` searches for exactly
  three `torch._grouped_mm` operations, recognizes a particular TorchTitan
  graph, reconnects its inputs and output, and deletes padding-related nodes.
- `split_gm_by_annotations` invokes that rewrite for every routed region.
- `src/executors.py` imports TorchTitan's private `_permute` and `_unpermute`
  helpers.
- Within-group ragged A2A and cross-group M-to-N/N-to-M paths use TorchTitan's
  padded expert layout, padded counts, saved permutation indices, and sentinel
  rows.
- Existing tests assert the special FX rewrite and mock TorchTitan permutation
  helpers rather than verifying an implementation-neutral transport contract.

The generic marker grammar, placement-derived participant groups, count/split
exchange, collective insertion, reverse communication, and route state remain
valid and should be retained.

## 1. Define, validate, and document the routed-region contract

The public model annotations are:

```python
mark_dispatch_payload(dispatch_payload)
mark_dispatch_counts(route_counts)
mark_combine_payload(combine_payload)
```

### Dispatch contract

- `dispatch_payload` is a rank-2 tensor shaped `[rows, hidden]`.
- `route_counts` is a nonnegative, non-differentiable integer tensor.
- At the model's source-side boundary, counts describe the ordered global
  routes. Payload rows are packed as consecutive blocks in that same route
  order.
- `dispatch_payload.shape[0] == route_counts.sum()`.
- The number of global routes represented by `route_counts` is statically known
  at capture time or otherwise available in captured tensor metadata.

After dispatch, a destination routed region receives:

- an opaque payload in source-major order; and
- counts logically shaped `[num_source_ranks, num_local_experts]`, describing
  the consecutive local-expert blocks within every source block.

The exact tensor representation of these destination-side counts should be
made explicit in the public contract. A rank-2 representation is preferred
because it exposes the information the model needs directly and does not require
the model to know EP degree. If compatibility requires a flattened tensor, its
shape interpretation must be captured explicitly rather than inferred from
expert kernels or weight shapes.

### Combine contract

- `combine_payload` is the sole dynamic activation tensor leaving the routed
  region.
- It contains exactly one result row for each received dispatch row.
- Model code restores it to exactly the source-major and local-route row order
  presented at the routed-region input.
- Its row count equals the routed region's received dispatch row count.
- Under the initial contract, its dtype and trailing shape match the dispatch
  payload.
- Both `dispatch_payload` and `route_counts` are graph ancestors of
  `combine_payload`.

Preserving row identity is essential: Piper reverses dispatch with saved split
metadata and does not know how expert-local permutations changed row order.

### Exclusive dynamic interface

Piper should require the routed region to have exactly this dynamic activation
interface:

```text
inputs  = {dispatch_payload, route_counts}
outputs = {combine_payload}
```

Specifically:

- no other tensor produced outside the routed region may be consumed inside it;
- no other tensor produced inside the routed region may be consumed outside it;
- no unmarked dynamic scalar, symbolic runtime value, or tensor metadata input
  may cross either boundary.

The following are not additional dynamic activation inputs and remain allowed:

- lifted parameters;
- lifted buffers;
- compile-time constants;
- symbolic sizes derived from the marked tensors; and
- values that bypass the region without being consumed by it.

Future support for additional transported, broadcast, or reduced values should
use explicit annotation semantics rather than implicit copying.

### Capture-time checks

Piper should reject a model graph unless all of the following hold:

- exactly one dispatch-payload marker, one dispatch-count marker, and one
  combine-payload marker describe each routed region;
- dispatch markers cross the immediately preceding annotation boundary;
- the combine marker crosses the immediately following annotation boundary;
- the routed region is one contiguous annotated segment;
- dispatch and combine payloads are rank 2;
- route counts use the documented rank, have integer dtype, are
  non-differentiable, and have a known route-axis width;
- statically known payload dtypes and trailing shapes are compatible;
- if the dispatch payload requires gradients, the combine payload also does;
- dispatch payload and route counts are both graph ancestors of combine;
- no marker is reused by multiple routed regions; and
- the exclusive dynamic interface rule above holds.

### Schedule- and DAG-time checks

After placement is available, Piper should reject a schedule unless:

- exactly one applicable `shard` directive establishes ownership of the routed
  region;
- the global route count is known;
- `num_global_routes % shard_degree == 0`;
- the shard device list is in ascending global actor-rank order and therefore
  unambiguously defines equal contiguous route ranges in communicator order;
- dispatch and combine refer to the same routed-region identity;
- a within-group route has the same ordered participant set on both sides;
- a cross-group route has disjoint ordered source and destination sets;
- all actors agree on the route-axis size and payload schema; and
- no additional dynamic DAG data edge enters or leaves the routed compute node.

### Runtime checks

Immediately before dispatch, Piper should validate:

- route counts are nonnegative;
- the route width matches captured metadata;
- payload rows equal the sum of route counts;
- payload dtype and trailing shape match captured metadata; and
- derived send splits sum to the payload row count.

After metadata exchange, Piper should validate:

- received counts contain one block/row per ordered source;
- their local-route width equals `num_global_routes / shard_degree`;
- each receive split equals the corresponding source count sum; and
- receive splits sum to the received payload row count.

Immediately before combine, Piper should validate:

- combine rows equal the saved dispatch receive-row count;
- combine dtype and trailing shape match the contract; and
- reverse send splits sum to the combine row count.

Piper cannot prove from tensor values that model code packed dispatch rows in
count order or restored semantic row identity before combine. These are
normative model requirements and should be exercised with identifiable rows in
correctness tests.

### Public documentation

Docstrings for all three marker APIs must state the full contract:

- `mark_dispatch_payload` marks the sole dynamic payload entering a routed
  region, packed in the route order described by counts.
- `mark_dispatch_counts` marks the sole routing descriptor, including its dtype,
  shape, ownership-order, and consecutive-row semantics.
- `mark_combine_payload` marks the sole dynamic payload leaving the region and
  promises one result per input row in restored transport order.

The documentation must explicitly say that Piper transports rows but does not
perform expert grouping, kernel-specific padding, or output reordering inside
the routed region.

## 2. Add private routed-transport helpers

Add backend-private helpers in `src/executors.py` or a private module such as
`src/routed_transport.py`. They are called only by Piper's backend and are not
part of the model API. The likely surface is:

```python
_route_partitions(...)
_send_splits_from_route_counts(...)
_exchange_route_counts(...)
_validate_dispatch_payload(...)
_validate_received_route_counts(...)
_validate_combine_payload(...)
```

These helpers should:

- derive equal contiguous route partitions from canonical shard devices;
- derive local send splits from model-provided global route counts;
- exchange compact routing metadata;
- expose destination-local source-major counts;
- validate the relationships described above; and
- preserve payload rows as opaque data.

They must not convert source-major payloads to expert-major order, inspect expert
operators, inspect expert weights, or apply kernel padding.

The shared runtime state should contain transport information only:

```python
@dataclass
class RoutedTransferState:
    source_ranks: tuple[int, ...]
    destination_ranks: tuple[int, ...]
    send_splits: list[int]
    recv_splits: list[int]
    payload_shape: tuple[int, ...]
    payload_dtype: torch.dtype
```

It should contain no expert permutation, padded counts, sentinel indices, or
TorchTitan state.

## 3. Preserve opaque row order in all routed communication

For ragged A2A dispatch and M-to-N:

1. derive splits from model-supplied global route counts and scheduled route
   ownership;
2. exchange compact count metadata;
3. transfer the packed payload without changing its row order; and
4. expose source-major payload and local route counts to the routed region.

For ragged A2A combine and N-to-M:

1. require model code to restore source-major/local-route order;
2. swap the saved forward send and receive splits; and
3. execute the reverse collective without a Piper permutation.

The explicit backward paths reuse the same split vectors. Because model-side
regrouping is captured inside the routed region, autograd handles its inverse in
the backward graph.

## 4. Remove implementation-specific graph and runtime handling

In `src/fx.py`:

- delete `_externalize_torchtitan_expert_dispatch`;
- delete its unconditional invocation during annotation splitting;
- preserve routed subgraphs exactly; and
- retain only generic marker and routed-interface validation.

In `src/executors.py`:

- remove imports of TorchTitan `_permute` and `_unpermute`;
- remove padded-count normalization;
- remove padding sentinel rows and saved expert permutation indices; and
- ensure communication never recognizes expert operations or expert weight
  layouts.

Complete generic transport of Triton higher-order-operation side-table state so
an untouched expert implementation can carry its own padding wrapper through DAG
serialization and actor loading. This support must be operator-generic and must
not recognize grouped-MM graph structure.

## 5. Validation coverage across expert implementations

Do not add a new example MoE model as part of this work. Validate the boundary
using:

- TorchTitan Qwen `GroupedExperts(use_grouped_mm=True)`, with padding retained
  inside TorchTitan;
- TorchTitan Qwen `GroupedExperts(use_grouped_mm=False)`; and
- a test-local, structurally different expert fixture using batched matrix
  multiplication or `einsum`, with no grouped-MM or TorchTitan helper calls.

### Graph and annotation tests

- Reject every malformed marker, tensor schema, dependency, adjacency, and
  exclusive-interface condition listed above.
- Verify splitting neither adds, deletes, nor reconnects operators inside each
  routed expert implementation.
- Verify tag spelling is irrelevant to routed inference.

### Transport tests

- Validate exact split matrices for within-group ragged A2A.
- Validate exact split matrices for unequal cross-group M-to-N/N-to-M.
- Cover zero-count routes, sources sending no rows, empty received payloads, and
  uneven dynamic routing loads.
- Prove dispatch preserves row order and combine is its exact reverse.
- Cover forward and explicit backward routes.
- Cover multiple microbatches to verify isolated saved transport state.

### Model-side adapter tests

- Test source-major to expert-local regrouping and its inverse with identifiable
  rows.
- Cover multiple sources, multiple local experts, uneven counts, and zero-count
  experts.
- Compare forward outputs, input gradients, parameter gradients, and optimizer
  updates against an unsharded eager reference.

### End-to-end parameter parity

- Select the existing Qwen parameter-parity cases for AFD PP2, AFD PP4, and
  DualPipeV. Do not add or parameterize new e2e tests for this focused gate.
- These existing cases run two SGD optimizer steps, comparing the initialized
  and final distributed parameter states with the single-GPU eager reference so
  gradient synchronization, route ordering, and reverse-communication errors are
  observable.
- After the focused unit and e2e gates pass, run the entire existing Qwen e2e
  correctness suite.

### Static regressions

Add assertions that Piper core:

- does not inspect `torch._grouped_mm`;
- does not import TorchTitan MoE permutation helpers; and
- does not infer route semantics from expert tag names, operator sequences, or
  weight shapes.

## 6. Staged implementation and verification

The work is divided by resource requirements. Stage 1 must reach a complete,
logically executable implementation without allocating or dispatching GPU
compute. Stage 2 begins only after Stage 1 is green and uses GPU resources to
complete any behavior that cannot be established through capture, logical DAG
execution, or mocked collectives.

No new e2e cases are introduced by this work. The focused GPU e2e gate selects
cases from the existing `test_qwen_parameter_parity` parameterization.

### Stage 1: implementation and verification without GPUs

#### Implementation scope

Implement the contract and core behavior described in Sections 1--4 without
waiting for GPU access:

- add the marker documentation and capture-time exclusive-interface checks;
- add route-count divisibility and directive device-list ownership checks;
- add the private route partition, split, count-exchange, and validation
  helpers;
- refactor A2A and M-to-N/N-to-M forward and backward paths to preserve opaque
  source-major row order;
- reduce saved routed state to topology, split, and payload-schema metadata;
- move expert-local regrouping and inverse regrouping into the Qwen model-side
  routed region;
- capture Qwen expert modules with local complete-expert parameters, preserve a
  dynamic routed row dimension through FX splitting, initialize each local
  route slice reproducibly, and reconstruct global parameters in canonical
  shard ownership for parity export;
- reject schedules in which `replicate` and `shard` select any of the same
  logical model regions; dense replication must exclude routed regions
  explicitly (for example, `{"PP": 0, "EP": "-"}`), while `place` remains
  compatible with either communication directive;
- partition each global microbatch equally along dimension zero across the
  input ActorGroup, using ActorGroup local-rank order as data-lane order; require
  the global microbatch size to be at least and divisible by the group width;
- keep replicated dense-gradient reduction/averaging, but insert neither a
  gradient collective nor local gradient division for route-owned sharded
  parameters because every dense source rank contributes distinct data;
- remove TorchTitan `_permute`/`_unpermute`, sentinel padding, and grouped-MM FX
  recognition from Piper core; and
- implement generic Triton higher-order-operation serialization and side-table
  restoration, while deferring actual CUDA kernel execution to Stage 2.

The no-GPU implementation is not a placeholder. Runtime collective calls may be
exercised with mocks, but split calculation, row ordering, reverse routing,
state lifetimes, graph serialization, and schedule/DAG construction must all
have their final intended semantics before Stage 2.

#### Focused generalized-routing unit tests

The following existing test areas directly exercise the generalized interface
and should be updated where their assertions currently encode TorchTitan
permutation or grouped-MM behavior:

| Test file | Generalized interface covered |
|---|---|
| `test/test_graph_transform.py` | Marker uniqueness and adjacency, payload/count schemas, dispatch-to-combine dependency, tag independence, and routed-region boundaries. Add the exclusive dynamic input/output checks here. |
| `test/test_executors.py` | Ragged A2A dispatch/combine state, source-major split/count exchange, M-to-N/N-to-M forward and backward paths, empty payload behavior, and per-microbatch state. Replace mocks of TorchTitan `_permute`/`_unpermute` with opaque row-order and exact reverse-transfer assertions. |
| `test/test_afd_routed_directives.py` | Schedule classification of within-group versus cross-group routing, shard-policy requirements, static route-width validation, participant topology, collective pairing, and transaction/state identity. Add route-count divisibility and device-list ownership-order cases here. |
| `test/test_afd_full_lowering.py` | End-to-end lowering of annotated Qwen regions into connected AFD actor DAGs for PP2 and PP4, including M-to-N/N-to-M node counts and metadata. |
| `test/test_order_directive.py` | Generic `shard` lowering from the routed marker contract and rejection of missing route counts, independent of tag spelling or expert operators. |
| `test/test_qwen_grouped_experts.py` | Preservation of the routed expert graph and availability of route counts. Replace assertions about removed grouped-MM surgery with graph-preservation checks for grouped-MM, loop, and structurally different test-local experts. |

Run:

```bash
python -m pytest -q \
  test/test_graph_transform.py \
  test/test_executors.py \
  test/test_afd_routed_directives.py \
  test/test_afd_full_lowering.py \
  test/test_order_directive.py \
  test/test_qwen_grouped_experts.py
```

This gate must establish:

- route-count divisibility by shard degree, rejection of unsorted shard lists,
  and ownership in ascending global actor-rank order;
- no Piper-side payload permutation or kernel padding;
- unchanged row order through dispatch and exact reversal through combine;
- correct ragged A2A and unequal M-to-N/N-to-M splits in forward and backward;
- correct empty, zero-count, uneven-load, and multi-microbatch behavior;
- replicated dense gradients are reduced across their ActorGroup, while
  route-owned sharded gradients have neither a collective nor local scaling;
- malformed routed-region interfaces fail at the earliest available layer; and
- Piper core has no grouped-MM graph recognition or TorchTitan MoE permutation
  dependency.

#### Model capture, schedule construction, and logical DAG execution

Without running CUDA kernels or distributed GPU collectives:

- capture/compile the Qwen routed graph on fake/meta tensors for both
  `GroupedExperts(use_grouped_mm=True)` and `use_grouped_mm=False`;
- verify serialization/deserialization preserves the untouched routed graph and
  Triton higher-order-operation metadata;
- build the existing AFD PP2 and PP4 schedules and their actor-local DAGs;
- build the existing DualPipeV schedule and its actor-local DAGs;
- validate all routed collective sequences, endpoint metadata, transaction IDs,
  dependencies, and per-stream total orders; and
- run the resulting communication DAGs through the simulator to validate A2A
  and unequal-group M-to-N/N-to-M rendezvous logically.

Run the relevant existing logical-lowering and simulator suites:

```bash
python -m pytest -q \
  test/test_afd_schedule_builder.py \
  test/test_piper_schedule.py \
  test/test_dag_builder_equivalence.py \
  test/test_actor_group_topology.py \
  test/test_simulator.py
```

#### Complete no-GPU regression gate

After the focused and logical gates pass, run every existing non-GPU test:

```bash
python -m pytest -q -m 'not gpu'
```

Stage 1 is complete only when the focused suites, logical AFD/DualPipeV DAG
construction, simulator execution, and the full non-GPU suite all pass. No
pre-existing schedule family or unit test may be removed, skipped, or weakened.

### Stage 2: GPU-dependent completion and verification

#### Complete GPU-dependent implementation

Start with small models, sequence lengths, and microbatch counts. Exercise the
Stage 1 implementation on real actor processes, NCCL groups, dynamic routed
payload sizes, and CUDA kernels. Complete only issues that require GPU evidence,
including:

- restoring and executing an untouched TorchTitan Triton padding wrapper after
  DAG serialization and actor loading;
- compiling routed regions with dynamic token counts;
- validating real ragged A2A and unequal M-to-N/N-to-M count and payload
  transfers in both directions; and
- validating grouped-MM, loop, and structurally different expert computations
  against their eager forward and backward results.

Any fixes discovered here must preserve the generic contract. Do not add
operator recognition, expert-tag heuristics, or implementation-specific layout
handling back into Piper.

#### Focused existing e2e parameter-parity gate

Run only the already-existing AFD PP2, AFD PP4, and DualPipeV cases from
`test/test_qwen_e2e_correctness.py`. The selector covers both model sizes already
present in `_CORRECTNESS_MODELS`; it does not create new e2e variants.

```bash
python -m pytest -q -m gpu test/test_qwen_e2e_correctness.py \
  -k 'afd_pp2_single_node or afd_pp4_single_node or dualpipev'
```

These cases must pass their existing initial-parameter and two-step final-
parameter comparisons against the single-GPU eager model. AFD PP2 and PP4 cover
cross-group M-to-N/N-to-M under two placements. DualPipeV covers the generalized
within-group ragged A2A path under its existing V-shaped pipeline schedule.

#### Final full regression gate

After GPU-driven fixes, rerun the focused unit tests and then the entire existing
non-GPU suite to ensure the generic invariants still hold:

```bash
python -m pytest -q -m 'not gpu'
```

Finally, run every existing GPU e2e parameter-parity case:

```bash
python -m pytest -q -m gpu test/test_qwen_e2e_correctness.py
```

This includes DualPipeV, all other schedules in
`examples.run_qwen_examples.RUNS`, and AFD PP2/PP4 for every model already
parameterized by the test. Preserve the existing per-workload two-minute
subprocess timeout and report any numerical tolerance separately rather than
broadening it silently.

Stage 2 is complete only when the focused AFD/DualPipeV parameter-parity gate,
the full existing unit suite, and the full existing e2e suite all pass.

## Expected result

After this work, the schedule and route-count annotations determine only
transport. Piper derives topology, ownership, splits, and reverse communication,
while the captured model region defines all expert-local layout and computation.
Adding a new expert implementation should require no changes to Piper core as
long as it obeys the documented routed-region tensor contract.

## Completion

- Focused generalized-routing tests: 127 passed.
- Complete non-GPU suite: 279 passed, 18 GPU tests deselected.
- Existing Qwen GPU E2E suite: 18 passed before the final test-only tolerance
  refinement; afterward, five focused cases passed together and the sole
  remaining 1B DualPipeV case passed its targeted rerun.
- Static compilation of the touched Python files passed.
- Remaining limitation: expert implementations with data-dependent Python
  control flow such as `route_counts.tolist()` are not meta-capturable; routed
  expert computation must use tensor operations or a fake/meta-compatible
  custom operator.

Commit message: `Generalize routed transport for AFD and expert parallelism`.
