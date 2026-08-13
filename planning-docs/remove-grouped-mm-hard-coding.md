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
| Whole-expert ownership | `shard` directive and its ordered device list |
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

The initially supported ownership rule is **equal contiguous route-axis
sharding**. For `shard(N)`, Piper divides the ordered global route/expert axis
into `N` equal contiguous ranges. Piper must:

- validate that the global route count is divisible by the shard degree; and
- document and enforce that device-list order determines ownership.

For example, with eight experts and devices `[1, 3]`, device 1 owns experts
0--3 and device 3 owns experts 4--7. Uneven or noncontiguous ownership requires
a future explicit route-to-device mapping and must not be inferred
heuristically.

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
- the ordered shard device list unambiguously defines equal contiguous route
  ranges;
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

- derive equal contiguous route partitions from the ordered shard devices;
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

## 5. Validation across expert implementations

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

### GPU end-to-end tests

- Run Qwen grouped-MM and loop experts against the single-GPU eager reference
  under AFD PP2 and PP4 schedules.
- Run two optimizer steps so gradient scaling and ordering errors affect final
  parameters.
- Then run the existing Qwen e2e correctness suite for the default expert
  implementation.

### Static regressions

Add assertions that Piper core:

- does not inspect `torch._grouped_mm`;
- does not import TorchTitan MoE permutation helpers; and
- does not infer route semantics from expert tag names, operator sequences, or
  weight shapes.

## Expected result

After this work, the schedule and route-count annotations determine only
transport. Piper derives topology, ownership, splits, and reverse communication,
while the captured model region defines all expert-local layout and computation.
Adding a new expert implementation should require no changes to Piper core as
long as it obeys the documented routed-region tensor contract.
