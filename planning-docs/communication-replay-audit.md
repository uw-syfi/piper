# Communication Replay Audit

Date: 2026-08-18

## Summary

Piper's simulator replays the finalized DAG and its encoded dependencies, but
its communication abstractions do not consistently match the current runtime.
Symmetric reductions, parameter collectives, and routed A2A/M-to-N/N-to-M
nodes have an explicit all-participant rendezvous model, while ordinary P2P
has a paired-endpoint model.

The remaining material approximations are:

1. Routed nodes with several internal collectives use one node-entry
   rendezvous and one aggregate endpoint tail rather than separate phase
   rendezvous.
2. Dense-state P2P uses an ordinary-P2P microprofile that does not explicitly
   represent its shape headers and multiple dynamic tensors.
3. Communication/network resources and CPU issue latency are not modeled.

## Common Replay Semantics

- A profiled node's duration is the CUPTI-correlated envelope from its first
  attributed GPU activity to its last, including internal GPU gaps.
- A specific replay iteration uses the corresponding raw profile sample;
  otherwise replay uses the stored estimator.
- The finalized DAG remains authoritative. Incoming dependencies target an
  endpoint arrival vertex, and outgoing dependencies originate from endpoint
  completion.
- Same-physical-stream FIFO is encoded as temporal DAG edges before profiling.
- Schedule-declared streams are preserved except for SEND/RECV, whose
  directional P2P group selects the physical stream.
- Observed CUPTI start offsets are used for diagnostics and collective
  wait-calibration, not as replay release constraints.
- Independent streams and transactions overlap completely unless constrained
  by DAG or rendezvous edges. Replay does not model GPU capacity, network-link
  contention, communicator occupancy, or host dispatch capacity.

Relevant code:

- CUPTI envelopes: `src/actor.py:831`
- Profile sample selection: `src/simulator.py:300`
- Physical stream domains: `src/ordering.py:30`
- Same-stream FIFO: `src/ordering.py:296`, `src/piper.py:2205`
- Execution-plan dependencies: `src/execution_plan.py:79`,
  `src/simulator.py:1561`
- Arrival/completion lowering: `src/simulator.py:1087`,
  `src/simulator.py:1414`

## Per-Kind Audit

| Communication kind | Current replay behavior | Runtime behavior and mismatch |
|---|---|---|
| `SEND_COMM` / `RECV_COMM` | Groups endpoints by transaction and ActorGroup-local lane. SEND and RECV arrivals gate a two-party rendezvous, followed by independent endpoint completions. | Runtime uses `isend`/`irecv` followed by `Work.wait()`. Production replay substitutes an isolated matched-transfer microprofile, avoiding scheduled rendezvous wait, but one worst endpoint/lane median is reused and CPU issue/wait behavior is not modeled. |
| Dense-state `SEND_COMM` / `RECV_COMM` | Identical to ordinary P2P, while preserving the dedicated directional dense-state streams. | Runtime sends shape headers and multiple dynamically sized tensors. The generic P2P microprofile is based on receiver payload specifications and may not represent this protocol faithfully. |
| `REDUCE_COMM` | Symmetric collective keyed by kind and transaction ID. All participants rendezvous; endpoint service is calibrated as observed endpoint end minus the latest participant activity start. | Runtime performs all-reduce followed by local averaging. If one node synchronizes multiple buckets, replay approximates several sequential collectives as one initial rendezvous and one endpoint tail. |
| `ALL_REDUCE_COMM` | Same symmetric collective model as `REDUCE_COMM`. | This kind is recognized by replay and validation but is not currently lowered to a corresponding executable runtime task; it is effectively legacy/simulator compatibility. |
| `REDUCE_SCATTER_COMM` | Symmetric all-participant rendezvous with endpoint-specific post-rendezvous service. | Runtime performs reduce-scatter, local averaging, and accumulation into persistent shard storage. Local kernels are included in the profiled envelope; allocator/default-stream effects are not modeled independently. |
| `ALL_GATHER_COMM` | Symmetric all-participant rendezvous with endpoint-specific post-rendezvous service. | Runtime materializes/reuses full parameter storage and performs `all_gather_into_tensor`. CPU allocation/cache delays before GPU activity are deliberately absent from replay. |
| `M_TO_N_COMM` | Symmetric union-group rendezvous with an endpoint-specific aggregate post-rendezvous tail. Unequal and zero-byte endpoints are supported. | Runtime performs two metadata all-gathers before the forward payload all-to-all; other paths perform the payload collective. Replay assumes the same fixed participant set and call order, and does not recompute internal-phase skew. |
| `N_TO_M_COMM` | Same symmetric union-group aggregate model as M-to-N. | Runtime performs a variable-split union-group payload all-to-all. Payload sizes may differ; symmetry applies only to group readiness. |
| `A2A_COMM` | Symmetric ActorGroup rendezvous with an endpoint-specific aggregate post-rendezvous tail. Uneven and zero-byte endpoints are supported. | Forward dispatch performs count and payload all-to-all phases, while combine and gradient paths perform payload all-to-all. Replay treats those ordered internal calls as one logical node transaction. |

Relevant code:

- P2P grouping and rendezvous: `src/simulator.py:960`,
  `src/simulator.py:1307`, `src/simulator.py:1498`
- P2P runtime: `src/executors.py:298`
- P2P microprofiling: `src/actor.py:1397`, `src/piper.py:3146`
- Dense-state runtime: `src/executors.py:390`, `src/executors.py:2176`
- Symmetric collective replay: `src/simulator.py:449`,
  `src/simulator.py:1318`, `src/simulator.py:1354`
- Reduction runtime: `src/executors.py:483`, `src/executors.py:522`
- Parameter all-gather runtime: `src/runtime.py:771`
- Routed symmetric replay: `src/simulator.py:449`, `src/simulator.py:1318`
- M-to-N/N-to-M runtime: `src/executors.py:1663`,
  `src/executors.py:1885`, `src/executors.py:1925`
- A2A runtime: `src/executors.py:1354`

## Negative Interference-Free Savings Investigation

The PP2 world12 profile reported:

| Statistic | Mean | Sample SD |
|---|---:|---:|
| Instrumented normal | 1105.353 ms | 1.006 ms |
| Simulated normal | 1098.129 ms | 1.157 ms |
| Simulated interference-free | 1138.027 ms | 1.058 ms |

Normal replay underpredicted instrumented normal by 7.225 ms, or 0.654%.
The artifact did not retain uninstrumented hardware samples, so its accuracy
against uninstrumented execution cannot be stated. Reported simulated savings
were -39.898 ms, or -3.633% (95% CI: -3.685% to -3.582%).

The representative critical path changed as follows:

| Critical-path work | Normal | Interference-free | Change |
|---|---:|---:|---:|
| M-to-N | 340.301 ms | 399.256 ms | +58.955 ms |
| Compute | 385.892 ms | 375.388 ms | -10.504 ms |
| N-to-M | 361.724 ms | 353.934 ms | -7.790 ms |
| Reductions | 4.442 ms | 2.847 ms | -1.595 ms |

A category counterfactual confirmed the source of the reversal:

- Replacing only M-to-N normal profiles with interference-free profiles raised
  replay by 71.536 ms.
- Interference-free compute reduced replay by 12.801 ms.
- Interference-free N-to-M reduced replay by 8.993 ms.
- Interference-free reductions reduced replay by 0.336 ms.

Interference-free serialization placed destinations after sources in 18 of 32
M-to-N transactions, with a mean positive gap of 4.610 ms and a maximum of
8.681 ms. Normal profiling did so in only 6 of 32 transactions, with a mean
gap of 0.179 ms. Because source endpoints bypass the simulated rendezvous,
their CUPTI envelopes retain this serialized peer wait and replay treats it as
communication service.

An in-memory diagnostic that treated the current M-to-N transport as a
symmetric union-group rendezvous produced:

- Simulated normal: 1098.365 ms
- Simulated interference-free: 1024.726 ms
- Simulated savings: 73.640 ms

Normal changed by only 0.237 ms while the negative interference-free result
disappeared. Capture loss and clock alignment cannot explain the discrepancy:
GPU capture loss was 0.00144%, CPU launch loss was 0.00153%, and maximum clock
uncertainty was 0.469 ms.

Conclusion: the negative interference-free savings are a replay-modeling
artifact, not credible evidence that removing interference slows execution.
The production replay now uses the symmetric aggregate model exercised by the
diagnostic above.

## Alignment and Validation

Cross-actor CUPTI alignment is mandatory for reductions, all-gathers,
reduce-scatters, A2A, M-to-N, and N-to-M. Endpoint sample IDs must match, and
negative post-rendezvous service is rejected beyond declared clock uncertainty.

P2P is excluded because it uses isolated microprofile service and does not
require aligned scheduled endpoints. Routed collectives require complete,
consistent participant metadata. Zero-activity routed endpoints are accepted
as zero service, while their arrival still gates the logical rendezvous.

Malformed cross-actor transactions or inconsistent local ordering that creates
a replay cycle are rejected.

Relevant code: `src/simulator.py:463`, `src/simulator.py:1065`,
`src/simulator.py:1138`, and `src/profile_validation.py:11`.

## Remaining Limitations

- Replay does not model network/link contention, communicator occupancy, GPU
  resource competition, or CPU dispatch latency.
- Different streams overlap perfectly unless DAG or rendezvous edges constrain
  them, even when operations share links or communicators.
- A routed node envelope can contain several internal communication phases,
  but replay represents it with one rendezvous and one aggregate endpoint tail;
  later internal wait remains baked into that tail.
- Profiled placement cannot safely be extrapolated to a different physical
  topology.
- ZeRO-2's post-update parameter all-gather occurs inside `UPD`, not an
  `ALL_GATHER_COMM` node. It is replayed as rank-local update duration without
  explicit collective rendezvous and can retain embedded peer wait.
