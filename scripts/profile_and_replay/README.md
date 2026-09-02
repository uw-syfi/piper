# Profile-and-replay tools

This package contains the public offline replay CLI, profile-ratio exporter,
and reusable diagnostics for comparing instrumented traces with dependency
replays.

Profile collection must be launched with `--no-use-inductor`. The current
profile-and-replay path operates on the eager stage GraphModules; compiling
those already-lowered graphs can fail when TorchDynamo encounters embedded
Triton higher-order operators such as `triton_kernel_wrapper_mutation`.

## Approach

Profile-and-replay measures the same finalized `TrainingDAG` under two CUDA
execution policies and then replays both sets of node measurements against the
same dependency graph. The normal replay estimates the profiled schedule with
its observed interference; the isolated replay estimates the same
work with otherwise-independent node work serialized during measurement.

### Normal-mode profiles

- Piper restores the same model, optimizer, input, and RNG state before every
  sample, so all samples and both policies execute the same logical work.
- After the configurable warmup (five iterations by default), Kineto/CUPTI
  captures 30 iterations by default. Existing `record_function` ranges
  correlate CUDA activity with each DAG node; Piper does not add per-node CUDA
  timing events, which would perturb concurrent execution.
- Nodes execute with the finalized DAG's ordinary dispatch order, dependencies,
  and logical streams. Directional P2P still uses its normal high-to-low and
  low-to-high physical streams. Independent work can overlap as it does during
  an ordinary training step.
- A node profile is the aggregate CUDA activity envelope from its first
  attributed activity to its last. It therefore includes kernels, copies, and
  gaps inside that range. Synchronized whole-step wall-clock samples are kept
  separately as instrumented-step diagnostics; they are not node durations.

Actor clocks and CUDA timelines are aligned so participant wait time can be
separated from endpoint-local service for both collectives and P2P. For each
P2P node, a CPU-only marker immediately before its first `isend`, `irecv`, or
`batch_isend_irecv` call records that endpoint's compound-transaction post
time. This timestamp is distinct from the later start of NCCL GPU activity.

### Isolated profiles

Isolated capture runs the same DAG operations and communication
protocols from the same restored state. It preserves every node's normal
logical/physical stream instead of moving work to a profiler-only stream. A
transaction-consistent global order keeps the participants of P2P and
collective operations in a deadlock-safe order; CUDA events outside the node
ranges serialize successive work across streams. Same-stream serialization is
satisfied by CUDA FIFO rather than an extra event wait. Matching endpoints of
one communication transaction still rendezvous together.

This policy removes overlap between otherwise-independent node work while
retaining stream-specific kernel behavior and the actual runtime protocol. Its
CUPTI envelopes are used as the isolated service measurements. Capture stores
each communication node's combined activity envelope as one node measurement.

### DAG replay

The CUDA executor and replay consume the same lowered execution plan. For
each captured iteration, the replay discards the observed node start times
and computes the earliest legal start of every node from:

- all explicit data and temporal DAG dependencies;
- physical-stream FIFO order already encoded in the finalized DAG;
- the selected mode's rank-local CUPTI node durations; and
- communication readiness and completion semantics.

Replayed completion is determined by node family:

- **Compute and other non-communication nodes:** completion is the maximum end
  of the node's data, temporal, and DAG-encoded same-stream predecessors plus
  the selected profile's complete CUDA activity-envelope duration.
- **Ordinary collectives and A2A** (`all-reduce`, `all-gather`,
  `reduce-scatter`, and `A2A`): every participant first reaches an
  all-participant GPU rendezvous. Each endpoint completes after its
  policy-specific post-rendezvous service tail. Every participant must arrive;
  payloads and endpoint durations need not be equal.
- **M-to-N and N-to-M routed communication:** replay rejects these nodes. A
  node may issue an asymmetric or compound set of peer sends and receives, but
  its whole-node activity envelope does not expose their matching structure,
  per-transfer service times, issue order, or overlap. Those details are
  required for accurate peerwise replay.
- **P2P send/receive:** a matched SEND/RECV node pair is one compound
  transaction, including any internal header, payload, or batched operations.
  Send completion adds its aggregate profiled tail after its own first-call
  timestamp and send-source readiness; it does not wait for the receive call.
  Receive completion adds its own tail after both first-call timestamps plus
  send-source and receive-destination readiness. The endpoints therefore
  complete independently, and send may complete before receive starts.

The normal profile's host post schedule is reused by both replays because it is
the schedule's intended asynchronous issue behavior. Isolated post timestamps
are used only to remove matching and local-readiness wait from isolated P2P
service measurements. Issue-only DAG edges do not become GPU timing
predecessors. The replayed step time is the makespan of the resulting event
graph.

Reported metrics use these signed formulas:

- prediction error = `(normal replay - instrumented normal step) / instrumented normal step`;
- interference overhead = `(normal replay - perfect-overlap replay) / normal replay`.

The prediction-error endpoints are arithmetic means across their respective
normal-replay and synchronized instrumented-normal samples. The ordinary
uninstrumented step time remains a separate reported measurement and is not the
prediction-error reference.

Durable Kineto profiles containing P2P nodes must include these host post
timestamps. Replay fails closed with a re-profile error for older P2P artifacts
instead of falling back to GPU-kernel rendezvous. Duration-only synthetic
profiles may omit them and are treated as preposted at time zero.

Replay accounts for the exact profiled DAG, placement, dependencies, stream
assignment, rank-local kernel envelopes, participant readiness, and measured
endpoint asymmetry. Normal replay uses normal-mode envelopes, in which hardware
interference is already present; isolated replay uses the serialized
measurements. Their difference estimates interference/overlap overhead for
that captured workload and topology.

Replay is not a GPU or network resource replay. It does not independently
model SM or memory-bandwidth contention, network links/NIC capacity,
communicator occupancy, general CPU launch latency beyond the captured P2P post
releases, allocator behavior outside the attributed CUDA ranges, or
interference created by a different schedule.
Otherwise-independent streams overlap without resource limits. A2A nodes are
modeled with one all-participant entry rendezvous and one aggregate tail;
replay does not model their underlying peer-to-peer transfer graph, nonzero
split pattern, or independent peer progress. M-to-N and N-to-M nodes are
unsupported because their potentially asymmetric, compound transfers cannot
be reconstructed from whole-node timing alone. Dense-state P2P headers and
multiple tensors are also not separate replay phases.
Consequently, results should not be extrapolated to a different topology,
tensor shape, routing distribution, or kernel selection without collecting
new profiles.

The capture and merge path is implemented in `src/piper.py` and `src/actor.py`;
the shared lowering is in `src/execution_plan.py`; and replay is implemented in
`src/replay.py`.

## Offline tools

From the repository root:

```bash
# Collection through the example harness requires --no-use-inductor.
python examples/run_harness.py ... --profile-and-replay --no-use-inductor

python scripts/profile_and_replay/replay_step_times.py PROFILE.json
python scripts/profile_and_replay/export_node_profile_ratios.py PROFILE.json
python scripts/profile_and_replay/attribute_replay_savings.py PROFILE.json
python scripts/profile_and_replay/visualize_replay_savings.py \
  NORMAL_INPUT ATTRIBUTION.json --output savings-trace.json
```

Additional trace diagnostics live beside these tools:

```bash
python scripts/profile_and_replay/diagnostics/analyze_gpu_interference.py \
  OUTPUT_DIRECTORY
python scripts/profile_and_replay/diagnostics/analyze_node_profile_interference.py \
  OUTPUT_DIRECTORY
```

The tools import production profile and replay types from `src/`; they do
not contain a separate replay implementation.
