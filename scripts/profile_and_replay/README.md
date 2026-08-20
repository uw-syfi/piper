# Profile-and-replay tools

This package contains the public offline replay CLI, profile-ratio exporter,
and reusable diagnostics for comparing instrumented traces with simulator
replays.

## Approach

Profile-and-simulate measures the same finalized `TrainingDAG` under two CUDA
execution policies and then replays both sets of node measurements against the
same dependency graph. The normal replay estimates the profiled schedule with
its observed interference; the interference-free replay estimates the same
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

Scheduled P2P ranges can contain time spent waiting for a peer, so Piper first
microprofiles each matched transfer and records an isolated transfer baseline
for replay. Actor clocks and CUDA timelines are also aligned so collective wait
time can be separated from endpoint-local service.

### Interference-free profiles

Interference-free capture runs the same DAG operations and communication
protocols from the same restored state. It preserves every node's normal
logical/physical stream instead of moving work to a profiler-only stream. A
transaction-consistent global order keeps the participants of P2P and
collective operations in a deadlock-safe order; CUDA events outside the node
ranges serialize successive work across streams. Same-stream serialization is
satisfied by CUDA FIFO rather than an extra event wait. Matching endpoints of
one communication transaction still rendezvous together.

This policy removes overlap between otherwise-independent node work while
retaining stream-specific kernel behavior and the actual runtime protocol. Its
CUPTI envelopes are used as the interference-free service measurements. A
routed communication node may internally issue several metadata and payload
collectives, but capture currently stores their combined envelope as one node
measurement.

### DAG replay

The CUDA executor and simulator consume the same lowered execution plan. For
each captured iteration, the simulator discards the observed node start times
and computes the earliest legal start of every node from:

- all explicit data and temporal DAG dependencies;
- finalized dispatch order and physical-stream FIFO order;
- the selected mode's rank-local CUPTI node durations; and
- communication readiness and completion semantics.

Ordinary and routed collectives (`all-reduce`, `all-gather`, `reduce-scatter`,
`A2A`, `M-to-N`, and `N-to-M`) use an all-participant rendezvous followed by an
endpoint-specific service tail derived from the aligned capture. "Symmetric"
here means that every participant must arrive; it does not require equal
payload sizes or equal endpoint durations. P2P transactions use a paired
send/receive rendezvous and the isolated transfer baseline, while preserving
distinct send and receive completion times. The simulated step time is the
makespan of the resulting event graph.

Replay accounts for the exact profiled DAG, placement, dependencies, stream
assignment, rank-local kernel envelopes, participant readiness, and measured
endpoint asymmetry. Normal replay uses normal-mode envelopes, in which hardware
interference is already present; interference-free replay uses the serialized
measurements. Their difference estimates interference/overlap overhead for
that captured workload and topology.

Replay is not a GPU or network resource simulator. It does not independently
model SM or memory-bandwidth contention, network links/NIC capacity,
communicator occupancy, CPU launch latency, allocator behavior outside the
attributed CUDA ranges, or interference created by a different schedule.
Otherwise-independent streams overlap without resource limits. Routed nodes
are modeled with one entry rendezvous and one aggregate tail even when the
runtime issues multiple internal collectives, and dense-state P2P headers and
multiple tensors are not separate replay phases. Consequently, results should
not be extrapolated to a different topology, tensor shape, routing
distribution, or kernel selection without collecting new profiles.

The capture and merge path is implemented in `src/piper.py` and `src/actor.py`;
the shared lowering is in `src/execution_plan.py`; and replay is implemented in
`src/simulator.py`.

## Offline tools

From the repository root:

```bash
python scripts/profile_and_replay/simulate_step_times.py PROFILE.json
python scripts/profile_and_replay/export_node_profile_ratios.py PROFILE.json
python scripts/profile_and_replay/attribute_simulation_savings.py PROFILE.json
python scripts/profile_and_replay/visualize_simulation_savings.py \
  NORMAL_INPUT ATTRIBUTION.json --output savings-trace.json
```

The tools import production profile and simulator types from `src/`; they do
not contain a separate simulator implementation.
