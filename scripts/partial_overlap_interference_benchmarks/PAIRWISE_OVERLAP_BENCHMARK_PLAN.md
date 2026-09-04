# Pairwise and Selected Multi-Aggressor Overlap Benchmark Suite Plan

## 1. Purpose

This document specifies Benchmark Suite 1 in two layers:

- **Suite 1A:** controlled pairwise overlaps between two GPU kernel activities on different CUDA streams;
- **Suite 1B:** six selected multi-aggressor cases that test whether the pairwise results compose under realistic concurrency and kernel trains.

Both layers are directly motivated by overlap cases extracted from the eight Piper traces in `out/20260804_143055`, while varying only model-valid execution parameters so the results can generalize beyond that exact run.

The suite is intended to answer five questions:

1. How much does each kernel slow down when it overlaps another kernel?
2. How does slowdown change with the amount and temporal placement of overlap?
3. Which kernel and system characteristics explain the slowdown?
4. Which relationships remain stable as sequence length increases and, later, as the same cases are validated at larger distributed scale?
5. Can slowdown under multiple aggressors be predicted from pairwise measurements, or do concurrency width, ordering, gaps, and resource interactions introduce non-additive effects?

Each concurrent trial has a designated **victim** and either one aggressor or a defined aggressor group for scheduling and analysis. Every participant is timed, so the same trial also captures aggressor slowdown and joint efficiency.

The exhaustive trace inventory that motivates the suite is summarized in:

- `out/20260804_143055/kernel_overlap_report.md`
- `out/20260804_143055/kernel_overlap_summary.json`
- `out/20260804_143055/kernel_overlap_pair_types.csv`

## 2. Scope and constraints

### 2.1 In scope

- Only the Tier A pair families and selected multi-aggressor cases listed in Section 4 are swept.
- Kernels must come from real operations in the traced Piper model and schedule.
- The model architecture remains `Qwen3-30B-A3B-empirical1to1-L4`.
- Hidden dimensions, expert dimensions, number of experts, attention heads, layer structure, data types, and operator implementations remain consistent with the trace.
- Sequence length is the only model-shape knob used to change kernel duration.
- Collectives are real distributed operations with the payload implied by the model execution; payload sizes are not independently fabricated.
- NCCL runs with its library behavior defaults. The benchmark removes inherited
  behavior-changing `NCCL_*` overrides before communicator creation and records
  the resulting environment audit. Observability-only `NCCL_DEBUG` and
  `NCCL_DEBUG_*` variables may remain and are recorded separately. In
  particular, implicit launch ordering remains disabled (the default); the
  suite must not enable `NCCL_LAUNCH_ORDER_IMPLICIT`.
- Every pair is tested in exactly two dispatch modes:
  - victim dispatched first;
  - aggressor dispatched first.
- Every multi-aggressor case is likewise tested in exactly two group-level dispatch modes:
  - victim dispatched before the aggressor group;
  - aggressor group dispatched before the victim.
- The internal order, per-stream order, and model dependencies of a multi-aggressor group are preserved rather than permuted.
- Actual GPU execution order is measured but is not independently forced or swept.

### 2.2 Out of scope

- Tier B or speculative kernel pairs.
- Synthetic SM, memory, or network blocker kernels.
- Artificial padding or delay inside the measured victim or aggressor kernels.
- Independent tensor-dimension sweeps.
- Fabricated duration-ratio targets.
- Arbitrary NCCL payload sweeps disconnected from the model.
- Algorithm, protocol, channel-count, queue-depth, stream-priority, CUDA Graph, or eager-versus-graph sweeps in Suite 1.
- Multi-aggressor cases other than M01–M04 and M05a–M05b, including the 85-partner P2P-send extreme.
- Arbitrary permutations, repetitions, truncations, or synthetic enlargement of the selected aggressor groups.

## 3. Design principles

### 3.1 Preserve model-valid kernels

The benchmark runner should execute the corresponding model operation or Piper node, not attempt to launch a kernel by its exported symbol. Kernel selection, grid dimensions, and implementation may change naturally with sequence length or library autotuning; those changes must be recorded rather than overridden.

### 3.2 Do not manufacture duration ratios

Victim and aggressor duration ratios are outcomes of the model-valid operations at each sequence length. They are recorded as explanatory variables but never targeted directly.

For a given sequence length:

\[
R_{A/V} = \frac{T_{A,\mathrm{isolated}}}{T_{V,\mathrm{isolated}}}
\]

If a requested overlap percentage is impossible because the aggressor is too short, that point is marked infeasible. The aggressor must not be repeated, padded, or resized merely to hit the target.

For M01–M04 and M05a–M05b, record every component duration ratio and the isolated aggressor-group-span-to-victim ratio. None of these ratios is targeted directly, and the group is not modified to achieve a desired ratio.

### 3.3 Separate requested and realized overlap

Requested overlap is calculated from isolated durations and the intended launch offset. Realized overlap is calculated from measured GPU intervals.

For M01–M04 and M05a–M05b, requested and realized victim coverage use the union of all aggressor intersections. Per-aggressor and cumulative coverage are recorded separately so simultaneous aggressors do not get mistaken for additional elapsed coverage.

This distinction is mandatory because interference may lengthen either kernel, thereby increasing the realized overlap. Treating realized overlap alone as the independent variable would introduce reverse causality.

### 3.4 Use CUPTI timing as the concurrency ground truth

The primary timing run must preserve natural concurrent execution. CUPTI Activity records and CUDA API callbacks provide launch, correlation, stream, and GPU activity data without requiring Nsight Compute kernel replay. Hardware-counter collection is performed in separate diagnostic passes, as described in Section 9.

## 4. Tier A cases to sweep

The trace contained 22,603 unique steady-state cross-stream overlap pairs. The following cases cover the frequently observed and structurally important pair classes.

### 4.1 Compute versus communication

For compute cases, select trace-real dense and grouped GEMMs that showed meaningful duration and slowdown. Include one bandwidth-bound model kernel as a contrast where indicated; do not substitute a synthetic kernel.

| Case | Victim/aggressor families | Trace motivation | Required compute anchors |
|---|---|---:|---|
| A01 | Compute ↔ P2P send | 4,169 pairs | Dense GEMM, grouped GEMM, bandwidth-bound model kernel |
| A02 | Compute ↔ P2P receive | 1,891 pairs | Dense GEMM, grouped GEMM, bandwidth-bound model kernel |
| A03 | Compute ↔ A2A NCCL | 3,849 pairs | Dense GEMM, grouped GEMM, bandwidth-bound model kernel |
| A04 | Compute ↔ all-gather | 1,912 pairs | Dense GEMM, grouped GEMM, bandwidth-bound model kernel |
| A05 | Compute ↔ reduce-scatter | 549 pairs | Dense GEMM, grouped GEMM, bandwidth-bound model kernel |

Each arrow is bidirectional for analysis and scheduling. For example, A01 includes both a compute victim with a P2P-send aggressor and a P2P-send victim with a compute aggressor.

### 4.2 A2A auxiliary work versus communication or compute

The auxiliary family consists of real routing, packing, scan, sort, histogram, cat, and copy kernels executed inside Piper forward/backward A2A nodes. Prefer model-real instances that reach at least the millisecond regime at one or more planned sequence lengths. Inherently short auxiliary kernels may be retained as trace anchors, but they are not padded and are not assigned a fabricated duration bin.

| Case | Victim/aggressor families | Trace motivation |
|---|---|---:|
| A06 | A2A auxiliary ↔ A2A NCCL | 4,234 pairs |
| A07 | A2A auxiliary ↔ P2P send | 1,826 pairs |
| A08 | A2A auxiliary ↔ P2P receive | 908 pairs |
| A09 | A2A auxiliary ↔ compute | 1,232 pairs |
| A10 | A2A auxiliary ↔ all-gather | 274 pairs |
| A11 | A2A auxiliary ↔ reduce-scatter | 68 pairs |

### 4.3 Communication versus communication

| Case | Victim/aggressor families | Trace motivation |
|---|---|---:|
| A12 | A2A ↔ P2P send | 270 pairs |
| A13 | A2A ↔ P2P receive | 144 pairs |
| A14 | A2A ↔ all-gather | 230 pairs |
| A15 | A2A ↔ reduce-scatter | 18 pairs |
| A16 | All-gather ↔ P2P send | 160 pairs |
| A17 | All-gather ↔ P2P receive | 209 pairs |
| A18 | Reduce-scatter ↔ P2P send | 16 pairs |
| A19 | Reduce-scatter ↔ P2P receive | 16 pairs |
| A20 | P2P send ↔ P2P receive | 191 pairs |

Low occurrence count does not automatically make A15, A18, or A19 unimportant. Reduce-scatter had no strictly isolated occurrences in the trace, making controlled pairwise measurement especially valuable.

### 4.4 Selected multi-aggressor cases

Multi-aggressor cases are not an exhaustive replay of the trace. They are a bounded validation layer selected to cover simultaneous aggressors from different stream roles, communication-on-communication interference, and a sequential train from one stream. In the steady-state trace, 4,697 original kernels had multiple overlapping partners and 3,878 had partners on more than one stream, so pairwise results alone are not sufficient evidence that interference composes.

| Case | Victim and aggressor group | Steady-state trace motivation | Representative trace anchor | Primary question |
|---|---|---:|---|---|
| M01 | Dense GEMM ← P2P receive + all-gather | 272 occurrences of this two-aggressor shape; 269 fully covered | `d1.p0.i2.e01393`: 3.679 ms GEMM; receive covers 100% and all-gather covers the leading 55.51% | Does a partial DP collective add interference beyond a persistent PP receive, and is that increment predicted by the corresponding pairwise cases? |
| M02 | Dense GEMM ← P2P send + P2P receive | 208 fully covered occurrences | `d6.p1.i2.e00371`: 4.988 ms GEMM fully covered by both P2P kernels | How do two simultaneous PP communication kernels affect a model-real GEMM compared with either P2P kernel alone? |
| M03 | Compute ← all-gather + A2A | 108 fully covered occurrences | `d1.p0.i2.e00757`: 0.457 ms compute kernel fully covered by DP all-gather and EP A2A | Do simultaneous collectives from different parallelism groups create non-additive compute interference? |
| M04 | A2A collective ← P2P send + P2P receive | 48 fully covered occurrences | `d3.p1.i3.e00377`: 8.422 ms A2A fully covered by both P2P kernels | How do concurrent PP transfers affect an EP communication victim, including network progress and GPU-side NCCL contention? |
| M05a | All-gather ← sequential two-kernel compute train on one stream | 37 occurrences; the most common all-gather/compute train length | `d5.p0.i2.e01219`: 1.510 ms all-gather, maximum simultaneous width 1, and 99.75% union coverage; the two compute kernels cover 93.18% and 6.56% of the victim | Can pairwise interference predict a short train whose two aggressors hand off near the end of the victim? |
| M05b | All-gather ← sequential three-kernel compute train on one stream | 35 occurrences; the second-most-common train length | `d2.p0.i2.e00449`: 2.625 ms all-gather, maximum simultaneous width 1, and 99.75% union coverage; the three compute kernels cover 0.05%, 6.22%, and 93.48% of the victim | Does adding a third, differently sized compute aggressor introduce order- or boundary-dependent effects not predicted by the component pairs? |

M03 is retained because the DP-plus-EP topology is structurally important even though its trace-anchor victim is shorter than 1 ms. It is labeled `natural_short_kernel` at the anchor sequence length and contributes to millisecond-duration response claims only at sequence lengths where the unchanged model operation naturally reaches at least 1 ms.

The representative event IDs identify concrete evidence in `kernel_overlap_cases.jsonl`; the benchmark should recreate the corresponding model operations and stream relationships rather than directly launch exported kernel symbols or hard-code rank-specific timestamps.

### 4.5 Exact kernel execution contract

The finalized assignments resolve to the unabridged launch names, SHA-1
identities, trace signatures, payloads, event IDs, and distributed participant
requirements in
`out/20260809_pairwise_overlap_foundation/EXACT_KERNEL_EXECUTION_MATRIX.md`
(with machine-readable `.json` and `.csv` companions). Calibration must reject
a silent kernel-name or signature change; an intentional implementation change
is reported as a separately calibrated specialization.

## 5. Sequence-length and duration plan

### 5.1 Sequence length is the sole duration-control knob

Start with the trace sequence length and increase it while keeping all model dimensions and execution semantics fixed. The initial candidate levels are:

- 12,288 tokens: trace anchor;
- 18,432 tokens: 1.5× anchor;
- 24,576 tokens: 2× anchor;
- 36,864 tokens: 3× anchor;
- 49,152 tokens: 4× anchor.

Before the full sweep, a calibration run must verify that each level is supported by the model configuration, fits memory, and does not trigger an unintended implementation change that invalidates comparison. Unsupported levels are removed rather than replaced by arbitrary tensor-shape edits.

Batch size, microbatch count, model width, expert configuration, and parallel placement remain fixed unless a memory constraint makes a sequence length impossible. A memory-driven change to any of those values constitutes a different configuration and must not be silently mixed into this suite.

### 5.2 Duration reporting

There is no requested duration-ratio sweep. After isolated calibration, results are grouped descriptively by the naturally measured victim duration:

- 1–2 ms;
- 2–5 ms;
- 5–10 ms;
- 10–25 ms;
- 25–50 ms;
- at least 50 ms.

The lowest planned duration range is therefore in milliseconds. A Tier A auxiliary or communication kernel that naturally remains below 1 ms is still a valid trace-real observation, but it is labeled `natural_short_kernel` and excluded from claims about the millisecond duration response.

No operation is repeated inside one measured kernel, padded with a wait loop, or resized independently to force it into a duration range.

### 5.3 Collectives that do not scale with sequence length

Some communication payloads, especially parameter all-gathers or gradient reduce-scatters, may be determined primarily by parameter size and remain nearly invariant as sequence length changes. That is a valid result. The suite records the naturally constant payload and does not enlarge it to match the growth of compute, P2P activation traffic, or A2A traffic.

## 6. Overlap sweeps

### 6.1 Victim coverage targets

For every feasible Tier A orientation and sequence length, request the following percentages of the isolated victim duration:

- 0%;
- 1%;
- 5%;
- 10%;
- 25%;
- 50%;
- 75%;
- 90%;
- 100%.

The 1% and 5% points preserve the sliver-overlap regime observed in the trace. Because the kernels are in the millisecond regime when possible, these points correspond to measurable tens or hundreds of microseconds rather than sub-microsecond offsets.

### 6.2 Dispatch modes

Only two modes are tested:

1. **Victim dispatched first.** The aggressor is launched later to produce a trailing overlap with the victim.
2. **Aggressor dispatched first.** The victim is launched later to produce a leading overlap with the victim.

Both streams use equal priority and begin without unrelated prequeued work. The runner uses ordinary eager launches for this suite. No queue-depth, priority, graph, or deliberately inverted execution-order variants are introduced.

GPU execution order is recorded. If it differs from dispatch order naturally, the sample remains valid and is labeled with an inversion flag; execution order is not forced.

### 6.3 Feasibility

For target victim coverage `p`, the required overlap time is:

\[
T_{\mathrm{requested}} = p \cdot T_{V,\mathrm{isolated}}
\]

The point is feasible only if the isolated aggressor can supply that much overlap under the selected dispatch mode. Otherwise the point is recorded as `infeasible_natural_duration` and skipped.

The runner must never change kernel dimensions, repeat the aggressor, or add a measured delay kernel to make an infeasible point feasible.

### 6.4 Scheduling mechanism

The harness should calculate a requested host launch offset from the most recent isolated baselines, launch the two real operations on separate equal-priority streams, and record the realized GPU intervals with CUPTI.

Host-side timing will have jitter. The analysis therefore uses the requested point for experimental assignment and the CUPTI-derived realized overlap for response-curve plots. Samples whose realized geometry falls outside a predefined tolerance are not silently discarded; they are marked and included in a sensitivity analysis.

### 6.5 Multi-aggressor scheduling

For M01–M04 and M05a–M05b, the aggressors form one scheduling group. The harness may translate that complete group relative to the victim to vary requested victim coverage, but it must preserve:

- the model-derived operations and data dependencies;
- each stream assignment;
- dispatch order within every stream;
- dispatch gaps between aggressors, modulo unavoidable host jitter;
- the natural durations and payloads produced at the selected sequence length.

The group uses the same requested victim-coverage targets listed in Section 6.1 whenever those targets are feasible for the unmodified sequence.

The two dispatch modes are defined at group level: victim before the group, or group before the victim. Individual aggressor permutations are not swept. Actual global dispatch order and GPU execution order for every participant remain measured outcomes.

Requested coverage is based on the union of aggressor intervals predicted from isolated calibration, while realized coverage is the union of measured GPU intersections. The complete selected group is always executed; aggressors are not repeated, removed, or padded to hit a requested point. Infeasible points are labeled `infeasible_natural_sequence`.

For M01–M04, matching victim-plus-one-aggressor trials from Suite 1A provide the component pairwise controls. M05a and M05b use their complete model-real compute trains plus constituent pairwise signatures. No other train lengths are swept.

## 7. Trial protocol

For each pairwise case or selected multi-aggressor case, orientation, sequence length, dispatch mode, and feasible coverage target:

1. Initialize the model-derived tensors and distributed process groups outside the timed region.
2. Warm up each operation until kernel selection and isolated timing stabilize.
3. Measure isolated victim samples.
4. Measure isolated aggressor or aggressor-group samples.
5. Execute concurrent treatment samples.
6. Re-measure the isolated victim and aggressor or aggressor group to quantify drift.
7. Randomize the order of isolated and concurrent blocks across repetitions.
8. Validate outputs and collective completion.

For M01–M04 and M05a–M05b, isolated calibration includes the victim, every participant as it appears in the model-real sequence, and the aggressor sequence without the victim. Concurrent treatment timing includes every participant and the complete group makespan. Component pairwise controls are reused from Suite 1A when configuration and realized geometry match; otherwise the required component control is collected as a diagnostic trial.

Use adaptive sampling:

- at least 50 valid samples for long operations;
- at least 100 valid samples for 1–10 ms operations;
- continue until the median slowdown confidence interval is within 1–2%, or until a configured maximum sample count is reached.

Allocations, compilation, autotuning, communicator creation, and model initialization must not occur inside a measured region.

## 8. Required metrics

### 8.1 Configuration and provenance

- Run, configuration, trial, and sample identifiers.
- Git commit and dirty-worktree indicator.
- Complete pre-sanitization and post-sanitization `NCCL_*` environment audits,
  with retained debug variables separated from removed behavior overrides,
  plus a pass/fail assertion that no behavior-changing NCCL override reached
  communicator creation.
- GPU model, CUDA driver/runtime, NCCL, PyTorch, and compiler versions.
- Host, node, GPU, rank, local rank, PP rank, DP group, and EP group.
- Model identifier and all fixed model dimensions.
- Sequence length, batch size, and microbatch count.
- Case ID, victim/aggressor orientation or victim/group role, and dispatch mode.
- Requested coverage and requested host launch offset.
- Tensor shapes, strides, layout, and dtype for every participant operation.
- Piper node/task identifiers.
- Kernel name, grid, block, dynamic shared memory, registers, and stream.
- Collective type, communicator, group size, endpoint ranks, payload elements/bytes, datatype, NCCL algorithm/protocol when observable, and direction.

### 8.2 Host dispatch

For the victim and every aggressor:

- CUDA launch or collective API entry timestamp.
- CUDA launch or collective API exit timestamp.
- API duration.
- Host thread.
- CUPTI correlation and external IDs.
- Dispatch order.
- Dispatch timestamp delta.
- Requested-versus-actual host dispatch error.
- Aggressor-group membership, index, and per-stream predecessor.
- Inter-aggressor dispatch gaps and preservation error relative to the calibrated sequence.

### 8.3 GPU timing

For every participant:

- GPU start timestamp.
- GPU end timestamp.
- GPU duration.
- CUDA stream and context.
- GPU execution-start order.
- GPU completion order.
- Launch-to-GPU-start delay.
- Start-time and end-time deltas relative to the victim.
- Dispatch/execution-order inversion flag.
- Number and identities of other participants active at its start and completion.

### 8.4 Isolated baselines

For every participant, and for the complete aggressor sequence in M01–M04 and M05a–M05b:

- Pre-treatment isolated samples.
- Post-treatment isolated samples.
- Median pre- and post-treatment duration.
- Interpolated paired baseline used for the treatment sample.
- Baseline drift in microseconds and percent.
- Mean, median, standard deviation, MAD, p5, p25, p75, p95, and p99.

### 8.5 Overlap geometry

- Requested overlap duration and victim percentage.
- Realized pair intersection duration.
- Realized overlap percentage of the victim.
- Realized overlap percentage of the aggressor.
- Per-aggressor intersection duration and percentage of the victim.
- Union overlap duration and percentage across all aggressors.
- Cumulative aggressor overlap, which may exceed the victim duration when aggressors are simultaneous.
- Time and fraction of victim duration spent at aggressor concurrency widths 0, 1, 2, and 3.
- Maximum simultaneous aggressor count and distinct partner-stream count.
- Inter-aggressor GPU execution gaps and per-stream execution sequence.
- Overlap start and end as percentages into each kernel.
- Interval relationship: victim leads, aggressor leads, victim contains aggressor, aggressor contains victim, or coincident.
- Requested-versus-realized overlap error.
- Additional realized overlap attributable to concurrent duration inflation.
- Whether the target was feasible.
- Whether realized geometry was within tolerance.

### 8.6 Interference outcomes

For each participant:

\[
S = \frac{T_{\mathrm{concurrent}}}{T_{\mathrm{isolated}}}
\]

Collect:

- Concurrent duration.
- Absolute duration increase.
- Slowdown ratio.
- Slowdown percentage.
- Confidence interval of median slowdown.

For the pair or multi-aggressor group:

- Concurrent pair or group makespan.
- Isolated serialized-work baseline: `T_victim + T_aggressor` for a pair, or `T_victim + sum(T_aggressor_i)` for a group.
- Isolated aggressor-group schedule span with its internal order and gaps preserved.
- Ideal-overlap baseline: `max(T_victim, T_aggressor)` for a pair, or `max(T_victim, T_aggressor_group_span)` for a preserved group.
- Makespan inflation over ideal overlap.
- Time saved relative to isolated serial execution.
- Sum-of-duration work inflation.
- Maximum participant slowdown.
- Geometric-mean participant slowdown.
- Slowdown imbalance/fairness.
- Operation-specific throughput, such as GEMM work per second or collective bytes per second.
- Pairwise-model prediction of multi-aggressor victim slowdown.
- Multi-aggressor interaction residual: observed slowdown minus the pairwise-model prediction.
- Whether pairwise predictions preserve the ordering of M01–M04 and M05a–M05b by observed slowdown.

### 8.7 Distributed communication

- Per-rank collective API start and completion.
- Per-rank NCCL GPU-kernel start and completion.
- Endpoint arrival skew.
- Earliest/latest rank start and completion.
- Collective wall duration and local GPU-kernel duration.
- Effective algorithmic and bus bandwidth where meaningful.
- Network/NVLink/PCIe bytes and throughput when available.
- NCCL warnings, asynchronous errors, retries, and completion status.

### 8.8 Runtime health and correctness

- GPU SM and memory clocks.
- Power and temperature.
- Clock or power throttling reasons.
- Memory allocated, reserved, and free.
- ECC or hardware errors.
- Trial wall time.
- Rank synchronization failures or timeouts.
- Output checksum and numerical correctness result.
- Invalid/dropped trace records and CUPTI buffer drops.

### 8.9 Statistical outputs

For every primary metric:

- valid and invalid sample counts;
- mean and median;
- standard deviation and MAD;
- p5, p25, p50, p75, p95, and p99;
- bootstrap 95% confidence interval;
- minimum and maximum;
- baseline drift;
- execution-order inversion rate;
- realized-overlap distribution within each requested coverage point.

## 9. CUPTI and Nsight Compute strategy

### 9.1 Nsight Compute is not required for primary metrics

The primary timing, dispatch, overlap, and slowdown metrics do not require Nsight Compute. They should be collected with:

- CUPTI Activity API for CUDA kernel, runtime/driver API, and memory activity records;
- CUPTI Callback API for API entry/exit and correlation;
- CUDA events only as an internal cross-check, not as the sole source of cross-stream ordering;
- NVTX ranges for case and sample identity;
- NCCL/application logging for collective semantics;
- NVML or DCGM for clocks, power, temperature, and health;
- fabric-specific tools or counters for NIC/network traffic.

The CUPTI documentation explicitly distinguishes Activity/Callback tracing of CUDA APIs and kernels from hardware performance profiling.

### 9.2 Hardware metrics can also be collected with CUPTI

CUPTI provides three relevant mechanisms:

1. **Range Profiling API:** collects hardware counters for a user-defined range. A range can span both overlapping kernels and preserves asynchronous launches, but its metrics describe the range as a whole rather than uniquely attributing shared activity to each concurrent kernel.
2. **PM Sampling API:** periodically samples GPU performance monitors during natural execution. This is useful for low-overhead time-varying SM, memory, and other supported hardware signals, subject to sampling resolution and metric availability.
3. **PC Sampling API:** samples warp program counters and scheduler states for selected diagnostic questions.

Hardware counter capacity is limited. Some requested metric sets require multiple passes or application replay. Such data must be collected in separate diagnostic executions, not mixed into the primary timing samples.

### 9.3 Recommended collection modes

#### Mode A: timing corpus — required for every sweep point

- CUPTI Activity and Callback tracing.
- No hardware-counter replay.
- Natural concurrent execution.
- Produces the authoritative duration, order, overlap, and slowdown dataset.

#### Mode B: CUPTI hardware diagnostics — required for representative points

Select at least these points for every Tier A family and every selected multi-aggressor case at each sequence length:

- isolated victim;
- isolated aggressor;
- 0% control;
- approximately 50% realized overlap;
- highest feasible realized overlap;
- the point with maximum observed slowdown.

Collect a compact single-pass counter set where possible. If multiple passes are required, use deterministic application replay and label the data as a diagnostic replay rather than timing ground truth.

For isolated kernels, collect per-kernel range metrics. For a concurrent treatment, collect pair- or group-range aggregate metrics and PM-sampled timelines. Do not claim per-kernel attribution for counters that only describe the shared range.

#### Mode C: optional Nsight Compute deep dive

Nsight Compute is optional and should be reserved for a small number of isolated kernels or anomalous cases where its sections, rules, and reports materially simplify diagnosis. Standard per-kernel profiling may serialize concurrent kernels, replay them, alter cache state, or lock clocks, so its durations must not replace Mode A timing.

If Nsight Compute is used on a concurrent pair or group, use a range/application-range mode that supports concurrency, disable unintended cache/clock changes where appropriate, and treat results as range-level diagnostics.

### 9.4 Metric attribution limitations

Neither CUPTI range counters nor Nsight Compute can generally assign a shared SM, L2, or DRAM counter event unambiguously to one of multiple simultaneously executing kernels. The plan therefore uses:

- isolated per-kernel hardware signatures;
- concurrent pair- or group-range aggregate counters;
- PM-sampled time series;
- per-kernel CUPTI timing;
- statistical comparison across overlap points.

This is sufficient to test whether, for example, a pair or group with high isolated Tensor Core demand slows more under overlap, without pretending that every concurrent hardware event has exact per-kernel ownership.

## 10. Goals of the former compute-resource and communication sweeps

### 10.1 Former point 5: compute-resource sweep

The original goal was to identify whether slowdown is caused by SM residency, Tensor Core contention, register/shared-memory limits, or DRAM/L2 pressure rather than by kernel name alone.

That goal remains important, but Suite 1 no longer fabricates resource levels with synthetic kernels. Instead it:

- characterizes the real Tier A kernels in isolation with CUPTI counters;
- observes how their resource signatures change naturally with sequence length;
- correlates those signatures with pairwise slowdown;
- uses different real Tier A compute families—dense GEMM, grouped GEMM, bandwidth-bound compute, and A2A auxiliary kernels—as naturally occurring contrasts.

A synthetic causal resource sweep can be a separate later suite if the Tier A results cannot distinguish competing explanations.

### 10.2 Former point 6: communication sweep

The original goal was to distinguish local GPU contention from network progress, communicator scheduling, and peer-arrival effects. That distinction is necessary because a long NCCL kernel can spend substantial time waiting rather than continuously consuming GPU or network resources.

Suite 1 retains this goal without arbitrary payload sweeps. It:

- executes the real Tier A collectives;
- uses payloads generated by the fixed model and selected sequence length;
- records group, payload, topology, endpoint timing, and network counters;
- measures per-rank arrival skew rather than intentionally sweeping it;
- keeps the traced DP8/EP8/PP2 placement for the initial suite.

Running the same Tier A cases at larger group sizes or different rank placement is a later validation phase. It should reuse this schema and keep model dimensions fixed, but it is not part of the initial Suite 1 sweep.

## 11. Data products

The runner should emit:

1. A manifest containing every requested configuration and feasibility result.
2. One participant row per kernel occurrence.
3. One pair row per concurrent treatment sample.
4. One group row per selected multi-aggressor treatment, including membership, sequence, union/cumulative overlap, and concurrency-width exposure.
5. One per-rank communication row for every distributed trial.
6. A CUPTI Activity trace for each retained sample or sampled subset.
7. Hardware-counter tables keyed by the same run/configuration/sample IDs.
8. Aggregated response curves by case, sequence length, dispatch mode, and requested/realized overlap.
9. A validation report covering correctness, CUPTI drops, baseline drift, timing stability, and missing sweep points.
10. A visualization bundle for every analysis product. At minimum this
    includes slowdown/overlap response curves, case-family heatmaps,
    launch/execution timelines, distributed endpoint-skew plots,
    sequence-length scaling plots, observed-versus-predicted multi-aggressor
    plots, and validation/health dashboards. Every figure is emitted with its
    source table and uncertainty/sample-count annotations; no conclusion is
    reported only as an unvisualized table.

The pair-level schema should remain compatible with the existing trace-derived fields in `kernel_overlap_pairs.csv`. The group-level schema should remain compatible with `kernel_overlap_events.csv` and `kernel_overlap_cases.jsonl`, especially their dispatch/execution sequences, union coverage, cumulative coverage, partner counts, stream counts, and maximum simultaneous-partner fields.

## 12. Execution phases

### Phase 1: operation extraction and calibration

- Implement model-real launch functions for A01–A20 and group launch functions for M01–M04 and M05a–M05b.
- Confirm tensor shapes, payloads, kernel identities, and communicators against the trace.
- Measure isolated duration at every candidate sequence length.
- Mark unsupported sequence lengths and infeasible overlap points.
- Select the exact dense GEMM, grouped GEMM, bandwidth-bound, and A2A auxiliary anchors.

### Phase 2: timing corpus

- Run both orientations of A01–A20.
- Run both dispatch modes.
- Sweep every feasible requested overlap point.
- Collect Mode A CUPTI timing and required runtime metrics.
- Run M01–M04 and M05a–M05b with their preserved internal orders, both group-level dispatch modes, and every feasible group-coverage point.
- Collect component pairwise controls needed to evaluate multi-aggressor non-additivity.

### Phase 3: hardware diagnostics

- Select representative and maximum-slowdown points.
- Collect isolated per-kernel CUPTI range metrics.
- Collect concurrent range aggregates and PM-sampled timelines.
- Use optional Nsight Compute only for unresolved anomalies.

### Phase 4: analysis and suite acceptance

- Produce slowdown-versus-realized-overlap curves.
- Produce the corresponding visualization bundle for every analysis and link
  every figure to its machine-readable source rows.
- Compare leading and trailing overlap.
- Compare observed M01–M04 and M05a–M05b slowdown with predictions from the pairwise corpus.
- Quantify the effects of concurrency width, sequential train gaps, and heterogeneous aggressor groups.
- Quantify sequence-length effects without fitting fabricated duration ratios.
- Model slowdown using isolated kernel signatures, natural durations, payloads, dispatch mode, and realized geometry.
- Verify conclusions on held-out sequence lengths before proposing a larger-scale validation suite.

## 13. Acceptance criteria

Suite 1 is complete when:

- all feasible Tier A cases and orientations are represented;
- M01–M04 and M05a–M05b are represented with their specified participants, stream relationships, and preserved internal orders;
- all supported sequence lengths have isolated calibration;
- both dispatch modes and all feasible overlap targets are measured;
- timing traces have no unexplained CUPTI record loss;
- numerical and collective correctness checks pass;
- baseline drift is reported and bounded;
- median slowdown confidence intervals meet the configured threshold for primary points;
- CUPTI hardware diagnostics exist for the required representative points;
- every conclusion distinguishes requested overlap, realized overlap, and natural duration ratio;
- multi-aggressor conclusions distinguish union coverage, cumulative coverage, and time at each concurrency width;
- pairwise prediction error is reported for every feasible M01–M04 and M05a–M05b configuration;
- no result depends on fabricated kernel dimensions, payloads, duration ratios, or synthetic blockers.
- no communicator was created with a behavior-changing non-default `NCCL_*`
  override, including implicit launch ordering; retained `NCCL_DEBUG*`
  observability variables are reported but do not fail this gate;
- every reported analysis has a generated visualization with sample counts and
  uncertainty where applicable.

## References

- [NVIDIA CUPTI overview](https://docs.nvidia.com/cupti/13.0.0/index.html)
- [NVIDIA CUPTI Range Profiling and replay documentation](https://docs.nvidia.com/cupti/main/main.html)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
