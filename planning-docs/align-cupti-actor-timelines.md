# Align CUPTI actor timelines for collective replay

## Task summary

Align every actor's CUPTI/Kineto node occurrences to one shared per-iteration timeline so collective service can be separated directly from observed rendezvous waiting. Remove the unaligned-profile subtraction and negative-residual fallback rather than approximating malformed collective timings.

## Objectives

- Capture each profiled node occurrence's GPU activity start and end in an actor-local CUPTI/Kineto timeline without per-node CUDA timing events.
- Calibrate actor-local timelines to one shared coordinator timeline, including multi-node ActorGroups, and retain alignment uncertainty as profile metadata.
- Preserve occurrence identity through warmup, spare-sample rejection, profile merging, serialization, and replay.
- Derive each collective endpoint's observed post-rendezvous service tail from aligned timestamps exactly once, then combine that tail with the rendezvous time predicted by replay.
- Require aligned endpoint timings for profiled collective replay and fail clearly when alignment is absent, incomplete, or inconsistent.
- Remove the legacy rank-local wait subtraction, negative residual, and latest-arrival median fallback.

## Non-goals

- Using observed node starts as replay release constraints; replay remains governed by the finalized DAG, physical-stream FIFO, and communication rendezvous.
- Modeling the single-threaded host issue loop, Ray dispatch latency, profiler overhead, or other cross-node idle gaps. Timeline alignment fixes collective service decomposition but will not by itself eliminate the instrumented-wall-time underprediction.
- Changing DAG construction, schedule directives, actor placement, stream assignment, communicator selection, or collective implementations.
- Reintroducing per-node CUDA-event timing or legacy DP topology metadata.
- Changing ragged A2A semantics or the isolated directional-P2P transfer baseline unless alignment exposes a correctness defect that requires a separately approved scope change.

## Correctness criteria

- For every retained profile mode and iteration, all actors express node GPU start/end times in the same coordinate system and the artifact records how the alignment was obtained and its uncertainty.
- Node duration remains the correlated first-to-last GPU activity envelope, and aligned start plus duration reconstructs the corresponding endpoint end time.
- For every collective transaction and retained iteration, observed rendezvous is the maximum aligned endpoint start; every endpoint's service tail is its aligned end minus that rendezvous and is finite and non-negative within declared alignment uncertainty.
- Normal and interference-free replay use the same DAG dependencies, physical streams, and predicted rendezvous rule; only their profiled service samples differ.
- Replay never estimates collective service by subtracting replay-predicted waiting from an unaligned rank-local interval and never substitutes a median endpoint interval.
- A collective profile with missing actors, mismatched occurrences, absent alignment, or irreconcilable endpoint timestamps is rejected before producing a simulation result.
- Timeline alignment and collective decomposition are general over actor count, ActorGroup width, placement, collective kind, and single- or multi-node topology.
- Existing aligned-profile serialization and replay remain deterministic across save/load.

## Subtasks

1. Extract per-occurrence GPU activity start/end timestamps from existing Kineto attribution and associate them with retained profiler iterations.
2. Calibrate each actor's Kineto timeline to the coordinator timeline without instrumenting individual DAG nodes.
3. Merge, validate, filter, and serialize aligned offsets alongside duration samples.
4. Require aligned collective endpoints in simulation and delete the legacy fallback path.

## Assumptions / open questions

- Kineto's CPU ranges and correlated GPU activities share one actor-local trace coordinate system. Phase 2 will identify the smallest iteration-level host anchor needed to map that coordinate system through an NTP-style actor/coordinator clock calibration.
- Clock calibration must cover every active capture occurrence and expose uncertainty; an implementation that merely assumes host clocks are synchronized is insufficient.
- Historical unaligned artifacts may still deserialize, but any attempt to replay a rendezvous collective from them will fail rather than silently approximate service. Profiles without rendezvous collectives need not be rejected solely for lacking alignment.
- Alignment is used to measure collective service, not to reproduce observed scheduling delays. A separate host-dispatch model would require its own scope and approval.

## Implementation plan

1. **Calibrate every actor to the program driver's monotonic clock.**
   - Add a read-only actor RPC that returns the actor's `time.monotonic_ns()`.
   - In `src/piper.py`, immediately before each profiling mode, collect several driver/actor midpoint samples, select the minimum-round-trip sample, and pass its actor-to-driver offset and half-round-trip uncertainty into the CUPTI capture.
   - Perform this independently for every global rank. Do not assume hosts share a boot-time, wall-time, or NTP clock, and do not introduce DP coordinates or a persistent coordination actor.

2. **Anchor every captured Kineto iteration to the calibrated actor clock.**
   - In `src/actor.py`, wrap each profiled DAG occurrence in one uniquely named CPU `record_function` timeline anchor. Bracket entry into that range with actor-local monotonic timestamps and retain the midpoint plus uncertainty. This is one host-only anchor per iteration, not a timing event on a CUDA stream.
   - Extend `_kineto_node_attribution_audit` to identify those anchors, associate every node CPU range and its correlated GPU activities with exactly one captured iteration, and return the first/last GPU activity timestamp for each node occurrence.
   - Express each node start as an offset from that occurrence's Kineto anchor. Convert the paired host anchor to the driver's clock using the calibrated offset and store it in the existing `ActorProfileResult.timeline_origin_samples_us` fields.
   - Require one occurrence of every executable DAG node in each captured iteration. A collective occurrence with activities on any participant must have GPU timestamps on every participant. An all-participant zero-activity collective remains a zero-service occurrence; its aligned CPU node-range start is used only as the timestamp for that no-op occurrence, never as a substitute service measurement.

3. **Preserve alignment through filtering and profile merging.**
   - Populate `NodeProfile.start_offset_samples_us` from the Kineto activity starts while retaining the existing correlated GPU envelope as `samples_us`.
   - Update `_filter_kineto_profile_iterations` to filter timeline origins, origin uncertainties, start offsets, durations, and attribution records with the same retained occurrence indices.
   - Reuse `merge_actor_profiles` to normalize every actor occurrence against the earliest calibrated origin for that mode/iteration. Extend its alignment metadata with a conservative pairwise uncertainty bound and validate complete actor/mode/sample coverage.
   - Keep the serialized representation rank-keyed and topology-neutral. Save only normalized offsets and alignment metadata; actor-local absolute clocks remain merge-time data.

4. **Make aligned collective decomposition mandatory and exact.**
   - In `src/simulator.py`, require timeline alignment and aligned start offsets whenever a replay contains a rendezvous collective (`REDUCE`, all-reduce, all-gather, reduce-scatter, M-to-N, or N-to-M).
   - For each transaction occurrence, compute observed rendezvous as the maximum aligned endpoint start and endpoint service as `aligned_start + measured_envelope - observed_rendezvous`.
   - Reject service below the conservative alignment-error bound. Clamp only a non-positive residual whose magnitude is within that declared measurement uncertainty to zero; do not substitute another endpoint's interval.
   - During replay, combine the fixed measured endpoint service with the rendezvous determined by replay-predicted arrivals. Keep observed starts diagnostic-only and preserve all existing DAG/stream dependencies.
   - Delete the unaligned `profile_interval - replay_wait` path, interval-underflow warning, and latest-arrival median fallback. Historical plans containing rendezvous collectives but no alignment fail with an actionable validation error.
   - Preserve the existing directional-P2P contract: its explicitly isolated microprofile is already service time and is used after the predicted P2P rendezvous without collective wait decomposition. Ragged A2A remains non-rendezvous communication.

5. **Validate capture and replay inputs at the earliest useful boundary.**
   - In `src/profile_validation.py` and profile merge checks, reject missing anchors, duplicate/misassigned occurrences, mixed zero/nonzero collective activity, mismatched sample counts, incomplete participant coverage, non-finite timestamps, or alignment uncertainty that cannot support a non-negative service decomposition.
   - Report mode, iteration, collective ID, rank, aligned start/end, rendezvous, and uncertainty in failures so invalid profiles are diagnosable without simulator heuristics.

### Compatibility and implementation risks

- Existing profile artifacts without `timeline_alignment` will no longer replay rendezvous collectives. They remain deserializable and may still replay graphs without those collectives.
- Kineto trace timestamps are not in Python's `time.monotonic_ns()` epoch. The per-iteration `record_function` anchor is therefore required; applying the actor clock offset directly to raw Kineto timestamps would be incorrect.
- Actor/driver clock calibration measures scheduling asymmetry as uncertainty. The implementation must fail closed when uncertainty is too large to distinguish valid service from an inconsistent endpoint timeline.
- The current world6 artifact includes all-participant zero-activity reduction nodes. They carry zero service and require an aligned occurrence timestamp, but must not trigger the removed median fallback.
- Correct alignment may shorten collective service estimates that currently retain rendezvous waiting. The simulated-normal estimate can therefore move farther from instrumented wall time until host issue/release latency is modeled separately.

## Evaluation plan

1. **Kineto attribution and iteration anchoring** — verifies shared-coordinate node starts without per-node CUDA events.
   - Extend `test/test_actor_profile_timing.py` with synthetic Chrome traces covering multiple active occurrences, different node streams, zero-activity nodes, missing/duplicate anchors, and node ranges assigned to the wrong occurrence.
   - Verify extracted start offsets and durations use correlated GPU activity boundaries, while zero-activity occurrences use only their CPU range timestamp and zero service.
   - Verify the profiler path creates no per-node timing events and adds only one host anchor per profiled iteration.

2. **Clock calibration, filtering, and profile merge** — verifies arbitrary actor clocks and rejected spare samples remain aligned.
   - Add deterministic midpoint-calibration tests with asymmetric fake RPC latency and ensure the minimum-RTT offset and uncertainty are selected with the correct sign.
   - Extend `test/test_profiling.py` and `test/test_actor_profile_timing.py` for actors with deliberately different clock epochs, multiple modes, multiple retained/rejected occurrence IDs, serialization round-trips, and incomplete or inconsistent alignment metadata.
   - Confirm filtering applies the identical index set to duration samples, start offsets, timeline origins, uncertainties, and attribution signatures.

3. **Collective service decomposition and fail-closed behavior** — verifies exact aligned replay and removal of the fallback.
   - Extend `test/test_simulator.py` with parameterized collective kinds and actor widths. Use unequal aligned arrivals to verify observed service tails are reused with different replay-predicted arrival skews.
   - Cover alignment-error-bound normalization for a tiny negative residual and rejection beyond that bound.
   - Reject unaligned collective plans, missing endpoint offsets, mismatched sample IDs, incomplete participants, and mixed zero/nonzero collective activity.
   - Assert no interval-underflow fallback or warning remains and that a historical unaligned graph without rendezvous collectives still replays.
   - Retain focused P2P and ragged-A2A tests to prove their existing non-collective timing contracts are unchanged.

4. **Dependency and replay regression** — verifies alignment changes timing inputs, not execution semantics.
   - Run `test/test_execution_plan.py`, the relevant stream/dependency cases in `test/test_simulator.py`, `test/test_profiling.py`, `test/test_profile_validation.py`, and `test/test_actor_profile_timing.py`.
   - Re-run the saved-plan structural audit: execution-plan predecessors must equal finalized DAG edges, every physical-stream FIFO must remain reachable, and collective endpoint membership must be unchanged.

5. **Focused GPU integration** — verifies real CUPTI clocks on single- and multi-node ActorGroups.
   - Run one small conventional collective schedule and one small unequal-group AFD schedule through normal and interference-free profile-and-simulate. At least one run must place collective participants on different Ray nodes.
   - Verify both modes contain aligned offsets for every retained actor/node occurrence, report bounded calibration uncertainty, pass profile validation, produce no legacy underflow warning, and replay successfully.
   - Re-run the exact world6 AF-Pipe placement with reduced focused sampling, compare instrumented normal and simulated normal, and report separately the change caused by aligned collective service and the remaining host-issue gap. This is diagnostic; wall-time parity is not a completion criterion for this task.

6. **Completion checks**
   - Run `git diff --check` and Python compilation for modified files.
   - Record every command and result in this document. Do not run the full unit or E2E suites; the focused tests above cover the affected profiling, merge, and replay interfaces.

## Implementation log

### Implemented

- Added minimum-RTT driver/actor midpoint clock calibration and passed its offset plus uncertainty into each CUPTI capture.
- Added one uniquely named host-only Kineto anchor around every profiled DAG occurrence. Kineto attribution now associates every node occurrence with exactly one captured anchor and records correlated GPU activity start/end offsets; zero-activity nodes retain only their CPU-range occurrence timestamp and zero duration.
- Preserved aligned origins, uncertainties, node offsets, and occurrence records through spare-iteration filtering and profile merging. Merge now validates finite origins, uncertainty/sample counts, and consistent iteration IDs, and records a conservative pairwise uncertainty bound.
- Collective replay now requires aligned profiles, derives fixed post-rendezvous endpoint service from aligned activity envelopes, clamps only negative residuals within declared uncertainty, and rejects larger inconsistencies. The rank-local subtraction and latest-arrival fallback were removed. Directional P2P continues using its isolated transfer baseline; ragged A2A remains non-rendezvous communication.
- Added validation that each rendezvous-collective occurrence is either zero-activity on every endpoint or nonzero-activity on every endpoint. All-zero occurrences replay with zero service regardless of differing CPU-range starts.
- Split symmetric collectives from directional M-to-N/N-to-M transfers. Sources now complete from their own arrival and measured service, while each destination is gated by itself and every source. Role/device metadata is validated against transaction participants.
- Made savings-attribution profile selection atomic for every communication transaction, including routed A2A, M-to-N/N-to-M, and all-reduce, so a hybrid replay cannot splice aligned endpoints from different capture modes.

### Issues and resolutions

- Kineto trace timestamps do not share Python's `time.monotonic_ns()` epoch. Resolved by mapping activity offsets through a per-iteration CPU `record_function` anchor rather than applying the actor clock offset directly to raw Kineto timestamps.
- Existing world6 profiles contain reductions with no GPU activity on any participant. Treating their CPU timestamps as rendezvous timestamps would create false negative service; they are now explicitly modeled as all-endpoint zero-service occurrences.
- Historical unaligned collective unit fixtures reached unrelated validation paths first. They were migrated to explicit aligned fixtures where the intended assertion concerns collective structure, while a dedicated test now verifies that historical unaligned collective replay fails closed.
- Real AFD timestamps proved that M-to-N/N-to-M is not a symmetric barrier: send-only ranks can complete before receive-only ranks post. The simulator now uses the directional role contract instead of rejecting those valid traces.
- Attribution initially mixed one directional endpoint from normal profiling with its peer from interference-free profiling. Grouping every endpoint of a communication transaction into one attribution feature resolved the invalid hybrid timeline.
- Conventional interference-free profiling stalls because its actor-local global serialization chain adds cross-direction P2P dependencies that are absent from the DAG. The observed cycle is PP1 `RECV(F, MB1) -> SEND(B, MB0)` and PP0 `RECV(B, MB0) -> SEND(F, MB1)`. The synchronous route-count read in the following A2A only exposes the prior GPU-queue deadlock; it does not create it. Changing this serialization policy is outside this task.
- A read-only CUDA-GDB attach was unsupported by the installed driver and terminated one actor. That run was discarded; a separate two-minute dispatch-log run established the cycle above without changing repository code.

### Assumptions and remaining limitations

- A single minimum-RTT clock calibration is taken immediately before each profiling mode. The implementation assumes monotonic-clock drift during that mode is negligible relative to recorded calibration uncertainty; it does not interpolate between pre/post capture calibrations.
- Collective service is estimated from per-iteration paired endpoint measurements before applying the configured estimator, avoiding independently estimated start/duration residuals.
- Only a single-node eight-H100 allocation was available for final evaluation, so the planned multi-node calibration check could not be run.
- The conventional interference-free serializer deadlock above prevents validating the symmetric-collective workload end to end. No runtime or schedule workaround was added because it is orthogonal to timeline alignment and changes profiling execution semantics.

### Evaluation run so far

- `gpu-venv/bin/python -m py_compile src/actor.py src/piper.py src/profiling.py src/simulator.py` — passed.
- `gpu-venv/bin/python -m pytest -q test/test_profiling.py test/test_simulator.py test/test_actor_profile_timing.py test/test_profile_validation.py` — 72 passed after the latest focused additions.
- `gpu-venv/bin/python -m pytest -q test/test_afd_routed_directives.py test/test_afd_full_lowering.py test/test_execution_plan.py test/test_profile_streams.py test/test_dag_builder_equivalence.py test/test_profiling.py test/test_simulator.py test/test_actor_profile_timing.py test/test_profile_validation.py` — 147 passed before the final small validation additions; the directly affected four-file subset was rerun afterward and passed.
- `git diff --check` — passed at the recorded checkpoint.
- `gpu-venv/bin/python -m pytest -q test/test_afd_routed_directives.py test/test_afd_full_lowering.py test/test_execution_plan.py test/test_profile_streams.py test/test_dag_builder_equivalence.py test/test_profiling.py test/test_simulator.py test/test_simulator_attribution.py test/test_actor_profile_timing.py test/test_profile_validation.py` — 160 passed.
- `gpu-venv/bin/python -m py_compile src/actor.py src/piper.py src/profiling.py src/profile_validation.py src/simulator.py src/attribution.py` — passed.
- Saved AFD artifact `out/20260818_074102/step_time_profiles.json` — normal, IF, validation, and attribution replay passed after directional transaction selection became atomic.
- Fresh unequal-group AFD GPU capture — passed with three actors on one H100 node; artifact `out/20260818_080453/step_time_profiles.json`. Ordinary hardware median was `37.553 ms`, simulated normal `35.515 ms` (5.427% low), simulated IF `34.847 ms`, and aligned pairwise uncertainty was at most `469.269 us`. Kineto activity/timing and attribution-losslessness checks passed.
- Conventional `pp2_dp2_ep2_zero2` GPU capture — ordinary steps passed (`34 ms`, `31 ms`), then IF capture reproduced the serialization cycle documented above. A separate mode-only diagnostic reproduced the same cycle and hit its two-minute guard.

## Approved Phase 2 amendment — implemented

### GPU evidence requiring a model correction

The first real unequal-group AFD capture produced valid aligned timelines but rejected the existing symmetric-rendezvous interpretation of M-to-N/N-to-M. For the final backward N-to-M transaction in iteration 0:

- routed source rank 2: start `60194.892 us`, end `60201.197 us`;
- dense destination rank 0: start `62203.575 us`, end `62209.591 us`;
- dense destination rank 1: start `62241.739 us`, end `62248.075 us`;
- conservative pairwise clock uncertainty: `405.836 us`.

The source therefore completed roughly 2 ms before either receive-only destination posted its endpoint. Other transactions align normally, and the per-rank transaction order and `comm_id` sequence match, ruling out a clock-offset or communicator-order mismatch. This follows the implementation: cross-group routing lowers to a variable-split `all_to_all_single` in which source ranks have send-only splits and destination ranks have receive-only splits; NCCL may complete/buffer a send before its peer posts the receive. M-to-N/N-to-M is directional communication, not a symmetric all-participant barrier.

### Revised correctness criterion

- Symmetric collectives (`REDUCE`, all-reduce, all-gather, and reduce-scatter) retain the all-participant rendezvous: observed rendezvous is the maximum aligned endpoint start.
- Cross-group M-to-N/N-to-M uses the schedule/DAG `role` metadata. Every source is a send endpoint and every destination is a receive endpoint. A source completion is gated only by its own arrival and measured source service. Each destination completion is gated by its own arrival and every source arrival, but not by other destinations. Its observed rendezvous is `max(destination_start, all_source_starts)`, and its service is `destination_end - observed_rendezvous` within alignment uncertainty.
- Replay must preserve those same directional dependencies. It must not special-case a particular model tag, payload size, empty tensor, actor count, or motivating transaction.

### Minimal implementation amendment

1. Split symmetric collective and directional routed-transaction construction in `src/simulator.py`; keep common aligned occurrence/uncertainty validation.
2. For M-to-N/N-to-M, validate nonempty, disjoint source/destination role sets whose union matches `participants`; create independent source completion vertices and one receive rendezvous per destination, gated by all sources plus that destination.
3. Derive source service from its aligned activity envelope and destination service from the directional observed rendezvous. Preserve per-iteration pairing before estimating service across samples.
4. Add topology-neutral simulator tests in which a send endpoint completes before destinations arrive, destinations arrive at different times, and malformed/missing roles are rejected. Do not encode the exact AFD transaction or device ordinals.

### Risks / limitation

This model treats the profiled source envelope as empirical send service. NCCL backpressure for payloads that exceed eager buffering can make that envelope arrival-dependent; modeling such protocol capacity would require peer-level transport profiling and is outside this task. The directional model is nevertheless faithful to the observed endpoint dependency semantics and avoids imposing a barrier that the runtime does not provide.

### New evaluation observations

- Focused CPU suite after paired-service estimation: 152 passed; `git diff --check` passed.
- Single-node three-H100 AFD profile captured two IF and two normal iterations and produced aligned metadata for all three actors. Replay failed closed on the directional transaction above, which motivated this amendment; artifact: `out/20260818_074102/step_time_profiles.json`.
- A four-H100 conventional `pp2_dp2_ep2_zero2` run completed ordinary warmup/timed steps but stalled during CUPTI profiling and hit the 10-minute guard before producing an artifact. This is recorded as an evaluation limitation; no scheduling/runtime workaround was introduced.

## Remaining known limitations

- Clock drift is bounded only by one minimum-RTT calibration immediately before each profile mode; no post-capture interpolation is performed.
- Directional source service is empirical. Payload-dependent NCCL backpressure is not modeled separately from the captured source envelope.
- Interference-free profiling cannot currently execute schedules whose actor-local total serialization orders induce a cross-direction P2P cycle. This affects the conventional validation case above but not the unequal-group AFD acceptance run.
- Final multi-node GPU validation was unavailable on the current single-node allocation.

## Commit message

`Align CUPTI actor timelines for directional collective replay`
