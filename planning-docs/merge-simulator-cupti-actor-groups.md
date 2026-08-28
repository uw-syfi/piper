# Merge simulator CUPTI profiling with ActorGroups

## Task summary

Resolve the current `origin/simulator` merge by integrating CUPTI/Kineto node profiling and replay into AFD while retaining explicit ActorGroup topology, routed communication streams, and ActorGroup collective semantics.

## Objectives

- Retain ActorGroups as the sole runtime and coordination topology; remove merged uses of legacy `dp_rank`, `dp_degree`, `dp_group`, and `PIPER_DP_*` metadata.
- Replace per-node CUDA-event timing with CUPTI/Kineto range attribution for profile-and-simulate and replay.
- Support configurable CUPTI warmup and active iterations. Direct trace capture defaults to two warmup iterations followed by capture of the third iteration; profile-and-simulate sampling retains its established `5/30` default.
- Preserve every DAG node's expected logical CUDA stream during profiling.
- Make interference-free profiling serialize nodes across logical streams while preserving a globally consistent DAG-valid execution order across ActorGroups.
- Preserve dedicated M-to-N and N-to-M streams and their normal execution semantics.
- Preserve ActorGroup gradient synchronization and averaging while incorporating the reduce-scatter temporary-allocation stream-lifetime fix.
- Reconcile example drivers and tests so profiling, visualization, dynamic inputs, model seeds, routed communication, and ActorGroup placement continue to work together.
- Ensure CUPTI-derived node timings feed replay/simulation with the same DAG identities, dependencies, actor placement, and stream assignments used by execution.

## Non-goals

- Reintroducing rectangular DP topology or true data-parallel replication metadata.
- Changing AFD placements, routed communication algorithms, or schedule ordering.
- Redesigning simulator interference models or benchmark datasets beyond what is required to integrate the merged CUPTI path.
- Adding extra ordering constraints solely to prevent a hypothetical interference-free profiling deadlock; if straightforward dependency-respecting serialization deadlocks, stop and report it for joint debugging.
- Broad cleanup of unrelated simulator-branch changes.

## Correctness criteria

- Runtime, coordinator, profiling output, and replay contain no dependency on legacy DP rank/degree/group metadata.
- All actors in explicit ActorGroups participate in profiling and produce one coherent profile consumable by replay and simulation.
- Default direct trace profiling performs two unrecorded warmup iterations and records exactly the third iteration; callers can configure both warmup and active counts. Default profile-and-simulate sampling remains five warmup and thirty active iterations.
- CUPTI attribution maps every profiled DAG node to the correct actor, node identity, task type, logical stream, and measured GPU work without per-node CUDA-event timing.
- Normal profiling preserves concurrent execution on the DAG-assigned streams; interference-free profiling uses those same stream assignments but measures nodes in dependency-valid serialized order consistently across ActorGroups.
- Dedicated routed streams remain represented faithfully in captured profiles and replay.
- Reduce-scatter uses the ActorGroup synchronization communicator, averages exactly once across replicas, and safely allocates/uses its temporary across the compute and collective streams without recurring allocator-cache growth.
- Existing ActorGroup, routed-DAG, CUPTI attribution/replay, profiling-timing, stream, and reduce-scatter behavior covered by focused tests remains correct after conflict resolution.
- No unresolved merge conflicts or stale references to removed DP profiling APIs remain.

## Subtasks

1. Resolve topology, actor lifecycle, and profile data-store conflicts around ActorGroups.
2. Port CUPTI capture, attribution, profile aggregation, and replay away from DP metadata.
3. Reconcile logical-stream and interference-free execution behavior, including routed streams.
4. Merge reduce-scatter allocator-stream handling with ActorGroup synchronization and averaging.
5. Reconcile example-driver behavior and conflicting tests, then audit auto-merged code for legacy metadata.

## Assumptions / open questions

- ActorGroup IDs and global actor ranks provide sufficient stable identity for profile aggregation and replay; no DP-shaped coordinate is required.
- CUPTI ranges and DAG metadata provide the authoritative node-to-kernel attribution; CUDA events may still be used for runtime dependency/lifetime correctness, but not as the node-duration profiling source.
- Interference-free serialization can follow a deterministic dependency-valid node order shared by all ActorGroups without adding schedule edges. If collective launch ordering makes this deadlock, implementation will pause and report the exact ordering failure.
- Simulator-branch changes unrelated to the conflicted profiling path will be retained unless they violate ActorGroup semantics or fail focused verification.

## Implementation plan

1. **Resolve topology and ownership around one ActorGroup program driver.**
   - In `src/coordinator.py` and `src/compile.py`, retain explicit schedule-derived ActorGroups, one driver owning all actors, exact global-rank membership, and the ActorGroup placement plan.
   - Do not retain profile generations, per-DP shards/barriers, rectangular placement, or `PIPER_DP_*` environment state. Keep only compiled-data-store functionality still required by the ActorGroup compile/runtime path.
   - In `src/piper.py`, launch every profiling iteration on all actors from the one driver and merge results directly by `(profile_mode, global_rank)`.

2. **Make CUPTI/Kineto the DAG-node timing path.**
   - In `src/actor.py`, reconcile the profiler lifecycle into one configurable warmup/active API and make two warmup iterations plus one active iteration the direct trace-capture default.
   - Retain `record_function`/NVTX node ranges and Kineto correlation-based attribution. Remove the legacy per-node CUDA-event recorder, its clock-calibration/start-offset state, and the CUDA-event implementation of `piper_profile_dags`.
   - Keep ordinary CUDA events required for DAG dependencies, buffer lifetimes, and profiling-only cross-stream serialization; these are synchronization mechanisms rather than node-duration instrumentation.
   - Preserve the existing isolated directional-P2P transfer baseline if required to exclude rendezvous waiting. It runs outside DAG-node capture and does not surround workload nodes with timing events.
   - Name trace selection and artifacts by global actor rank, with no DP coordinate.

3. **Preserve logical streams and serialize only the interference-free policy.**
   - In `src/runtime.py`, use the normal execution stream for every profile mode, including dedicated directional P2P, dense-state, M-to-N, and N-to-M streams.
   - In `src/executors.py`/`src/actor.py`, retain the profiling-only event handoff outside each node range for interference-free capture. Normal CUPTI capture remains concurrent.
   - Use the prepared DAG's deterministic topological dispatch order on each ActorGroup. Do not add schedule edges or remap communication to a common stream. If collective rendezvous deadlocks under this direct serialization, record the failing node order and stop for user direction.

4. **Port CUPTI aggregation, validation, and replay to ActorGroups.**
   - In `src/piper.py`, collect all rank results centrally, retain complete iteration indices shared by every actor/mode, validate Kineto launch attribution and normal-versus-interference-free activity equivalence, and merge them into the finalized ActorGroup DAGs.
   - In `src/profiling.py`, keep rank-keyed `ActorProfileResult`/`NodeProfile` artifacts free of DP fields and retain `ProfileConfig` defaults of five warmup and thirty active iterations while preserving readable legacy timing-source values in stored artifacts.
   - In `src/simulator.py`, `src/attribution.py`, and `src/profile_validation.py`, retain CUPTI-derived durations and ensure simulation obtains actor ranks, node placement, dependencies, and logical streams exclusively from the saved DAG/topology. Remove output gating or layout assumptions based on DP ranks.
   - Retain unified driver wall-time samples for instrumented and optional uninstrumented modes; convergence decisions, when enabled, use the single driver's global step samples.

5. **Combine the reduce-scatter allocator fix with ActorGroup semantics.**
   - In `src/executors.py`, allocate the reduce-scatter scratch tensor on Piper's logical default/compute stream, record its use on the collective stream, reduce-scatter through `actor_group_sync_group`, divide the summed shard exactly once by `actor_group_size`, then accumulate it into the persistent shard buffer.
   - Update the focused test to use ActorGroup terminology and verify allocation stream, `record_stream`, communicator, one-time averaging, and accumulation together.

6. **Reconcile examples, tests, and independent merge additions.**
   - In `examples/run_qwen.py`, `examples/run_llama.py`, and `examples/run_harness.py`, retain placement-aware schedule visualization, Ray-safe absolute paths, dynamic-input configuration, and model seeding. Direct trace capture defaults to `2/1`; profile-and-simulate remains `5/30`.
   - Retain both branches' independent routed-boundary and split-backward/bucketing tests. Recompute the conflicted DAG digest only after confirming the resolved DAG is semantically intended.
   - Audit all auto-merged source and example code for legacy DP identifiers; compatibility-only parsing tests may mention old fields, but live runtime/profile/replay code may not.

### Compatibility risks

- Direct trace capture records only one active iteration by default; performance studies should use profile-and-simulate, whose default remains `5/30`, or explicitly request more active trace iterations.
- Trace-selection arguments containing `dp` in their public names will be replaced with global-rank terminology rather than silently retaining obsolete topology semantics.
- Stored artifacts may still deserialize historical `timing_source="cuda_event"` values, but newly generated node profiles will use CUPTI/Kineto timing.
- Interference-free serialization across disjoint ActorGroups has not yet been exercised with every routed topology; a collective launch-order deadlock is the explicit stop condition described above.

## Evaluation plan

1. **Static merge/topology audit** — criteria: no unresolved conflicts or legacy runtime topology.
   - Run `git diff --check`, Python compilation for changed Python files, and verify `git diff --name-only --diff-filter=U` is empty.
   - Search live source/examples for `dp_rank`, `dp_degree`, `dp_group`, `PIPER_DP_*`, profile-generation, and profile-shard APIs; review any remaining occurrence as either an error or an intentional compatibility test.

2. **Focused CPU profiling and replay tests** — criteria: rank-keyed CUPTI artifacts faithfully merge into and replay the DAG.
   - Run `test/test_profiling.py`, `test/test_profile_validation.py`, `test/test_profile_statistics.py`, `test/test_simulator.py`, `test/test_simulator_attribution.py`, `test/test_export_node_profile_ratios.py`, and `test/test_visualize_simulation_savings.py` with `cpu-venv`.
   - Verify explicit tests for complete ActorGroup rank coverage, duplicate/missing actor results, fingerprint/node coverage, iteration pairing across actors, CUPTI attribution ingestion, and simulation ordering from DAG dependencies plus stream FIFO.

3. **Focused CPU actor/runtime tests** — criteria: profiler schedule and stream policies match the requested semantics.
   - Run `test/test_actor_profile_timing.py`, `test/test_profile_streams.py`, `test/test_actor_group_topology.py`, and `test/test_coordinator_placement.py`.
   - Cover the direct-trace `2 warmup / 1 active` default, the profile-and-simulate `5/30` default, configurable counts, automatic profiler stepping at DAG boundaries, preservation of all logical/directional streams in both modes, and interference-free serialization being separate from stream selection.

4. **Focused CPU executor and DAG regression tests** — criteria: ActorGroup collectives and independently merged DAG behavior remain correct.
   - Run the reduce-scatter/all-reduce tests in `test/test_executors.py`, including the combined allocation-stream/lifetime/communicator/averaging assertion.
   - Run `test/test_order_directive.py`, `test/test_dag_builder_equivalence.py`, and `test/test_piper_schedule.py` to retain routed-boundary and split-backward coverage and confirm schedule metadata remains ActorGroup-native.

5. **Focused GPU CUPTI integration** — criteria: real capture, serialization, and replay work across ActorGroups without per-node event timing.
   - Run one small conventional multi-stage schedule and one small AFD schedule with disjoint, unequal ActorGroups through profile-and-simulate. Use reduced explicit counts for this focused integration test while separately verifying the configured default remains `5/30`.
   - For both normal and interference-free modes, verify every expected global actor rank and executable DAG node has a CUPTI-attributed sample, recorded stream metadata includes the dedicated routed streams, mandatory profile validation passes, and the saved plan simulates successfully.
   - Compare captured GPU-activity signatures between normal and interference-free modes; they must contain the same node work even though execution overlap differs.
   - Treat a hang at a routed/P2P collective during interference-free capture as the planned stop condition; report the per-ActorGroup dispatch order rather than adding implicit order constraints.

6. **Focused GPU reduce-scatter integration** — criteria: the merged allocator fix works under a real ActorGroup collective.
   - Run one existing small grad-sharded correctness workload that exercises reduce-scatter, confirming completion, parameter/gradient parity under its existing tolerance, and no immediate per-iteration allocator growth regression.

No full unit or end-to-end suite is planned; the evaluation is limited to components affected by this merge.

## Implementation log

- Resolved the topology/runtime conflicts around one driver and explicit ActorGroups; simulator profile generations and per-DP shard/barrier state were not retained.
- Combined the reduce-scatter allocator-stream fix with `actor_group_sync_group` and exactly one division by `actor_group_size`.
- Removed the legacy per-node CUDA-event recorder/profile path. CUDA events remain only for execution dependencies, lifetimes, interference-free stream handoffs, and the out-of-DAG directional-P2P baseline.
- The DAG-equivalence routed fixture predated the explicit runtime-value mapping contract and failed before digest comparison. Added valid producer-to-consumer mappings to routed fixture edges; only routed-case digests changed.
- ActorGroup-local rank is the same-position pairing key for ordinary P2P replay; it is not a DP coordinate. The stale simulator `dp_lane` name and rectangular-DP commentary were removed without changing this behavior.
- Schedule-defined routed ownership and communicator rank order remain separate concepts. The pre-existing routed-transport normalization/repacking contract is retained; this merge does not infer ownership from ActorGroup-local rank.
- `src/qwen_control_plane_workloads.py` retains explicit DP terminology because it describes checked-in benchmark workload cases, not runtime/profile topology. The live driver only clears stale `PIPER_DP_*` variables from reusable Ray workers and does not consume them.
- The generic executor timing-callback test seam remains, but no production implementation or profile path supplies a per-node CUDA-event recorder; all newly produced node profiles use `timing_source="kineto_cupti"`.

## Evaluation results

- Focused CPU evaluation: **213 passed** across ActorGroup topology/placement, CUPTI lifecycle and attribution, profile aggregation/validation/statistics, simulator/replay/attribution, stream selection, execution planning, routed/order DAG construction, reduce-scatter, and example metric plumbing.
- Python compilation succeeded for every added or modified Python file in the merge.
- The working diff has no unresolved merge entries or conflict markers; the unstaged diff passes `git diff --check`.
- `git diff --cached --check` reports trailing whitespace in three incoming simulator benchmark CSV artifacts under `benchmark_data/partial_overlap_interference/resource_usage/`. These files were retained byte-for-byte because benchmark-data cleanup is outside this integration.
- Focused GPU verification ran on Ray `10.52.48.236:6379` (two nodes, eight H100s each):
  - Conventional Qwen 9M, four singleton ActorGroups, 1F1B/4 microbatches: CUPTI validation covered 424 mode/rank node results and 56,687 launches; normal/IF activity matched exactly, all 212 replay events matched, and simulation completed (`out/20260817_104629`).
  - AFD Qwen 9M, unequal ActorGroups `[0,1]` and `[2]`, two microbatches: CUPTI validation covered 430 mode/rank node results and 55,328 launches; normal/IF activity matched exactly, all 215 replay events matched, and simulation completed without collective-order deadlock (`out/20260817_104847`). Every actor retained distinct M-to-N and N-to-M physical streams, both separate from the logical EP stream, in both modes.
  - Grad-sharded Qwen 9M, two width-2 ActorGroups: two-step parameter parity passed for all 51 tensors with the established BF16/routed tolerances; maximum error was one BF16 ULP (`4.76837158203125e-7`) (`out/20260817_105130`). Three subsequent timed steps had identical per-rank peak allocated memory, with zero growth on every rank (`out/20260817_105258`).
- The first AFD attempt on worker node `10.52.48.237` failed before Piper setup because Ray could not deserialize the driver (`ValueError: source code string cannot contain null bytes`). The identical workload passed on `10.52.48.236`, isolating this as a worker-node Ray/environment issue rather than an AFD/CUPTI execution failure.

## Remaining limitations

- The interference-free policy deliberately adds no schedule constraints. The focused conventional and unequal-group AFD schedules completed without deadlock, but schedules with different collective orders remain subject to the existing DAG-valid ordering requirement.
- Worker node `10.52.48.237` has a Ray deserialization problem that prevented using it for the AFD run; multi-node behavior was not needed for this focused topology/profile integration but should be repaired independently before cross-node validation.
- Historical artifacts with `timing_source="cuda_event"` remain readable; newly captured profiles do not use that timing source.

## Commit message

`Merge CUPTI profile replay with ActorGroup runtime topology`
