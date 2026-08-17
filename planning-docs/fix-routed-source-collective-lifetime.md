# Fix routed source collective lifetime

## Task summary

Stop one-way cross-group routed collectives from retaining every source payload
after its iteration completes, which currently causes linear GPU-memory growth
and prevents longer profile-and-simulate runs.

## Objectives

- Release source-side M-to-N and N-to-M collective results and their autograd
  inputs at a launch-safe iteration boundary.
- Preserve the existing routed communication, dependency, and overlap
  semantics.
- Restore steady-state GPU memory across repeated AFD training iterations.

## Non-goals

- Redesign the routed transport or replace functional collectives broadly.
- Change schedule placement, ordering, model annotations, or simulator policy.
- Optimize the remaining steady-state activation or optimizer memory.
- Change same-group ragged all-to-all behavior, whose result is consumed by a
  model region and does not exhibit this one-way-result leak.

## Correctness criteria

- **C1:** Repeated routed training steps do not retain an additional dispatch or
  combine payload from each completed step.
- **C2:** Both dense-source M-to-N payloads and routed-source N-to-M payloads reach a
  stable live-tensor count after optimizer initialization.
- **C3:** Cleanup does not block host launch before all ranks can enqueue the matching
  collective operations and does not deadlock relaxed pipeline schedules.
- **C4:** Forward results, backward gradients, and parameter updates remain equivalent
  to the existing routed execution within established tolerances.
- **C5:** The world-6 AF-Pipe profile-and-simulate workload can complete its requested
  warmups and samples without leak-induced OOM.

## Subtasks

1. Add launch-safe ownership and cleanup for discarded source-side routed
   collective results.
2. Add focused lifecycle and memory-regression coverage.
3. Run routed correctness tests and the motivating world-6 GPU workload.

## Assumptions / open questions

- Diagnosis found that `all_to_all_single_autograd` returns an asynchronous
  wrapper on source ranks even though their receive size is zero. Dropping that
  unconsumed wrapper without completing it retains its large input payload.
- Immediate source-side `wait()` is invalid for the current relaxed schedule:
  it blocks further host launch and deadlocks collective progress. A deferred
  cleanup after all DAG operations have been enqueued was validated
  experimentally and kept memory flat.
- The permanent cleanup point and failure-path handling will be specified in
  the Phase 2 implementation plan after this scope is approved.

## Implementation plan

1. In `src/executors.py`, give `DagExecutor` explicit ownership of discarded
   source-side results returned by cross-group M-to-N and N-to-M functional
   collectives. Destination results continue through the existing model-data
   path unchanged.
2. Do not wait inside the routed collective node. Append the source result to
   per-iteration pending state so the actor can enqueue every later DAG node
   needed for collective progress.
3. Add a small executor helper that completes and clears pending source results
   only after the actor has enqueued the full DAG. Run it before returning from
   a successful iteration, after profiling synchronization when profiling is
   active. This bounds ownership to one iteration and also releases the final
   iteration rather than depending on a subsequent call.
4. Keep failure handling conservative: if DAG dispatch raises before all nodes
   are enqueued, do not synchronously drain potentially unmatched collectives;
   propagate the original execution error. A failed distributed iteration is
   already non-recoverable for that actor set.
5. Update focused executor tests in `test/test_executors.py` to model an
   asynchronous collective result and verify deferred completion for both
   source directions. Reuse existing routed execution fixtures rather than
   adding schedule-specific behavior.

Compatibility risk is limited to completion timing for source-only functional
collective results. The change does not alter tensors, split vectors,
communicators, DAG edges, or model-visible interfaces. Immediate completion is
explicitly excluded because it deadlocked the motivating relaxed schedule.

## Evaluation plan

- **C1, C2 — focused GPU memory regression:** run at least four ordinary
  world-6 AF-Pipe steps on Qwen3-30B-A3B-L4, batch 4, sequence 1024, four
  microbatches. Record per-rank peak-memory samples and verify that dense and
  routed ranks stop accumulating one full source payload per step after
  optimizer-state initialization. Compare against the captured unfixed series.
- **C3 — focused unit tests and distributed regression:** run the new deferred
  completion tests plus existing cross-group executor tests. Then run the same
  world-6 schedule on Ray `10.52.48.236:6379`; successful repeated iterations
  establish that cleanup does not recreate the immediate-wait deadlock.
- **C4 — routed correctness:** run the existing GPU Qwen AFD parameter-parity
  tests for the single-node PP2 and PP4 placements, together with the focused
  CPU executor tests. These cover both cross-group directions and their
  backwards without broad unrelated suites.
- **C5 — motivating profile-and-simulate run:** use the world-6 AF-Pipe
  schedule, Qwen3-30B-A3B-L4, batch 4, sequence 1024, four microbatches, five
  profiling warmups, and 30 samples for each normal and interference-free
  mode. Report the output plan, measured/simulated times, variance, estimated
  interference savings, and any residual/replay-confidence warnings.
- Run `git diff --check` and the narrowest relevant CPU test selection before
  using GPUs. No full unit or end-to-end suite is planned because the change is
  confined to routed-result lifetime and the user requested relevant tests
  only.

## Implementation log

- Added per-iteration ownership of discarded cross-group source results in
  `DagExecutor`. Source nodes append without waiting; a successful executor run
  completes and clears them only after every local DAG node has been enqueued.
- Existing mocked source collectives returned ordinary tensors. Focused tests
  now use waitable stand-ins and assert both that completion is deferred and
  that the final drain occurs for M-to-N and N-to-M.
- The diagnosed immediate-wait alternative was rejected because it prevented
  later collective launch and deadlocked the relaxed world-6 schedule. Deferred
  completion preserved launch progress and bounded live payloads to one step.
- Sandboxed Ray clients could not reach the remote GCS. GPU commands used the
  task-authorized remote execution context. Correctness tests also had to use
  the active cluster's `/tmp/piper/ray_tmp`; a test-specific temp directory
  caused Ray to search for a nonexistent local session file.
- A first full profile run was terminated after GNU `timeout` sent `SIGTERM`
  but did not force-kill the Ray driver's shutdown. A 1-warmup/2-sample smoke
  run proved there was no collective deadlock, and the required run then used
  an explicit kill grace. It completed normally in 55 seconds.

## Evaluation results

- `cpu-venv/bin/python -m pytest -q test/test_executors.py -k
  'cross_group_n_to_m_uses_canonical_rank_block_order or
  m_to_n_source_uses_canonical_shard_and_collective_order'`: **2 passed, 25
  deselected**.
- `cpu-venv/bin/python -m pytest -q test/test_executors.py`: **27 passed**.
- `cpu-venv/bin/python -m pytest -q test/test_profile_streams.py
  test/test_execution_plan.py test/test_afd_routed_directives.py -k
  'cross_group or routed or m_to_n or n_to_m or profile'`: **48 passed, 9
  deselected**.
- `git diff --check`: **passed**.
- World-6 Qwen3-30B-A3B-L4 memory regression on Ray
  `10.52.48.236:6379`, batch 4, sequence 1024, four microbatches, five ordinary
  iterations: **passed**. Dense ranks stabilized at **9.154 GiB** after
  optimizer initialization. Routed ranks stayed around **19.0–19.8 GiB** as
  dynamic routed row counts changed, without the prior linear growth. Raw
  metrics: `out/20260817_090956/raw_metrics.json`.
- Focused Qwen 9M AFD parameter parity, PP2, two SGD steps: **passed** on three
  GPUs of node `10.52.48.236`. Metrics:
  `out/20260817_091731/raw_metrics.json`.
- Focused Qwen 9M AFD parameter parity, PP4, two SGD steps: **passed** on six
  GPUs of node `10.52.48.236`. Metrics:
  `out/20260817_091824/raw_metrics.json`.
- Required world-6 profile-and-simulate run, five warmups and 30 measured
  samples per mode: **passed**. Dependency replay medians were **190.188 ms**
  normal and **186.366 ms** interference-free, predicting **2.010%** savings.
  Their sample standard deviations were **11.889 ms** and **14.438 ms**.
  Instrumented policy medians were **184.017 ms** normal and **193.310 ms**
  interference-free, or **-5.050%** policy reduction. Replay errors against the
  instrumented modes were **+3.353%** normal and **-3.592%**
  interference-free, so the predicted saving should be treated as directional
  rather than a precise hardware speedup. Metrics:
  `out/20260817_092534/raw_metrics.json`; reusable plan:
  `out/afd_pp2_1f1b_comparison/step_time_profiles_afd_pp2_afpipe_world6_mbs4.json`.

## Remaining limitations

- A distributed iteration that raises before all ranks finish enqueueing is
  not recoverable; cleanup deliberately does not wait on potentially unmatched
  collectives while propagating that original failure.
- Routed-row variability changes the routed ranks' per-step peak memory, so the
  regression checks absence of monotonic payload accumulation rather than
  requiring byte-identical routed-rank peaks.
- Profile replay and directly instrumented stream policies disagree on the sign
  of the estimated interference benefit for this run; this is a profiling and
  simulation confidence limitation, not a memory-lifetime failure.

## Commit message

`Release discarded routed collective results after DAG launch`
