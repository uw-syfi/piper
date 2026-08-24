# Split profile/replay and communication-ordering commits

## Phase 1: Scope

### Task summary

Curate the current mixed staged/unstaged working tree into two reviewable
commits: a global communication-ordering/runtime commit and a
profile-and-simulate commit. Preserve unrelated work outside both commits,
exclude activation-checkpointing changes, and remove source-level diagnostics
that were added only for one-off investigations.

The profile/replay commit is created first, matching the requested commit-1
schedule/preset ownership. Transaction-aware IF serialization is kept with the
second commit because it consumes finalized global transaction semantics.

### Objectives

- Create one commit for global communication ordering, including:
  - explicit issue-only DAG edges and deterministic dispatch levels;
  - global P2P/collective transaction validation and ordering;
  - DAG-faithful stream/completion handling and directional P2P execution;
  - asynchronous P2P work and tensor lifetime retention;
  - routed/dense-state communication ordering fixes;
  - non-overlapping ZeRO parameter/gradient residency intervals and the
    bucket/runtime mappings required by those schedules;
  - focused tests, reusable schedule fixtures, and completed planning
    documents that directly verify or explain this behavior.
- Create one commit for profile-and-simulate, including:
  - ActorGroup-compatible Kineto/CUPTI capture and actor timeline alignment;
  - launch/activity capture validation and bounded capture-loss handling;
  - symmetric all-participant replay for routed and ordinary collectives with
    endpoint-specific service tails;
  - paired sample replay, arithmetic-mean statistics, sample standard
    deviation/confidence reporting, and harness/CLI reporting;
  - focused tests, reusable profile/replay tools and documentation, and
    completed planning documents that directly verify or explain this
    behavior.
- Remove only source logic that is purely investigative rather than required
  by normal profiling, replay, validation, or execution. The currently
  identified targets are the opt-in detailed Kineto overlap/GPU-gap report and
  the standalone CUPTI node-timing probe used only by its focused test.
- Include the experimental Qwen L1/L2 presets and their A4/F4 and A6/F4 AFD
  schedules in commit 1, together with the persistent AF-Pipe schedule used by
  the communication-ordering work.
- Leave `benchmark_data/partial_overlap_interference/` present but uncommitted.
  After both commits are verified, discard every other residual staged,
  unstaged, or untracked change.
- Prepare descriptive commit messages that state the behavior and important
  correctness guarantees rather than experiment-specific history.

### Non-goals

- Do not commit activation-checkpointing implementation, CLI plumbing, or its
  E2E test.
- Do not commit the benchmark-data bundle, benchmark drivers, trace-analysis
  scripts, sharding-matrix scripts, or profile diagnostic scripts.
- Do not commit the unrelated Qwen 9B/debug-routing edits, generated outputs,
  `gpu-venv`, or the unrelated root README deletion.
- Do not change communication, replay, ZeRO, or statistical semantics beyond
  removing investigation-only source hooks and making the two commits
  independently coherent.
- Do not rewrite history before `HEAD` or combine the two requested topics.

### Correctness criteria

- Exactly two new commits contain the requested work, with profile/replay first
  and global communication ordering second.
- Each commit is internally coherent: its source imports, public data flow,
  tests, docs, and any included schedule fixture agree at that commit boundary.
- The communication-ordering commit represents issue order separately from
  completion dependencies, preserves declared routed streams and directional
  P2P streams, retains asynchronous communication resources until safe, and
  rejects cyclic modeled communication orders.
- ZeRO full-parameter/full-gradient residency intervals for the same bucket do
  not overlap unless directly connected into one lifetime, and contradictory
  interval requirements fail during DAG construction.
- Profile/replay validates CUPTI attribution and paired captures, aligns actor
  timelines, models every supported collective as an all-participant
  rendezvous with endpoint-specific service, and reports replay endpoints and
  paired interference savings from aligned raw iterations.
- No activation-checkpointing diff, one-off analysis code/script, generated
  output, virtual environment, or benchmark-data change appears in either
  commit.
- General reusable profile/replay tooling remains available, while detailed
  investigation-only Kineto overlap/gap reporting and the standalone probe are
  absent from `src/`.
- The benchmark-data bundle remains byte-for-byte present and uncommitted after
  both commits; every other residual change is intentionally discarded only
  after the two commit contents and evaluations have been audited.

### Subtasks

1. Inventory every full `HEAD`-to-working-tree hunk and assign it to global
   ordering, profile/replay, source cleanup, activation checkpointing, or
   unrelated work.
2. Remove the identified investigation-only source hooks and their dedicated
   tests without weakening mandatory CUPTI/replay validation.
3. Rebuild the index from the working tree, stage and verify the profile/replay
   boundary, then commit it.
4. Stage and verify the global-ordering boundary, then commit it.
5. Audit both commits for accidental inclusion or omission, preserve the
   benchmark-data bundle, then discard every other residual change and verify
   the resulting status.

### Assumptions / open questions

- “One-off analysis/validation” is interpreted narrowly: optional overlap,
  allocator-gap, and standalone probing logic is removed; mandatory
  launch/activity equivalence, capture-loss classification, collective
  participant validation, replay endpoint validation, and reusable savings
  attribution remain because they enforce profile-and-simulate correctness.
- The reusable `scripts/profile_and_replay` CLI, attribution, and visualization
  entry points are product tooling, not one-off analysis. Its `diagnostics/`
  subtree is one-off and remains uncommitted.
- Experimental Qwen presets and schedules are retained in commit 1 by explicit
  request even when they are not imported by a focused unit test. They will be
  syntax/load checked as schedule examples.
- Existing relevant completed plans will be committed with their owning topic;
  investigation reports and incomplete/unrelated plans will remain
  uncommitted.

## Phase 2: Implementation and evaluation plan

### Implementation plan

#### 1. Establish a safe index baseline

- Record the current commit, complete path inventory, and benchmark-data
  checksums before changing the index.
- Reset only the index to `HEAD`; do not reset or delete working-tree content.
- Treat full `HEAD`-to-working-tree content as the source of truth and use
  path- and hunk-level staging for files shared by both topics.

#### 2. Build commit 2: global communication ordering and residency

Stage the behaviorally related portions of:

- DAG construction/order/runtime: `src/dag.py`, `src/ordering.py`,
  `src/execution_plan.py`, `src/directives.py`, `src/piper.py`,
  `src/executors.py`, `src/runtime.py`, `src/compile.py`,
  `src/control_plane_benchmark.py`, `src/qwen_control_plane_workloads.py`,
  `src/visualization.py`, and `src/zero.py`.
- ActorGroup/topology support required by finalized actor-local DAGs: relevant
  portions of `src/actor.py`, `src/coordinator.py`, and `src/state.py`.
- Schedule generation/examples: `examples/build_schedule.py`, only the L1/L2
  preset additions from `examples/models/qwen3.py`, all currently new reusable
  PP/EP unsharded/ZeRO-2 schedules, the persistent AFD AF-Pipe schedule, and
  the L1/L2 A4/F4/A6/F4 schedules including their ZeRO-3 variants. Do not stage
  the unrelated 9B layer-count or forced-routing debug edits from the same
  model file.
- Tests: global ordering, execution-plan, executor, order-directive, AFD
  lowering/routed-directive, DAG-equivalence, ActorGroup topology,
  visualization, and relevant Qwen E2E changes, excluding AC cases/plumbing.
- Plans: `dag-faithful-communication-ordering.md`,
  `global-communication-ordering.md`,
  `zero-full-parameter-residency-intervals.md`,
  `afd-zero-bucketing-runtime-mappings.md`, and
  `adjust-afpipe-overlap-schedule.md`, after making included plans accurately
  reflect completed work.

This commit includes asynchronous P2P work/tensor retention, truthful
completion dependencies, same-stream FIFO and minimal issue-only ordering,
global transaction cycle rejection, and bucket-local ZeRO residency ordering.
It excludes CUPTI collection, replay statistics, and AC behavior.

Proposed commit message:

```text
Make communication ordering globally DAG-faithful

Normalize P2P and collective endpoints into a shared transaction graph,
reject cyclic communication orders, and project a deterministic order into
same-stream FIFO and minimal cross-stream issue edges. Retain asynchronous P2P
resources until completion and enforce non-overlapping bucket-local ZeRO
parameter and gradient residency intervals.
```

#### 3. Remove investigation-only profile source hooks

Before staging commit 2:

- Remove the `detailed=True` overlap/gap branch from Kineto attribution,
  including allocator overlap, per-neighbor overlap, kernel-gap classification,
  and `largest_gpu_gaps` payloads not consumed by normal profiling or replay.
- Remove the standalone `piper_profile_cupti_node_times` entry point and its
  dedicated test; `piper_profile_dags` remains the supported capture API.
- Retain compact per-occurrence activity/launch signatures, actor timeline
  anchors, capture-loss classification, and mandatory profile/replay
  validation because they are production correctness inputs.

#### 4. Build commit 1: aligned CUPTI profile and replay

Stage the profile/replay portions of:

- Capture/artifacts: `src/actor.py`, `src/piper.py`, `src/profiling.py`, and
  profile-related portions of `src/compile.py`, `src/executors.py`, and
  `src/runtime.py`.
- Replay/reporting: `src/simulator.py`, `src/profile_validation.py`,
  `src/profile_statistics.py`, and `src/attribution.py`.
- Harness/tooling: profile portions of `examples/test_harness.py`,
  `examples/test_qwen.py`, and `examples/test_llama.py`; reusable files directly
  under `scripts/profile_and_replay/`, including its README, simulator CLI,
  attribution/export tools, and savings visualizer. Exclude `diagnostics/`.
- Tests: actor/profile timing, profiling, profile validation/statistics,
  simulator/replay/attribution, profile streams, reporting, export, and
  savings-visualization tests. Exclude benchmark and AC tests.
- Plans/docs: `merge-simulator-cupti-actor-groups.md`,
  `align-cupti-actor-timelines.md`, `robust-cupti-activity-capture.md`,
  `symmetric-routed-collective-replay.md`,
  `remove-simulation-bootstrap-resampling.md`,
  `communication-replay-audit.md`, and this commit-splitting plan.

The commit retains one aligned aggregate envelope per communication node,
symmetric all-participant readiness for ordinary and routed collectives,
endpoint-specific service tails, paired raw-iteration replay, and mean/sample
SD/Student-t reporting. It does not reintroduce per-node CUDA-event profiling.

Proposed commit message:

```text
Replay DAG execution from aligned CUPTI profiles

Collect node activity with aligned Kineto/CUPTI ranges, validate paired
captures, and replay ordinary and routed collectives with symmetric participant
rendezvous and endpoint-specific service tails. Report paired normal and
interference-free results using means, sample variation, and confidence
intervals.
```

#### 5. Audit and discard residual work

- Inspect each commit's paths and representative diff; search both commits for
  AC, benchmark, diagnostic, generated, and environment paths.
- Verify benchmark-data checksums still match the pre-staging inventory and
  leave that directory untracked/uncommitted.
- Restore every remaining tracked change to the new `HEAD` and remove every
  remaining untracked path except
  `benchmark_data/partial_overlap_interference/`. Enumerate exact targets from
  `git status`; do not use a workspace-wide reset or broad clean.
- Confirm final status contains only the preserved benchmark-data directory.

### Compatibility and implementation risks

- `actor.py`, `piper.py`, `executors.py`, `runtime.py`, and `simulator.py`
  contain interleaved hunks. Each commit boundary must be tested independently;
  a final-only passing tree is insufficient.
- Symmetric replay itself does not require global issue edges. The
  transaction-aware IF serializer does, so its code, tests, and plan are kept
  in commit 2.
- ActorGroup topology is shared infrastructure. Construction/execution pieces
  needed by finalized DAGs belong in commit 1; CUPTI-specific actor state
  belongs in commit 2.
- The Qwen model file requires hunk-level staging to retain presets without the
  unrelated 9B/debug-routing edits.
- Residual cleanup is destructive but explicitly requested. It occurs only
  after both commits pass planned checks and benchmark data is verified.

### Evaluation plan

#### Commit 2

- Compile every staged Python source/test module and run `git diff --check`.
- Run focused CPU tests for global communication ordering, execution-plan
  lowering, stream/order directives, AFD routed/full lowering, executor P2P
  lifetimes, DAG equivalence, ZeRO residency, ActorGroup topology, and DAG
  visualization.
- Load every committed JSON schedule through the parser; build the L1/L2
  presets and dry-lower representative AFD schedules without GPU execution.
- If the active Ray cluster is reachable, run the existing small direct-harness
  AFD and DualPipeV parameter-parity cases. Otherwise record the environmental
  limitation and rely on the focused CPU suite plus prior GPU evidence in the
  included plans.
- Inspect the commit for profile-only, AC, benchmark/diagnostic, and unrelated
  Qwen hunks.

#### Commit 1

- Compile every staged Python source/test module and run `git diff --check`.
- Run focused CPU tests for Kineto attribution/timeline alignment, profile
  merging/validation, capture-loss classification, symmetric routed and
  ordinary collective replay, P2P microprofiles, paired statistics/reporting,
  transaction-aware IF serialization, attribution, and reusable profile tools.
- If a Ray GPU session is reachable, run one small direct-harness
  profile-and-simulate smoke. Otherwise record the limitation and rely on the
  focused CPU suite plus prior world-scale GPU evidence in the included plans.
- Inspect the commit for per-node CUDA-event timing, investigation-only audit
  payloads, AC, benchmark/diagnostic scripts, and ordering code that belongs in
  commit 1.

#### Final state

- Verify both hashes/messages and that each commit passes its topic-specific
  static/focused checks.
- Verify the combined tree imports and run the union of focused tests where
  practical.
- Verify `git status --short` reports only the preserved uncommitted
  `benchmark_data/partial_overlap_interference/` bundle.

## Implementation log

- Removed the opt-in detailed Kineto overlap/gap audit and standalone CUPTI
  probe while retaining launch/activity signatures, timeline anchors, and
  capture-loss validation.
- Isolated profile/replay from the mixed simulator diff by retaining the
  pre-global-ordering dense-state and routed-direction relaxation behavior in
  commit 1. DAG-faithful dependency handling remains for commit 2.
- Moved transaction-aware IF serialization to commit 2 because it relies on
  the global communication transaction graph.
- Verified the exact commit-1 index snapshot from a separate checkout:
  119 focused capture/replay/statistics/reporting tests passed (one expected
  synthetic warning), followed by 69 execution-plan/profile-stream/executor/
  ActorGroup tests passed. Python compilation and `git diff --check` passed.
- Removed the AC-only actor/compile hunks before staging commit 2. The existing
  checkpointing behavior remains unchanged.
- The new PP/EP fixtures initially failed the stricter replicate/shard overlap
  validator. Added explicit `EP: "-"` only to their replicate filters; all 40
  base schedule JSON files then loaded successfully.
- Verified the exact commit-2 index snapshot: Python compilation passed;
  243 focused ordering/routed-lowering/executor/ZeRO/ActorGroup/visualization
  tests passed, then 74 profile/replay regression tests passed (one expected
  synthetic warning). `git diff --check` passed.
