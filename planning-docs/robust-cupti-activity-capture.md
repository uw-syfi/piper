# Robust CUPTI activity capture

## Task summary

Record CPU launches per node occurrence so profile-and-simulate can distinguish
Kineto GPU-activity loss from a real difference in issued work. If the observed
normal/IF mismatch is definitively deletion-only capture loss, allow a global
loss rate of at most 0.01%; otherwise keep validation strict and investigate the
execution divergence.

## Investigation findings

- The 16-GPU ZeRO run compared 1,582,272 GPU activities across 69,984 node
  occurrences. Forty normal-mode occurrences each lacked exactly one activity
  (38 kernels and two D2D copies); interference-free never lacked an activity.
- The 40 losses were distributed across 15 ranks and 19 of 36 captured
  iterations. Rejecting every globally affected iteration left only 17 clean
  samples, so the fixed six-spare policy could not supply the requested 30.
- An independent one-iteration PyTorch trace of the same DAG contains all 40
  activities at exactly the interference-free multiplicity. None match the
  deficient normal multiplicity. This strongly indicates loss in the
  CUPTI/Kineto capture/export path rather than different model computation.
- No GPU external ID in that independent trace maps to more than one Piper node
  range, so external-ID collision is not supported as the cause.
- Piper's current `matched_launch_count` is computed from correlation-ID sets
  aggregated over every occurrence of a node. It cannot detect that one
  occurrence retained its CPU launch but lost the corresponding GPU activity.
- Kineto uses bounded CUPTI buffers and only queries/reports
  `cuptiActivityGetNumDroppedRecords` at verbose logging level. The run emitted
  no max-host-buffer warning, but default logging cannot prove that CUPTI's
  activity queue dropped no records.

## Objectives

- Classify a normal/IF mismatch as either capture corruption or a genuine
  difference in issued GPU work using per-node, per-occurrence evidence.
- Accept at most a 0.01% GPU-activity loss rate only when identical normalized
  CPU launch sequences, or deletion-only CPU launch loss at most 0.01%, and
  activity alignment prove that the differences consist solely of missing
  capture records.
- Preserve strict validation for CPU launch divergence, GPU activity
  substitution/reordering, unclassified mismatches, or capture loss above the
  threshold.
- Never synthesize or silently repair a missing activity; retain and report the
  affected occurrences and the measured loss rate.
- Keep profiling observational: accepted samples must represent the existing
  DAG, stream policy, and restored training state without changing execution or
  simulator semantics.

## Non-goals

- Changing DAG construction, executor scheduling, stream assignment, model
  routing, ZeRO lifetimes, or simulator replay behavior.
- Special-casing Qwen, MoE kernels, particular node kinds, schedules, ranks, or
  activity names.
- Treating a partial set such as the 17 clean samples from the failed run as a
  successful 30-sample result.
- Treating CPU-launch equality alone as sufficient when GPU activity changes by
  substitution, reordering, or an otherwise non-deletion difference.
- Depending on a particular Kineto environment variable or log message as the
  sole correctness check.
- Adding adaptive capture batching unless the CPU-launch investigation shows
  the loss cannot be safely classified and tolerated.

## Correctness criteria

- Every node occurrence records an ordered normalized CPU launch sequence and a
  per-launch mapping to captured GPU activities; correlation IDs, timestamps,
  and nested runtime/driver duplicates are not part of the cross-mode identity.
- A launch present in both modes whose expected GPU activity is absent in one
  mode is reported as capture loss for that exact occurrence, rather than being
  hidden by capture-wide correlation aggregation.
- A mismatch is definitively capture loss only when normal and IF CPU launch
  sequences are identical or one is an order-preserving subsequence of the
  other, and the GPU activity sequences are likewise identical or
  deletion-only. CPU launch or GPU substitutions and reorderings remain hard
  failures.
- CPU launch loss rate is `missing normalized CPU launches / expected CPU
  launches`, aggregated over all actors and paired occurrences. CPU launch and
  GPU activity loss are each independently limited to at most 0.0001 (0.01%).
- Capture-loss rate is `missing GPU activity records / expected GPU activity
  records`, where the expected denominator is the per-launch canonical
  one-mode activity sequence after normal/IF alignment (not the sum of both
  modes). The rate is aggregated across all executed nodes, actors, and paired
  captured iterations.
- Profile-and-simulate may pass activity validation with a loss rate no greater
  than `0.0001` (0.01%) for GPU activities and CPU launches only if every
  mismatch is definitively classified as
  deletion-only capture loss. The validation report lists every affected rank,
  node, occurrence, launch, and missing activity.
- Any unclassified mismatch, execution divergence, or capture-loss rate above
  0.01% fails validation and produces no simulator result.
- Accepted loss remains visible in profile metadata and uses the actually
  captured duration/timeline data; Piper does not invent replacement activity.
- Workloads with exact capture retain their existing behavior and report zero
  loss.

## Proposed solution boundary

1. Extend the Kineto audit with a per-occurrence normalized CPU launch sequence,
   per-launch captured GPU activity sequence, and unmatched-launch details.
2. Compare normal and IF launches first, then align GPU activities by normalized
   launch order. Classify only deletion-only differences under identical launch
   sequences as capture loss; retain all other differences as hard validation
   failures.
3. Rerun the 16-GPU sharded workload. If every mismatch is definitively capture
   loss, add the explicit 0.01% aggregate threshold and report the accepted loss
   diagnostics in the durable plan. If classification is ambiguous or reveals
   launch divergence, stop before adding the tolerance and debug that cause.
4. Keep Kineto verbose dropped-record reporting and raw trace export as optional
   corroboration, not as the acceptance oracle.

## Assumptions / open questions

- The failed artifact did not persist per-occurrence CPU launches, so the
  current 40-record mismatch cannot yet be called definitive capture loss.
- A normalized launch identity must be obtainable from Kineto without relying
  on process-local correlation IDs or timestamps. GPU kernel names remain part
  of the aligned activity identity, so equal generic `cudaLaunchKernel` API
  names cannot hide kernel substitution.
- A rare accepted missing activity can shorten that occurrence's measured GPU
  envelope. Evaluation must quantify the replay sensitivity of including the
  affected samples; if the 0.01% threshold can still create material bias, the
  plan must be revised rather than silently accepting it.

## Implementation plan

### 1. Record canonical CPU launches per node occurrence

Update `src/actor.py` so the Kineto audit groups launch callbacks by node-range
occurrence and correlation ID instead of aggregating correlation IDs by node
label. For each occurrence:

- Collapse nested runtime/driver callbacks for one correlation into one
  canonical launch, preferring the runtime callback and falling back to the
  driver callback.
- Normalize the selected callback to its operation class (`kernel`, `memcpy`,
  or `memset`) while retaining the selected API name for diagnostics.
- Order canonical launches by CPU issue timestamp and associate each with the
  ordered `(GPU activity category, Kineto name)` records carrying the same
  occurrence-local correlation ID.
- Record ambiguity explicitly: duplicate canonical callbacks, GPU activities
  without an occurrence-local launch, and launches/activities without usable
  correlation cannot support capture-loss classification.
- Store the launch sequence, its hash, per-launch activity sequence, and
  occurrence-local coverage in `range_activity_signatures`. Derive aggregate
  launch counters from these occurrence records so one successful occurrence
  cannot hide loss in another.

The full launch/activity lists are comparison inputs, not durable bulk profile
data. After central comparison, retain hashes/counts for every occurrence and
full launch/activity detail only in mismatch diagnostics.

Allow actor-side timing extraction to carry incomplete occurrences to central
paired-mode validation. It must continue to validate occurrence/sample shape,
but must not reject an occurrence solely because its launch lacks a captured
GPU activity; otherwise the normal and IF evidence cannot be compared. A
single-mode profile still fails incomplete launch coverage during central
validation.

### 2. Classify paired-mode differences from launches first

Refactor `_compare_kineto_activity_signatures` in `src/piper.py` into a
non-destructive comparison so it can be rerun after selecting iterations. For
each common rank, node, occurrence, and canonical launch:

1. Require identical normalized CPU launch sequences between normal and IF.
2. Require an unambiguous occurrence-local launch/activity mapping.
3. Compare the paired per-launch GPU activity sequences. Exact equality is an
   exact match. If one is an order-preserving subsequence of the other, the
   absent records are deletion-only capture loss. Any substitution, reordering,
   or pair for which neither sequence is a subsequence is an execution or
   capture divergence, not tolerated loss.

Accumulate the canonical expected activity count and missing activity count
over all paired occurrences. Report `exact_match`, `capture_loss_only`, the
loss counts/rate, and detailed mismatches separately so accepting capture loss
never relabels it as exact equivalence. A constant of `1e-4` defines the 0.01%
maximum.

### 3. Integrate conditional acceptance with sample selection

Keep the existing preference for exact paired samples:

- If spare captures provide the requested number of globally exact iterations,
  retain those iterations and preserve exact-match behavior.
- If they do not, retain the deterministic first requested paired iterations
  rather than selecting samples based on their timing or loss pattern. Accept
  only when both the complete raw capture and the retained subset contain
  exclusively classified deletion loss and each has loss rate at most 0.01%.
- Otherwise fail profile validation and emit no simulator result.

Update `src/profile_validation.py` to accept either exact equivalence or the
qualified deletion-only result. Keep per-mode attribution, timing identities,
collective endpoint checks, and single-mode launch coverage strict. Persist the
raw and retained loss counts/rates, threshold, acceptance decision, and every
affected rank/node/occurrence/launch/activity in profile metadata and the
validation report. Never create a replacement activity or duration.

### 4. Focused tests

Extend `test/test_actor_profile_timing.py` with synthetic Kineto traces that
verify occurrence-local launch recording, runtime/driver de-duplication,
launches missing a GPU record, multiple activities associated with one launch,
and explicit rejection metadata for ambiguous mappings.

Extend `test/test_profile_validation.py` to cover:

- exact launch and GPU activity equivalence;
- identical CPU launches plus a deletion-only GPU loss;
- CPU launch divergence;
- GPU activity substitution and reordering;
- loss exactly at, below, and above 0.01%;
- aggregation across ranks, nodes, occurrences, and both modes;
- strict incomplete-coverage behavior for single-mode profiling;
- exact-sample preference and deterministic fallback selection when spare
  iterations cannot provide the requested count; and
- retention of complete mismatch diagnostics while transient bulk sequences
  are removed.

No model-, schedule-, rank-, node-kind-, or kernel-name-specific rule will be
introduced.

## Compatibility and risks

- Existing exact captures remain exact and retain the existing clean-spare
  filtering behavior.
- Profile metadata gains explicit launch/capture-loss diagnostics. The existing
  `exact_match` field keeps its literal meaning; consumers must use the
  validation pass/acceptance field rather than interpreting tolerated loss as
  exact equality.
- Correlation IDs are used only within one capture to associate records. They
  are excluded from normal/IF identity and durable diagnostics.
- Normalizing runtime/driver callback names is intentionally limited to
  collapsing the two APIs for the same correlated operation class. Kernel
  identity still comes from the paired GPU activity name, preventing a generic
  `cudaLaunchKernel` match from hiding kernel substitution.
- Missing activities remain missing from measured envelopes. The GPU
  evaluation will quantify the effect on replay before the tolerance is
  considered complete.

## Evaluation plan

1. **Occurrence-local recording and mapping (correctness criteria 1–3):** run
   `test/test_actor_profile_timing.py`, including the new synthetic traces, and
   inspect that nested runtime/driver callbacks count once and a missing GPU
   record remains attached to the exact launch occurrence.
2. **Classification and threshold (criteria 3–6):** run
   `test/test_profile_validation.py`, including exact, deletion-only,
   ambiguous, substitution, reorder, aggregate-rate, and threshold-boundary
   cases. Verify no failing case produces an accepted validation report.
3. **Profile serialization/regression (criteria 5–8):** run the focused
   profiling tests that exercise `ActorProfileResult`, profile merging,
   filtering, and saved-plan validation (`test/test_profiling.py` plus the two
   files above). Verify exact captures report zero loss and transient full
   sequences do not inflate the saved artifact.
4. **Definitive GPU diagnosis (all criteria):** rerun the same 16-GPU,
   parameter/gradient-sharded DualPipeV PP2/virtual-PP4 EP8 Qwen workload with
   five warmups and 30 requested paired profile iterations. Inspect the raw
   CPU-launch/GPU-activity report before relying on the tolerance:
   - If CPU launches match for every discrepant occurrence, every GPU delta is
     deletion-only, and both raw and retained loss rates are at most 0.01%,
     require profile-and-simulate to complete and persist the accepted loss
     report.
   - If any launch diverges, mapping is ambiguous, activity is substituted or
     reordered, or either rate exceeds 0.01%, do not accept the run; continue
     debugging from the recorded occurrence and raw trace evidence.
5. **Replay sensitivity:** for a definitively classified run, compare the
   reported simulated normal, simulated IF, and savings against a diagnostic
   replay whose affected samples are omitted from the corresponding node
   estimators. Record the absolute and relative differences. If the accepted
   records materially change the replay conclusion rather than merely its
   sampling noise, stop and revise the policy instead of declaring completion.

Implementation will stop at step 4 if the evidence does not definitively prove
capture loss; the 0.01% path will not be used to mask an unresolved mismatch.

## Implementation log

- Added occurrence-local canonical CPU launch records. Runtime/driver callbacks
  sharing one correlation collapse to the runtime callback; comparison identity
  uses the normalized operation class while retaining the API name for
  diagnostics.
- Actor-side envelope extraction now defers incomplete launch coverage to the
  central paired-mode validator. This is necessary because only the matching
  mode can establish the expected GPU activity; single-mode validation remains
  strict.
- The comparator is non-destructive so raw and retained iteration sets can both
  be audited before transient full sequences are removed.
- Exact spare-sample filtering remains preferred. When too few globally exact
  iterations exist, selection uses the first requested paired iterations and
  requires both raw and retained captures to meet the deletion-only 0.01%
  condition.
- Focused CPU evaluation so far: `test/test_actor_profile_timing.py`,
  `test/test_profile_validation.py`, and `test/test_profiling.py`: **47 passed**.
- Two forward-order 16-GPU captures (IF followed by normal) found 11 and six
  raw mismatched occurrences. Every mismatch changed both the CPU-launch and
  GPU-activity sequence, so none qualified for the 0.01% capture-loss path.
  In the latter run, six spare iterations supplied 30 exact retained samples;
  paired validation passed without tolerance. The subsequent replay failed an
  unrelated pre-existing collective aligned-endpoint check.
- Reversing the mode order (normal followed by IF) reversed the mismatch
  direction: the second mode omitted launches and matching GPU activities.
  This rules out the normal-versus-IF stream policy as the cause.
- Fresh-process single-mode captures compared 3,888 occurrence records between
  normal and IF. CPU-launch hashes/counts and GPU-activity hashes/counts matched
  for every occurrence. The divergence is therefore specific to the second
  Kineto profiling session in one actor process, but the Kineto-recorded CPU
  callbacks are not an independent oracle for whether that session dropped
  both callback and activity records or process-global execution state skipped
  a launch. The observed mismatch is not yet definitive capture loss and was
  not accepted by the tolerance.
- Temporary single-mode and reversed-order diagnostic drivers were removed
  after their artifacts were saved under `out/20260818_143819`,
  `out/20260818_144844`, and `out/20260818_145058`.
- A same-process, double-normal control reproduced the asymmetry without any
  policy change: its second session recorded 15 fewer CPU launches and 12
  fewer GPU activities across 527,424 expected activities
  (`out/20260818_145818`). Three occurrences retained the GPU activity while
  losing its CPU callback, proving that Kineto can lose CPU launch records as
  well as GPU records.
- A short verbose control without CUPTI teardown failed because second-session
  all-gather activities were timestamped 300--569 us before their enclosing
  active-iteration anchors. All work had been device-synchronized before the
  session, so these records cannot faithfully represent the current node
  occurrence and expose stale cross-session CUPTI/correlation state.
- Enabling Kineto's supported `TEARDOWN_CUPTI=1` path removed the stale-anchor
  failure, and verbose CUPTI diagnostics reported no queue drops or early
  stop. The second identical normal session nevertheless omitted three paired
  CPU-launch/GPU-activity records out of 87,904 (0.00341%)
  (`out/20260818_150732`). Because the CPU launch sequences differ, this run
  did not satisfy the then-current GPU-only deletion proof; this observation
  motivated the subsequently approved independent CPU-loss policy below.
- `KINETO_LOG_LEVEL` is now forwarded to actors when explicitly set, allowing
  CUPTI's built-in dropped-record and early-stop diagnostics to corroborate
  future captures without changing default logging.
- The approved acceptance policy was extended to tolerate deletion-only CPU
  launch loss at most 0.01%, independently of the identical GPU activity
  threshold. Substitution or reordering in either sequence remains fatal.
- Focused evaluation after adding the CPU-loss boundary cases:
  `test/test_actor_profile_timing.py`, `test/test_profile_validation.py`, and
  `test/test_profiling.py`: **52 passed**.
- The requested 16-GPU sharded DualPipeV rerun completed under
  `out/20260818_153014`. Its retained comparison was exact: zero missing CPU
  launches and zero missing GPU activities out of 1,318,560 expected records
  each, across 32 actor-mode results and 116,640 paired occurrences. All
  profile validation and simulation endpoint checks passed.
- The exact GPU rerun measured a 568.383 ms hardware median and a 579.805 ms
  instrumented-normal median. Replay predicted 559.082 ms normal and 536.257
  ms interference-free; the endpoint reduction was 4.083% (the paired-median
  reduction estimator was 4.135%). Because this capture was exact, replay
  sensitivity to accepted missing records is zero for this run.
- The first rerun invocation was blocked before `ray.init` by the local command
  sandbox's GCS network isolation. Reissuing the same workload through the
  authorized test-harness command path reached the cluster; no Piper or GPU
  retry workaround was required.

## Superseded proposal: one paired Kineto session

The remaining corruption is specific to opening a second Kineto session in the
same actor process. Fresh-process normal and IF captures matched exactly, and
the same-process double-normal control reproduced the failure. The simplest
general fix is therefore to capture both requested policies in one Kineto
session rather than attempting to tolerate or repair second-session records.

### Implementation adjustment

- Before starting Kineto, execute each requested policy's configured warmup
  iterations from the restored training state so policy-specific caches are
  warm without contributing captured samples.
- Start one profiler session with its normal CUPTI warmup phase, then capture
  paired normal and IF occurrences in deterministic alternating order. Restore
  the same training state before every occurrence as today.
- Preserve the profile mode in every timeline-anchor label. Partition the one
  actor capture into one `ActorProfileResult` per mode from those labels, then
  use the existing cross-mode comparator, exact-spare selection, validation,
  merge, and replay paths unchanged.
- Require every captured node occurrence to belong to exactly one recognized
  mode/iteration anchor and require equal per-mode occurrence counts. Do not
  infer mode from activity order or node identity.
- Keep the 0.01% tolerance limited to identical CPU launch sequences with
  deletion-only GPU activity loss. A launch-sequence difference within the
  shared session remains fatal.

### Evaluation adjustment

- Add focused actor/profiling tests for mixed-mode timeline partitioning,
  missing/duplicate/unknown mode anchors, equal paired occurrence counts, and
  preservation of existing exact-spare selection.
- Rerun the 16-GPU sharded workload with five warmups and 30 requested samples.
  Require one profiler session per actor, no pre-anchor activity, and either an
  exact retained comparison or a qualified deletion-only loss no greater than
  0.01% before replay.

This proposed adjustment was not implemented: the approved deletion-only CPU
loss policy plus strict paired validation completed the requested rerun, and
the resulting capture was exact.

## Completion

### Checks run

- `gpu-venv/bin/python -m pytest -q test/test_profile_validation.py
  test/test_actor_profile_timing.py test/test_profiling.py`: **52 passed**.
- `git diff --check` on the scoped implementation and planning files: passed.
- Two-node, 16-GPU Qwen DualPipeV PP2/virtual-PP4 EP8 run with parameter and
  gradient sharding, batch 8, four microbatches, sequence length 12,288, five
  warmups, 30 hardware iterations, and five profiler warmups plus 30 captured
  iterations: completed successfully. Profile validation, normal replay,
  interference-free replay, and savings attribution all passed.

### Remaining limitations

- Piper can prove that an accepted discrepancy is deletion-only in the
  recorded normalized CPU/GPU sequences, but it cannot independently recover
  an operation omitted from both Kineto callback streams. The strict 0.01%
  cap, detailed diagnostics, and rejection of substitution/reordering bound
  this observational limitation.
- Missing records are never synthesized. If a future run uses the tolerance,
  its measured node sample can be shorter; the loss remains explicit in the
  validation artifact. The completed GPU rerun had no missing records.

### Commit message

`Tolerate bounded deletion-only CPU and GPU profile loss`
