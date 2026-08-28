# Remove simulation bootstrap resampling

## Phase 1: Scope

### Task summary

Replace bootstrap-based profile-and-simulate summaries with conventional
mean-based estimates: arithmetic-mean normal and interference-free (IF) replay
times with sample standard deviations and two-sided 95% Student-t confidence
intervals, plus paired absolute and percentage savings estimates that preserve
normal/IF iteration pairing. Retain medians only as secondary diagnostics.

### Objectives

- Report arithmetic-mean simulated normal and IF step times.
- For each endpoint mean, report the sample standard deviation and a two-sided
  95% Student-t confidence interval for the population mean.
- Report absolute savings as the arithmetic mean of paired differences
  `normal_i - IF_i`.
- Report percentage savings as the ratio of endpoint means,
  `100 * (mean_normal - mean_IF) / mean_normal`, with a two-sided 95%
  paired delta-method confidence interval that includes the observed
  normal/IF covariance.
- Retain each endpoint median as a secondary descriptive diagnostic, without a
  confidence interval.
- Remove bootstrap resampling and per-iteration percentage averaging from
  simulator result calculation and reporting.
- Keep the in-process profile-and-simulate harness, offline replay CLI,
  validation, and savings attribution consistent with the revised endpoint
  estimator.

### Non-goals

- Changing DAG replay, CUPTI capture, retained-iteration selection, sample
  pairing, or the number of profiling iterations.
- Removing raw normal/IF replay samples used for diagnostics and validation.
- Changing statistics for ordinary hardware timing, memory measurement,
  control-plane benchmarking, or partial-overlap interference benchmarking.
- Removing the generic bootstrap-capable helper where unrelated consumers still
  use it.
- Changing Monte Carlo feature-order sampling used to apportion endpoint
  savings among attribution features.

### Correctness criteria

- Each reported simulated endpoint point estimate is the arithmetic mean of
  its retained replay samples; its `std` is the Bessel-corrected sample standard
  deviation and its 95% interval is `mean +/- t_(0.975,n-1) * std/sqrt(n)`.
- The secondary endpoint median equals the mathematical median of the same
  samples and has no associated bootstrap confidence bounds.
- Normal and IF samples remain aligned by profile iteration ID. Absolute
  savings equals `mean(normal_i - IF_i)`, which also equals
  `mean_normal - mean_IF` up to floating-point roundoff.
- Percentage savings is the ratio of means
  `100 * (mean_normal - mean_IF) / mean_normal`; no result is formed by
  averaging or taking the median of per-iteration percentages.
- The percentage-savings standard error uses a paired delta-method estimate
  over the sample variance/covariance of `(normal_i, IF_i)`, and its 95%
  interval uses `t_(0.975,n-1)` without clipping negative or above-100%
  bounds.
- No reported simulator endpoint or savings value depends on a bootstrap seed,
  bootstrap repetition count, or resampled copies of captured iterations.
- The harness and offline replay CLI report identical point estimates,
  dispersion, confidence intervals, and medians for the same profile artifact.
- Replay validation and attribution compare or telescope against arithmetic-mean
  simulated endpoints rather than the former median endpoints.
- For a single aligned sample, point estimates and medians remain available,
  while sample standard deviations and Student-t/delta-method confidence
  intervals are explicitly unavailable rather than reported as zero-width
  evidence.

### Assumptions / open questions

- The paired delta-method interval is the first-order interval for
  `100 * (1 - mean_IF / mean_normal)` using the full sample covariance matrix
  of paired normal/IF observations and `n - 1` degrees of freedom.
- A non-positive normal mean is invalid for percentage savings. Step-time input
  validation already requires non-negative samples; the zero-mean case must be
  rejected explicitly.
- The revised simulator output schema intentionally replaces the existing
  paired-percentage samples, `paired_median` estimator label, and bootstrap CI
  fields. Raw normal and IF timing samples remain available.
- Statistics for measured or instrumented hardware timing remain outside this
  change unless they are used directly as a simulator endpoint comparison;
  their existing summaries are not globally redefined.

## Phase 2: Implementation and evaluation plan

### Implementation plan

1. Add dependency-free mean and paired-savings summaries in
   `src/profile_statistics.py`.
   - Represent endpoint sample count, arithmetic mean, sample standard
     deviation, Student-t 95% bounds, and diagnostic median without bootstrap
     fields or seeds.
   - Represent paired absolute-savings mean and the ratio-of-means percentage,
     including its paired delta-method standard error and Student-t 95% bounds.
   - Add a deterministic 97.5th-percentile Student-t calculation suitable for
     any supported positive degrees of freedom, without adding SciPy as a
     runtime dependency.
   - Keep the current bootstrap helper available only for unrelated existing
     consumers; simulator comparison code will no longer call it.

2. Change `simulate_profile_modes()` and `SimulationComparisonResult` in
   `src/simulator.py` to use the new summaries.
   - Summarize replayed normal/IF iteration makespans with arithmetic means.
   - Replace paired percentage samples and their median/bootstrap summary with
     one paired absolute/percentage savings summary.
   - Base simulator prediction errors and normal-to-IF reduction fields on the
     revised endpoint means.
   - Preserve raw samples and aligned iteration-ID validation.

3. Keep endpoint consumers consistent.
   - Update `src/attribution.py` so attribution objectives use the arithmetic
     mean across aligned replay iterations and continue to telescope exactly to
     the reported normal/IF endpoint difference.
   - Allow `src/profile_validation.py` to consume the revised endpoint summary
     without changing its capture-integrity or validation policy.

4. Update profile-and-simulate serialization and tabular reporting.
   - In `examples/run_harness.py`, emit normal/IF mean, sample SD, Student-t
     bounds, median diagnostics, paired absolute savings, ratio-of-means
     percentage savings, and its delta-method bounds; remove simulator
     bootstrap-CI and per-iteration-percentage fields.
   - Make the equivalent schema update in
     `scripts/profile_and_replay/simulate_step_times.py`.
   - Preserve raw normal/IF sample arrays for diagnostic tooling.

5. Update focused documentation or field descriptions only where existing
   profile-and-replay output semantics are described; do not alter unrelated
   benchmark statistics or schemas.

### Compatibility and implementation risks

- This intentionally changes simulator JSON/CSV fields and the meaning of the
  primary endpoint `estimate` from median to mean. Downstream consumers of the
  removed bootstrap/paired-percentage fields must migrate to the explicit
  mean, median, and paired-savings fields.
- Student-t and delta-method intervals require at least two aligned samples.
  The schema must represent unavailable dispersion/intervals for legacy
  one-sample artifacts without fabricating certainty.
- The ratio-of-means interval is an asymptotic first-order approximation and
  can be wide or extend beyond conventional percentage bounds when the normal
  mean is noisy; it will be reported without clipping.
- Attribution's feature-order Monte Carlo confidence diagnostics are distinct
  from bootstrap resampling of replay iterations and remain unchanged.

### Evaluation plan

1. Extend `test/test_profile_statistics.py` with focused numeric tests for:
   - arithmetic mean, Bessel-corrected sample SD, median, and a known
     Student-t 95% interval;
   - paired absolute savings and the covariance-aware delta-method percentage
     interval against an independently calculated expected value;
   - a dataset where the ratio of means differs from the mean and median of
     per-iteration percentages;
   - negative savings, zero variance, one-sample unavailable intervals,
     unequal pair counts, and a zero normal mean.

2. Update focused simulator tests in `test/test_simulator.py` to verify:
   - replay samples are consumed once without bootstrap resampling;
   - endpoint point estimates are means and medians remain diagnostic;
   - paired absolute savings and ratio-of-means percentage semantics;
   - nonlinear cross-stream makespan replay is reduced after each iteration is
     simulated, not by replaying per-node means.

3. Update `test/test_simulator_attribution.py` so attribution uses the same
   mean endpoint objective and remains additive to the reported absolute
   savings.

4. Add or extend focused output tests for `examples/run_harness.py` and the
   offline `scripts/profile_and_replay/simulate_step_times.py` serializer to
   assert matching fields, values, unavailable one-sample intervals, and the
   absence of bootstrap/per-iteration-percentage result fields.

5. Run the narrow suite first:
   - `test/test_profile_statistics.py`
   - the profile-comparison cases in `test/test_simulator.py`
   - `test/test_simulator_attribution.py`
   - the focused harness/offline-output tests added above

6. Then run the existing simulator, profile validation, and profile reporting
   unit tests to catch cross-component schema and estimator regressions. Replay
   an existing multi-iteration profile artifact through both the in-process
   result path and offline CLI and compare their emitted summaries. No GPU run
   is required because this change operates only on already captured samples.

## Phase 3: Implementation log

- Added dependency-free `MeanStatistics` and `PairedSavingsStatistics`
  summaries. Endpoint intervals use a numerically evaluated Student-t critical
  value; paired percentage intervals use the full normal/IF sample covariance.
- Kept the existing bootstrap helpers for unrelated callers, but removed every
  use of them from simulated endpoint and normal-to-IF savings reporting.
- Changed simulator endpoints to replay each retained iteration first and take
  the arithmetic mean of the resulting makespans. This preserves nonlinear DAG
  and stream effects that would be lost by replaying per-node averages.
- Changed savings attribution to use the same arithmetic-mean replay objective,
  preserving exact telescoping to the reported endpoint difference.
- Replaced paired per-iteration percentage samples and the `paired_median`
  summary in profile-and-simulate output with paired absolute savings and a
  ratio-of-means percentage summary. Raw normal/IF timing samples remain.
- Single-sample artifacts report their point estimate and median while emitting
  `null` sample SD and confidence bounds.
- Ordinary measured/instrumented timing summaries retain their existing
  semantics; this change is scoped to simulated normal/IF endpoints and their
  derived savings.

## Completion

### Evaluation performed

- Focused statistics tests: `test/test_profile_statistics.py` — **10 passed**.
- Focused simulator, attribution, and output integration tests — **58 passed**.
- Combined relevant suite covering statistics, simulator, attribution, harness
  and offline reporting, profile validation, profiling, actor profile timing,
  profile streams, and node-ratio export — **122 passed**.
- Python compilation for all changed Python files — **passed**.
- `git diff --check` for all scoped files — **passed**.
- Offline replay of
  `out/20260818_153014/step_time_profiles.json` — **passed** with 30 aligned
  samples. It reported normal mean `559260.722 us` (median `559081.689 us`), IF
  mean `536211.824 us` (median `536256.587 us`), absolute savings
  `23048.898 us`, and ratio-of-means savings `4.1213%` with paired delta-method
  95% interval `[4.0224%, 4.2202%]`; removed bootstrap fields were absent.
- No GPU run was needed for this sample-reduction-only change.

### Remaining limitations

- Student-t and paired delta-method uncertainty require at least two samples;
  one-sample uncertainty is reported as unavailable.
- The delta-method percentage interval is a first-order approximation and is
  intentionally not clipped to a nominal percentage range.
- The generic bootstrap helpers remain for unrelated repository consumers, but
  simulator result production no longer calls them.

### Concise commit message

`Report profile replay means with Student-t and paired savings intervals`
