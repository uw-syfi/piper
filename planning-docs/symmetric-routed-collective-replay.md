# Symmetric Routed-Collective Replay

## Task summary

Model each `A2A_COMM`, `M_TO_N_COMM`, and `N_TO_M_COMM` DAG node as one logical
symmetric all-participant rendezvous followed by an endpoint-specific aggregate
GPU-envelope tail. Symmetry describes group readiness, not equal payload sizes:
uneven and zero-byte routing remain supported.

## Objectives

- Require every declared participant in a routed communication node to arrive
  before any endpoint begins its replayed aggregate tail.
- Remove the directional source/destination timing model for M-to-N/N-to-M and
  the independent-endpoint timing model for A2A.
- Calibrate endpoint-specific aggregate service from aligned CUPTI occurrences
  without retaining initial peer-arrival wait in the service estimate.
- Apply collective identity, participant, activity, and alignment validation
  consistently to all routed collective kinds.

## Non-goals

- Inferring or validating split sizes or semantic routing correctness from
  schedules or profiles.
- Separately modeling a routed node's metadata collectives, payload collective,
  local preparation kernels, or internal collective skew.
- Modeling individual peer transfers, network contention, communicator
  occupancy, or CPU dispatch as a separate simulated resource.
- Changing runtime communication, DAG construction, schedules, P2P replay, or
  dense-state transfer behavior.
- Predicting another routing distribution or physical topology from a captured
  profile.

## Correctness criteria

1. A routed transaction is identified by its communication kind and logical
   communication ID, and has exactly its complete, consistently declared
   participant set.
2. In both normal and interference-free replay, each A2A, M-to-N, or N-to-M
   node has one shared rendezvous of all transaction participants.
3. Each endpoint retains its own calibrated aggregate post-rendezvous tail.
   Unequal and zero-byte endpoint payloads are valid, and initial peer-arrival
   wait does not become service time on an earlier endpoint.
4. Routed replay requires aligned endpoint iterations and rejects missing or
   inconsistent IDs, participants, samples, or timing beyond the declared
   clock-alignment uncertainty. A node endpoint must be present on every
   participant even when it produces no visible CUDA activity.
5. DAG dependencies, physical-stream FIFO, and independence between distinct
   communication IDs remain unchanged.
6. Inconsistent endpoint ordering that makes the all-participant transaction
   graph cyclic is rejected rather than relaxed by the replay engine.

## Subtasks

- Replay each routed communication node through the existing symmetric
  collective rendezvous path.
- Extend profile/alignment validation to A2A and zero-activity participants;
  remove obsolete directional replay logic.
- Replace directional/independent routed replay tests with symmetric
  rendezvous, uneven-payload, validation, and regression coverage.

## Assumptions / open questions

- Symmetry means common readiness, not equal payload or endpoint duration.
  Unequal group widths, uneven splits, zero-byte endpoints, and
  endpoint-specific aggregate tails remain supported. Predictions remain tied
  to the captured payload distribution and topology.
- Forward M-to-N dispatch currently contains two metadata all-gathers followed
  by a payload all-to-all. A2A forward dispatch contains a count all-to-all
  followed by a payload all-to-all; other routed paths may contain only the
  payload collective. The aggregate model assumes every internal collective in
  one node uses the same complete participant set and that all participants
  invoke those collectives in the same fixed order.
- The aggregate tail includes NCCL kernels, non-NCCL preparation kernels,
  inter-kernel GPU gaps, and later internal collective waiting. Local work before
  the first collective is effectively placed behind the node rendezvous, while
  wait at later internal collectives remains baked into the endpoint tail.
  Replay therefore does not recompute counterfactual internal-phase skew.
- The current durable `NodeProfile` already contains the node-wide envelope and
  aligned first-activity offset needed by this model; no per-phase profile
  format or kernel-name inference is required.
- CPU work before the first or after the last GPU activity remains outside the
  current GPU-only simulation. CPU-caused gaps between captured activities are
  included in the aggregate envelope.
- A symmetric model can expose cycles in schedules that order overlapping
  routed transactions inconsistently across participants. Treating such a
  cycle as invalid is intentional under this replay model.

## Implementation plan

1. In `src/simulator.py`, classify A2A, M-to-N, and N-to-M with the existing
   symmetric collective kinds. Group their rank-local endpoints by kind and
   communication ID, validate the common participant set, and construct the
   same arrival -> shared rendezvous -> endpoint completion vertices used by
   reductions and parameter collectives.
2. Remove the directional routed role/service decomposition and independent
   A2A endpoint path. Keep source/destination metadata available to the runtime
   and DAG validation, but do not use it to alter replay readiness.
3. Reuse aligned node-envelope calibration for endpoint tails. For a routed
   endpoint with no visible CUDA activity, retain its aligned node arrival and
   assign zero service rather than rejecting a mixture of zero- and nonzero-
   duration endpoints. Keep the stricter mixed-activity rejection for
   non-routed collectives.
4. In `src/profile_validation.py`, include A2A in rendezvous-collective identity
   and occurrence validation. Permit mixed visible activity only for routed
   collective kinds while continuing to require an endpoint record and
   communication ID for every participant.
5. Update the simulator module documentation and the routed rows of
   `planning-docs/communication-replay-audit.md` to state the aggregate
   symmetric model and its internal-phase approximation.
6. Update focused simulator and profile-validation tests to assert the new
   readiness, uneven/zero-byte, malformed-transaction, and cycle behavior.

This requires no profile-schema, CUPTI attribution, executor, DAG-builder, or
schedule change. Existing profiles remain usable when they contain aligned
per-iteration endpoint offsets and complete routed participant metadata; an
older A2A profile without cross-actor alignment must be regenerated.

### Compatibility and limitations

- Replay no longer permits a source endpoint to complete independently of the
  other participants in the same M-to-N/N-to-M node. This is an intentional
  semantic correction and may change critical paths or expose cycles in old
  profiles.
- The model is faithful only when every internal collective within one routed
  node has the same participant set and every participant invokes those calls
  in the same order. Runtime routed helpers currently satisfy this contract.
- A single aggregate tail cannot recompute counterfactual skew at later
  internal collectives. It may retain later rendezvous wait and relocates
  pre-first-collective local work behind the logical node rendezvous.
- Zero visible activity is treated as zero endpoint service, not proof of a
  missing collective call. Existing normal/IF launch-equivalence and capture-
  loss validation remain responsible for detecting missing capture records.

## Evaluation plan

1. **Complete identity and participants (criterion 1):** extend simulator tests
   for A2A and unequal-width M-to-N/N-to-M transactions; retain failures for
   mismatched communication IDs, incomplete participants, and inconsistent
   participant declarations.
2. **Symmetric readiness (criterion 2):** give one endpoint a later DAG arrival
   and verify every routed endpoint begins service only after the common
   rendezvous, for A2A and parameterized M-to-N/N-to-M cases.
3. **Endpoint-specific uneven/zero service (criterion 3):** use aligned endpoint
   profiles with unequal tails and a zero-duration participant; verify the zero
   endpoint completes at rendezvous and nonzero endpoints retain their distinct
   calibrated tails.
4. **Profile validation (criterion 4):** verify A2A requires a communication ID
   and aligned samples, routed mixed zero/nonzero activity is accepted, and the
   existing non-routed mixed-activity and clock-uncertainty failures remain.
5. **DAG/FIFO isolation (criterion 5):** retain existing dependency and
   same-stream tests, and verify separate communication IDs do not share a
   rendezvous.
6. **Cyclic ordering (criterion 6):** replace the prior directional-order test
   with a general two-transaction inconsistent-order case that must raise a
   simulation-cycle validation error.
7. Run the CPU-only focused suites:
   `test/test_simulator.py`, `test/test_profile_validation.py`, and
   `test/test_simulator_attribution.py`.

## Implementation log

- Unified A2A, M-to-N, and N-to-M with the simulator's symmetric collective
  transaction path. Routed replay no longer reads source/destination roles.
- Kept participant metadata mandatory for routed transactions and retained
  complete-set validation before constructing the shared rendezvous.
- Allowed a routed endpoint with no visible CUDA activity to contribute its
  arrival and zero service. Mixed activity remains an error for reductions and
  parameter collectives.
- The first full focused run exposed one attribution fixture whose early source
  envelope ended before the new symmetric rendezvous. The fixture represented
  the removed directional model; its source envelope was corrected to retain
  the same 2 us aggregate post-rendezvous tail.
- No implementation deviation from the approved aggregate model was required.

## Evaluation results

- `gpu-venv/bin/python -m py_compile src/simulator.py src/profile_validation.py test/test_simulator.py test/test_profile_validation.py`: passed.
- Targeted routed simulator/profile-validation selection: 20 passed.
- `gpu-venv/bin/python -m pytest -q test/test_simulator.py test/test_profile_validation.py test/test_simulator_attribution.py`: 75 passed, with one expected warning from a synthetic validation case that intentionally has no ordinary measured-normal samples.
- `git diff --check` on the changed replay, test, audit, and planning files:
  passed.
- GPU evaluation was not run because replay and its focused tests are CPU-only.

Known limitations remain the approved aggregate-envelope approximation: fixed
internal participant sets/order are assumed, later internal rendezvous wait is
not recalibrated, and pre-first-collective local work is placed behind the node
rendezvous.

Proposed commit message: `Model routed communication as symmetric replay collectives`
