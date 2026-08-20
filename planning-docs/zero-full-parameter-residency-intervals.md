# ZeRO full-parameter residency intervals

## Phase 1: Scope

### Task summary

Repair the existing two-stage ZeRO parameter-lifetime solution rather than replace it. The existing pruning pass already forms directly connected same-bucket components and removes redundant parameter all-gathers and intermediate local reshard/free markers within each component. The inter-component pass must choose a non-overlapping lifetime order compatible with DAG reachability instead of assuming that component-root topological levels define that order. Components from different buckets, or disconnected components from the same bucket, are never coalesced.

### Objectives

- Treat each directly connected same-bucket component as exactly one parameter-residency interval.
- Keep one parameter all-gather at each interval root, one local full-parameter reshard/free at each interval tail, and no redundant lifetime operations inside the interval.
- Never coalesce across bucket keys or across disconnected same-bucket components.
- Repair inter-chain ordering so a later all-gather cannot be issued until the preceding same-bucket chain's final consumer has completed and its full parameters can be freed.
- Use actual DAG reachability as the proof of an existing order; add the missing temporal edge when the two endpoints are unordered.
- Reject a bucket whose disconnected intervals have no acyclic non-overlapping order compatible with existing DAG dependencies, with the bucket and conflicting boundaries in the diagnostic.
- Diagnose malformed or ambiguous ZeRO lifetime structure during DAG construction rather than relying on runtime freshness assertions.
- Preserve parameter/gradient-sharding correctness for ordinary, interleaved, and disaggregated pipeline schedules.

### Non-goals

- Changing the runtime meaning of parameter all-gather, deferred local parameter free, or gradient reduce-scatter.
- Removing or coalescing `REDUCE_SCATTER_COMM` nodes merely because parameter residency intervals merge. Those nodes synchronize accumulated gradients and are not parameter reshard operations.
- Changing gradient accumulation intervals except where existing metadata must remain intact while parameter-lifetime nodes are rewired.
- Changing P2P dispatch-level placement, dispatch tie priorities, stream semantics, schedule directives, or executor/runtime freshness checks.
- Replacing the current directly connected component construction with a global lifetime optimizer.
- Merging disconnected components, regardless of whether their positions interleave in one linearization.

### Correctness criteria

1. For every parameter-sharded bucket, every trainable parameter consumer executes after exactly one live interval's all-gather and before that interval's reshard/free point.
2. A bucket has at most one live full-parameter interval at any point permitted by the DAG: every later chain's all-gather is reachable from the preceding chain's final consumer/free boundary.
3. No surviving all-gather can execute while the same bucket's full parameters are already resident, and no surviving reshard/free can invalidate parameters still required by a consumer in its interval.
4. Within each directly connected same-bucket component, later all-gathers and earlier reshard/free markers are removed; all data and temporal dependencies unrelated to the eliminated lifetime operation remain satisfied. No operation is removed across disconnected components or bucket keys.
5. Between distinct same-bucket chains, the finalized DAG contains an explicit ordering path from the earlier chain's final consumer/free boundary to the later chain's all-gather.
6. Each parameter consumer belongs to one interval, each interval has one surviving all-gather and one final reshard/free marker, and malformed structures fail during DAG construction with bucket and node identities in the diagnostic.
7. Gradient contribution coverage, allocation metadata, and the one required gradient synchronization per lowered bucket remain unchanged by parameter interval normalization.
8. Final execution dispatch order respects every normalized lifetime edge without any executor or simulator exception, relaxation, or extra synchronization outside the DAG.
9. Existing valid unsharded, ZeRO-2, and ZeRO-3 schedules retain their intended externally observable training semantics.

### Assumptions / open questions

- In the current implementation, parameter resharding is a local deferred free represented by `zero_free_full_params_after`; `REDUCE_SCATTER_COMM` is gradient synchronization. This task uses “scatter” in the parameter-lifetime sense and does not treat gradient reduce-scatter as the close of a parameter interval.
- Bucket identity is a hard partition. Within a bucket, the existing direct data/temporal connectivity defines interval membership; paths through another bucket do not connect components.
- A lower numeric topological level is not a happens-before proof. A reverse boundary path rules out that proposed interval direction and imposes the opposite interval precedence; reject only when the resulting per-bucket precedence constraints are cyclic.
- Dispatch levels need not participate in the interval algorithm. `assign_dispatch_levels` runs after ZeRO normalization and recomputes levels from the finalized DAG. Only P2P send/receive nodes are moved away from their ordinary topological levels; all-gather, parameter consumers, and reshard/free ownership remain governed by DAG reachability. Any changed topology will therefore be reflected automatically when dispatch levels are assigned, and the existing edge validation will constrain any affected P2P endpoint movement.

## Phase 2: Implementation and evaluation plan

### Implementation plan

1. **Finalize compute order before pruning.** After actor-DAG splitting, materialize default-compute order before calling `_prune_zero_lifetime_metadata()`. Mirror that sequence in focused lowering helpers. This gives parameter lifetime pruning the compute order the executor and simulator will use without involving P2P dispatch levels.
2. **Keep existing lifetime pruning.** Retain the current direct same-bucket component construction and its all-gather removal/rewiring and free-marker normalization. Retain gradient accumulation/reduce-scatter normalization unchanged. Do not merge components after this pass.
3. **Choose a feasible inter-component order.** For each bucket and lifetime type, represent each disconnected component by its start and end boundaries. Existing reachability determines mandatory interval precedence: for parameter intervals, if `gather(B)` reaches `tail(A)`, B must precede A because A-before-B would make `tail(A) -> gather(B)` cyclic. Build the analogous constraints for gradient intervals (`root(B) -> reduce_scatter(A)`). Stable-topologically sort this small per-bucket constraint graph, using deterministic component position only to break unconstrained ties. Reject a constraint cycle with the bucket and boundary identities.
4. **Materialize non-overlap.** In the selected interval order, retain an existing end-to-next-start path or add the temporal edge between consecutive intervals. Revalidate incrementally so every interval remains separate and no edge creates a cycle. Parameter boundaries remain `previous tail -> next all-gather`; gradient boundaries remain `previous reduce-scatter -> next allocation root`.
5. **Apply lifetime constraints before physical-stream FIFO.** Apply inter-interval lifetime edges before `resolve_total_order_per_stream()` chooses FIFO for gather/reduce streams, so stream serialization respects lifetime boundaries. Continue assigning dispatch levels only after stream FIFO is finalized. No executor, simulator, runtime, directive, or schedule JSON changes are planned.
6. **Retain runtime invariants.** Keep the `full_params_fresh` assertion in `all_gather_full_params()` as defense in depth; successful schedules should no longer reach it because of a missing DAG lifetime edge.

#### Compatibility risks

- Valid DAGs that previously relied only on numeric topological levels will gain an explicit temporal edge. This is the intended correction and may change actor-DAG digest fixtures.
- A schedule whose disconnected same-bucket components cannot be linearly ordered as separate lifetimes will now fail during DAG construction. Piper will not silently extend full-parameter residency across that disconnection.
- Constraint-derived interval order may differ from component-root topological order. This is intentional: root depth is not a proof of a feasible lifetime order.
- The shared reachability helper also protects multiple gradient allocation intervals. Normal lowering currently produces one accumulated gradient interval per bucket, so this should be behavior-preserving; focused tests will confirm the gradient reduce-scatter structure is unchanged.
- Adding the lifetime edge can indirectly change recomputed topological and P2P dispatch levels. This is not a feedback dependency: dispatch assignment remains a downstream projection that must satisfy all DAG edges.

### Evaluation plan

1. **Missing happens-before regression (criteria 1–5, 8):** retain the schedule-neutral unit case with two disjoint same-bucket parameter intervals whose lifetime boundaries are unordered. Verify inter-chain finalization adds `previous tail -> next all-gather`, dispatch assignment succeeds, and the next gather is issued after the previous tail.
2. **Bucket isolation and direct connectivity (criteria 1, 3, 4, 6):** add the generalized order `A0 -> B0 -> A1 -> B1`. Verify four intervals when A and B are distinct bucket keys because cross-bucket paths do not connect same-bucket nodes; verify one interval when all four nodes share a bucket and are directly connected by those edges.
3. **Constraint-derived ordering and rejection (criteria 2, 5, 6):** test disconnected same-bucket intervals where one apparent root order is cyclic but the reverse interval order is feasible; verify Piper selects the feasible order without coalescing. Add a case where neither interval order is feasible and verify a diagnostic with bucket and boundary context.
4. **Internal coalescing and gradient preservation (criteria 3, 4, 7):** run and, only if needed, extend the existing same-chain pruning and sharded-gradient tests. Verify one gradient reduce-scatter per bucket, unchanged contributor coverage, and unchanged gradient allocation roots.
5. **Lowering regression (criteria 1–9):** run focused ZeRO equivalence cases plus the existing real 397B control-plane lowering that exposed overlapping component spans. Review digest changes as explicit lifetime-node/edge differences before updating expectations.
6. **End-to-end parameter parity (criteria 7, 9):** run the existing small Qwen `zero3_1f1b` parameter-parity case, which exercises parameter and gradient sharding against the single-model reference.
7. **Motivating workload (criteria 1–3, 8–9):** rebuild and rerun the same 16-GPU DualPipeV PP2/virtual-PP4, EP8, parameter/gradient-sharded Qwen workload. Inspect the actor DAG first, then verify warmup and profiler capture complete without a freshness assertion.

Only these focused unit, lowering, end-to-end, and motivating workload checks are planned; the entire unit or end-to-end suite is out of scope for this narrow repair.

## Phase 3: Implementation log

- Repaired `_add_inter_chain_temporal_edges()` without changing chain construction: forward reachability is retained, unordered boundaries gain the required temporal edge, and reverse reachability raises a construction-time diagnostic.
- Added deterministic serial-topological position as the tie-break for same-level chain roots; no dispatch-level input or feedback was introduced.
- Added generalized parameter-lifetime tests for missing, existing, and contradictory reachability. The missing-edge case also verifies finalized dispatch order.
- Focused CPU results so far: `test/test_order_directive.py` passed (44 tests); `test/test_dag_builder_equivalence.py` passed (4 tests), with no digest updates required.
- The first focused GPU parity attempt exposed a sequencing issue not visible in the synthetic cases: both selected Qwen sizes failed during construction because finalized `dp_stream` FIFO already ordered a later all-gather before the preceding chain tail (`bucket='s0.seg2'`, `all_gather.5->s0.seg2`). This is not a valid schedule-level reverse lifetime; it is created by resolving unconstrained gather FIFO before applying lifetime boundaries. Resolution: move existing inter-chain enforcement before physical-stream FIFO finalization, then rerun the focused CPU and GPU checks. No interval merging or runtime workaround is introduced.
- After moving lifetime enforcement before physical-stream FIFO, the real 397B CPU lowering still found that component-root order selected an infeasible boundary (`bucket='s1.seg15'`, `all_gather.28->s1.seg15.bwd.splitMB1`). The proposed overlap-coalescing response was rejected: bucket keys and direct connectivity remain hard interval boundaries. Revised resolution: derive a feasible order among disconnected intervals from boundary reachability, or reject the schedule if that constraint graph is cyclic.
- The existing pruning result mixed parameter and gradient chains in one list and the inter-chain pass inferred type from shared node metadata. Backward nodes participate in both lifetimes, so this duplicated intervals in the parameter constraint graph. Resolution: return an explicitly typed internal `_ZeroLifetimeChains` value; no model tags or node-kind heuristics determine lifetime type.
- Implemented stable per-bucket interval constraint ordering. Cross-bucket paths never connect intervals; disconnected same-bucket intervals remain separate and receive end-to-next-start edges in a reachability-compatible order.
- A later two-microbatch AF-Pipe ZeRO-3 build exposed a branched same-bucket component: topologically sorting the undirected component placed nodes without a direct dependency next to one another. Split every such component into deterministic direct-edge paths before pruning gathers, preserving the approved rule that coalescing occurs only between consecutive directly connected same-bucket consumers.

### Evaluation results

- `gpu-venv/bin/python -m pytest -q test/test_order_directive.py test/test_dag_builder_equivalence.py`: **51 passed**.
- `gpu-venv/bin/python -m pytest -q test/test_afd_full_lowering.py test/test_qwen_control_plane_workloads.py -k 'not real_397b_meta_trace_bucketing_increases_finalized_dag_size'`: **8 passed, 1 deselected**.
- `test/test_qwen_e2e_correctness.py::test_qwen_parameter_parity[zero3_1f1b-9M]` on the active Ray cluster: **1 passed** after two training steps.
- Exact 16-GPU Qwen-30B DualPipeV PP2/virtual-PP4, EP8, parameter/gradient-sharded regression with two profiler warmups and one active iteration: **passed**. The finalized actor DAG contains explicit lifetime edges such as `s0.seg0 -> all_gather.2`; the combined third-iteration trace is `out/20260818_115853/pytorch_profile_dualpipev_pp2_world16_mbs4.json`.
- `test_real_397b_meta_trace_bucketing_increases_finalized_dag_size` could not complete: after the lifetime-cycle issue was removed, its bucketed build failed earlier in A2A insertion because `s0.seg0.bucket5->s0.seg1` lacks `runtime_outputs_by_destination`. This routed bucketing failure occurs before ZeRO lifetime normalization and was not changed under this task.
- `git diff --check` passed for the changed files. The full unit and end-to-end suites were not run, per the focused evaluation plan.
- Follow-up branched-component verification: `test/test_order_directive.py` plus `test/test_dag_builder_equivalence.py`: **52 passed**. Both two-microbatch Qwen3-30B-A3B-L1 AF-Pipe ZeRO-3 layouts then completed 5/30 profile-and-replay and separate 5/10 throughput runs; all replay validators passed with zero CPU/GPU capture loss. Artifacts are under `out/20260819_101639`, `out/20260819_101929`, `out/20260819_102159`, and `out/20260819_102255`.

### Remaining known limitations

- A schedule whose disconnected intervals for one bucket impose cyclic lifetime precedence is rejected; Piper does not merge those components or retain parameters across their disconnection.
- The unrelated bucketed routed-value mapping failure above remains unresolved.

### Commit message

`Fix ZeRO full-parameter lifetime ordering`

Follow-up: `Split branched ZeRO parameter lifetime components`
