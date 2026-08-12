# One-off analysis and removal candidates

This inventory identifies experiment-specific scripts that should not be
promoted into the reusable script packages. No file listed here was deleted by
the reorganization.

## Safe immediate cleanup

| Script | Reason |
|---|---|
| `benchmark_pairwise_overlap_cases.py.orig` | Unmanaged backup superseded by the committed partial-overlap runner. |
| `gpm_probe_one_gpu.py` | NVML GPM sampling-support smoke test. |
| `probe_ncu_physical_metrics.py` | One-time permissions probe for `ERR_NVGPUCTRPERM`. |
| `probe_pairwise_physical_metrics.py` | One-time physical-counter capability probe. |

## Superseded partial-overlap campaign drivers

| Script | Reason / prerequisite before removal |
|---|---|
| `run_pairwise_overlap_suite.py` | Incomplete precursor to the authoritative fixed-20 suite. |
| `run_pairwise_authoritative_12k_adaptive.py` | Adaptive campaign superseded by fixed-20 collection. |
| `postprocess_pairwise_authoritative_12k_adaptive.py` | Remove with the superseded adaptive driver after preserving provenance. |
| `convert_a20_pairwise_results.py` | Pilot-only converter; remove after pilot provenance is frozen. |
| `enrich_pairwise_overlap_results.py` | Migration helper; remove after confirming all retained corpora use the final schema. |
| `validate_pairwise_pilot_traces.py` | Pilot acceptance tool; remove after its receipts are frozen. |

## Workload-specific original-interference campaigns

These scripts produced particular Qwen397B or EP experiments rather than a
general benchmark API. Archive their result provenance before deletion.

```text
analyze_ep_collective_overlap_trace.py
estimate_ep_trace_interference_slowdown.py
analyze_ragged_alltoall_overlap.py
benchmark_qwen397b_a2a_allreduce.py
benchmark_qwen397b_a2a_backward.py
benchmark_qwen397b_bidirectional_p2p.py
benchmark_qwen397b_p2p_allgather.py
run_qwen397b_a2a_gradient_matrix.py
run_nccl_launch_order_ab.py
analyze_qwen397b_a2a_gradient_matrix.py
summarize_qwen397b_p2p_comm_results.py
summarize_a2a_slowdowns.py
benchmark_launch_order_interference_predictor.py
```

`analyze_ep_collective_overlap_trace.py` also fails Python 3.10 compilation due
to backslashes inside f-string expressions; fix it only if the historical
analysis is retained.

## Profile-and-replay report migrations

| Script | Reason / prerequisite before removal |
|---|---|
| `analyze_reduce_trace_nodes.py` | Narrow reduce-trace investigation rather than a reusable replay diagnostic. |
| `recompute_qwen_sharding_paired_if_savings.py` | One-time Qwen report migration; remove after the paired-savings table is finalized. |

## Probe-like scripts outside these workstreams

The following probe-like scripts should either become tests under `test/` or
be removed after their results are captured; they should not remain as
permanent top-level scripts:

```text
test_piper_ragged_all_to_all.py
test_torchtitan_expert_parallel_fullgraph.py
test_torchtitan_grouped_experts_fullgraph.py
```
