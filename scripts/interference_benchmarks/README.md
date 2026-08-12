# Original interference benchmarks

These are the earlier trace-driven interference experiments that preceded the
partial-overlap suite in `scripts/partial_overlap_interference_benchmarks/`.

- `launch_order/` contains the grouped-GEMM/collective SM-sharing benchmark and
  the trace-driven launch-order benchmark and analyzer.
- `collectives/` contains the distributed all-reduce microbenchmark and its
  isolated-trace analyzer.
- `probes/` contains the CUDA SM-ID probes used by the launch-order studies.
