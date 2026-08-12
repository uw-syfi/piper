# Interference benchmarks

This package contains the pairwise GPU-overlap benchmark implementation used
to produce Figure 1, its fixed-20 authoritative driver, and the scripts used to
derive the isolated-resource table. The benchmark design is documented in
`PAIRWISE_OVERLAP_BENCHMARK_PLAN.md`.

The frozen Figure 1 aggregate data, provenance receipts, and rendering tools
live in `benchmark_data/partial_overlap_interference/`.

From the repository root, validate and render the frozen figure with:

```bash
python benchmark_data/partial_overlap_interference/validate_bundle.py
python benchmark_data/partial_overlap_interference/render_figure1.py --output /tmp/pairwise-figure1
```

Inspect the benchmark and authoritative suite interfaces with:

```bash
python scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py --help
python scripts/partial_overlap_interference_benchmarks/run_pairwise_authoritative_12k_coretrend_fixed20.py --help
```

The historical suite manifest in the artifact bundle retains the runner path
used for the original measurement. The authoritative driver maps that legacy
path to the reorganized runner when constructing a new command.
