# SM-free pairwise overlap matrix

This directory contains the single-node comparison of normal NCCL and NCCL
2.28.3 Zero-CTA collectives. The matrix is fixed to three pair families:

- grouped GEMM with balanced A2A (`A03:C-GROUPED`);
- grouped GEMM with all-gather (`A04:C-GROUPED`); and
- all-gather with balanced A2A (`A14`).

Each case is measured in both victim orientations and both launch orders. The
prescribed-overlap run targets 0, 5, 25, 50, 75, and 100 percent overlap and
requires exactly 20 estimator-valid repetitions per data point. The optional
zero-gap run submits each pair back-to-back in both orders and produces a
launch-order table.

Balanced A2A uses 201,326,592 BF16 elements (384 MiB) in and out per rank at
sequence length 12,288, split into eight equal 48-MiB peer slices. Normal mode
uses `torch.distributed`; SM-aware mode uses the existing `ncclAllGather` and
`ncclAllToAll` implementations with `NCCLConfig(cta_policy=CTAPolicy.Zero)`.
The Zero-CTA buffers are allocated with `ncclMemAlloc` and registered as
`CollSymmetric` windows.

Install the repository-local NCCL 2.28.3 overlay once:

```bash
python scripts/partial_overlap_interference_benchmarks/sm_free_collectives/pairwise_overlap/install_nccl_228_overlay.py
```

Then configure the environment and validate a run without launching GPU work:

```bash
source scripts/partial_overlap_interference_benchmarks/sm_free_collectives/pairwise_overlap/setup_nccl_228_zero_cta_env.sh
python scripts/partial_overlap_interference_benchmarks/sm_free_collectives/pairwise_overlap/run_pairwise_collective_mode.py \
  --mode sm-aware --node-ip 10.65.10.75 \
  --output-dir /shared/results/pairwise-sm-aware
```

Add `--execute` to run the benchmark. Use `--mode nccl` for the normal-NCCL
comparison, or `--zero-gap-order-only` for the launch-order experiment.

`run_a2a_ag_backend_matrix.py` runs the four normal/Zero-CTA combinations for
the A2A/all-gather pair and renders their paired slowdown and makespan bars.
