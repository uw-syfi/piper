# NCCL 2.28 Zero-CTA overlay

The pairwise matrix uses an overlay isolated from the container's system
packages and PyTorch wheel. It pins NCCL 2.28.3 and `nccl4py` 0.2.0, then
extracts NVIDIA's CUDA 12.5 forward-compatibility driver library under the
local `.runtime/` directory.

Install or refresh it from the repository root:

```bash
python scripts/partial_overlap_interference_benchmarks/sm_free_collectives/pairwise_overlap/install_nccl_228_overlay.py
```

Configure the current shell and verify the CUDA Driver API, NCCL version,
NCCL4Py binding, and `CTAPolicy.Zero` support:

```bash
source scripts/partial_overlap_interference_benchmarks/sm_free_collectives/pairwise_overlap/setup_nccl_228_zero_cta_env.sh
```

The environment is shared by the normal-NCCL and Zero-CTA matrix runs so the
grouped-GEMM/A2A, grouped-GEMM/AG, and A2A/AG comparisons use the same pinned
NCCL runtime.
