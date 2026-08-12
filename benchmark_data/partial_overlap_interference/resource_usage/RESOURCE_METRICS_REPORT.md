# Figure 1 isolated resource metrics

Peak SM footprint is a static launch/residency estimate for each exact
primary Figure 1 kernel. Other entries are medians across 10 measured
trials; brackets contain the 25th and 75th percentiles. Each measured
trial first takes the median across active GPU endpoints. The run used
exactly 3 warmups.

| Operation | Topology | Estimated peak SM footprint (%) | DRAM BW util (%) | NVLink RX+TX (GiB/s) | PCIe RX+TX (GiB/s) | IB TX+RX per endpoint (GiB/s) |
|---|---|---:|---:|---:|---:|---:|
| Grouped GEMM | none (single GPU) | 100.00 | 32.45 [31.87, 32.94] | 0.00 [0.00, 0.00] | 0.024 [0.024, 0.026] | 0.000 [0.000, 0.000] |
| P2P send | PCIe + InfiniBand GPUDirect RDMA (cross-node) | 1.52 | 0.82 [0.81, 0.83] | 0.00 [0.00, 0.00] | 27.032 [26.850, 27.496] | 25.888 [25.877, 25.894] |
| A2A | NVLink 4/NVSwitch (intra-node, 8 GPU) | 24.24 | 10.16 [10.03, 10.34] | 219.53 [214.74, 225.76] | 0.018 [0.018, 0.018] | 0.000 [0.000, 0.000] |
| AG | NVLink 4/NVSwitch (intra-node, 8 GPU) | 18.18 | 22.91 [21.99, 23.41] | 523.81 [502.61, 535.67] | 0.025 [0.024, 0.025] | 0.000 [0.000, 0.000] |
| RS | NVLink 4/NVSwitch (intra-node, 8 GPU) | 18.18 | 20.56 [20.36, 20.68] | 478.48 [473.47, 481.66] | 0.023 [0.023, 0.023] | 0.000 [0.000, 0.000] |

## Peak-SM-footprint method

- The exact primary kernels, grids, blocks, registers, and shared-memory values come from the original 12,288-token PyTorch traces indexed by `EXACT_KERNEL_EXECUTION_MATRIX.json`.
- Hardware constants are H100: 132 SMs, 2048 threads/SM, 65536 registers/SM, 233472 bytes shared memory/SM, and warp size 32.
- All five kernels are limited to one resident block per SM. Assuming the initial resident wave fills, each resident block must occupy a distinct SM, so `peak footprint = min(grid blocks, 132) / 132`.
- This estimates the maximum spatial SM footprint, not active-cycle utilization, instruction throughput, or the fraction of each SM's execution pipelines consumed.
- Full kernel names and occupancy calculations are in `figure1_kernel_launch_geometry.csv`.

## Measured-metric semantics

- DRAM is NVML GPM ID 10.
- NVLink is the sum of NVML GPM RX and TX rate IDs 60 and 61.
- PCIe is the sum of NVML GPM RX and TX rate IDs 20 and 21.
- IB is derived from `/sys/class/infiniband/*/ports/*/counters/port_{xmit,rcv}_data`; all HCA rates are summed per node, then the two node endpoint rates are median-reduced.
- NVML values are GPU-wide wall-interval averages over repeated exact isolated operations, with no duty-cycle correction.
- The removed NVML SM averages remain preserved in `resource_metrics_trials.csv`, `resource_metrics_summary_nvml_sm_original.csv`, and `RESOURCE_METRICS_REPORT_NVML_SM_ORIGINAL.md`.
- P2P is a two-rank cross-node operation. A2A, AG, and RS are eight-rank collectives contained within one node.
