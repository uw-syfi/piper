# Qwen schedule and sharding benchmarks

These scripts run and visualize the Qwen schedule/sharding benchmark matrix:

- `run_qwen_sharding_matrix_20260813.py`: full schedule, sharding, and bucketing matrix.
- `run_qwen_schedule_pair_matrix_20260814.py`: selected DualPipeV/interleaved comparison matrix.
- `run_qwen_selected_pytorch_profiles_20260813.py`: profiler captures for selected matrix cases.
- `plot_qwen_sharding_matrix_20260813.py`: throughput and per-PP-rank memory charts.

Run them from the repository root. Benchmark output is written beneath `out/`.
