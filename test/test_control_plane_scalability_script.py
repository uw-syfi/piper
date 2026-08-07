from scripts.benchmark_control_plane_scalability import parse_args, workload_specs


def test_initial_dag_scaling_matrix_has_thirteen_rank_zero_cases() -> None:
    args = parse_args(
        [
            "--mode",
            "dag-scaling",
            "--parallel-configs",
            "1x8x8,1x16x16,4x8x8",
            "--stream-comparison-config",
            "4x8x8",
            "--stream-comparison-bucket",
            "16",
        ]
    )

    specs = workload_specs(args)

    assert len(specs) == 13
    assert all(spec.world_size < 64 for spec in specs)
    shared = [
        spec for spec in specs if spec.collective_stream_policy == "shared"
    ]
    assert len(shared) == 1
    assert (
        shared[0].pp_degree,
        shared[0].dp_degree,
        shared[0].ep_degree,
        shared[0].bucket_size_mb,
    ) == (4, 8, 8, 16)
