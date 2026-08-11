from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.visualization import (
    _build_device_table,
    _compute_columns,
    _format_region,
    _infer_region_order,
    _placed_device_groups,
    _rows_from_schedule_directives,
    print_training_dag_order,
)


def test_print_training_dag_order_supports_named_global_output(tmp_path) -> None:
    dag = TrainingDAG()
    dag.add_node(
        TrainingDAGNode(
            uid="forward",
            node_kind="COMPUTE",
            compute_subkind="FWD",
            tag={"PP": 0, "PASS": "F"},
            device=[0],
            stream="default_stream",
        )
    )
    dag.add_node(
        TrainingDAGNode(
            uid="backward",
            node_kind="COMPUTE",
            compute_subkind="BWD",
            tag={"PP": 0, "PASS": "B"},
            device=[0],
            stream="default_stream",
        )
    )
    dag.add_edge(
        TrainingDAGEdge(
            src_uid="forward",
            dst_uid="backward",
            dep_kind="data",
        )
    )

    print_training_dag_order(
        dag,
        label="global_training_dag_order",
        out_dir=str(tmp_path),
        output_name="global_training_dag_order",
    )

    output = (tmp_path / "global_training_dag_order").read_text(encoding="utf-8")
    assert output.startswith(
        "--- TrainingDAG execution order: global_training_dag_order ---\n"
    )
    assert "data_succs=['backward']" in output
    assert "data_preds=['forward']" in output


def test_schedule_visualization_resolves_non_pp_regions_from_placement() -> None:
    schedule = [
        {
            "op": "place",
            "filter": {"PIPE": 0, "A": "-", "F": "-"},
            "devices": [0],
        },
        {"op": "place", "filter": {"A": "*"}, "devices": [0, 2]},
        {"op": "place", "filter": {"F": "*"}, "devices": [1, 3]},
        {
            "op": "place",
            "filter": {"PIPE": 1, "A": "-", "F": "-"},
            "devices": [1],
        },
        {
            "op": "order",
            "filters": [
                [{"PIPE": 0, "A": "-", "F": "-", "MB": 0, "PASS": "F"}],
                [{"PIPE": 0, "A": "-", "F": "-", "MB": 0, "PASS": "B"}],
            ],
        },
        {
            "op": "order",
            "filters": [
                [{"A": 0, "MB": 0, "PASS": "F"}],
                [{"A": 0, "MB": 0, "PASS": "B"}],
            ],
        },
        {
            "op": "order",
            "filters": [
                [{"F": 0, "MB": 0, "PASS": "F"}],
                [{"F": 0, "MB": 0, "PASS": "B"}],
            ],
        },
        {
            "op": "order",
            "filters": [
                [{"PIPE": 1, "A": "-", "F": "-", "MB": 0, "PASS": "F"}],
                [{"PIPE": 1, "A": "-", "F": "-", "MB": 0, "PASS": "B"}],
            ],
        },
    ]

    rows = _rows_from_schedule_directives(schedule)

    assert [row.devices for row in rows] == [(0,), (0, 2), (1, 3), (1,)]
    assert [dict(region) for region in _infer_region_order(rows)] == [
        {"A": "-", "F": "-", "PIPE": 0},
        {"A": 0},
        {"F": 0},
        {"A": "-", "F": "-", "PIPE": 1},
    ]
    columns = _compute_columns(rows)
    assert [columns[(row_idx, 0)] for row_idx in range(4)] == [0, 1, 2, 3]

    device_groups = _placed_device_groups(schedule)
    table = _build_device_table(rows, columns, device_groups)
    assert device_groups == [(0, 2), (1, 3)]
    assert len(table) == 2
    assert _format_region(rows[0].slots[0][0].region) == "PIPE0"


def test_schedule_visualization_splits_concurrent_chunks_on_one_gpu() -> None:
    schedule = [
        {
            "op": "place",
            "filter": {"PIPE": 0, "A": "-"},
            "devices": [0],
        },
        {"op": "place", "filter": {"A": "*"}, "devices": [0, 2]},
        {
            "op": "order",
            "filters": [
                [{"PIPE": 0, "A": "-", "MB": 0, "PASS": "F"}],
                [{"PIPE": 0, "A": "-", "MB": 1, "PASS": "F"}],
                [{"PIPE": 0, "A": "-", "MB": 0, "PASS": "B"}],
                [{"PIPE": 0, "A": "-", "MB": 1, "PASS": "B"}],
            ],
        },
        {
            "op": "order",
            "filters": [
                [{"A": 0, "MB": 0, "PASS": "F"}],
                [{"A": 0, "MB": 1, "PASS": "F"}],
                [{"A": 0, "MB": 0, "PASS": "B"}],
                [{"A": 0, "MB": 1, "PASS": "B"}],
            ],
        },
    ]

    rows = _rows_from_schedule_directives(schedule)
    columns = _compute_columns(rows)
    device_groups = _placed_device_groups(schedule)
    table = _build_device_table(rows, columns, device_groups)

    assert device_groups == [(0, 2)]
    assert len(table[0][1]) == 2
    assert {(dict(op.region).get("A"), op.mb) for op in table[0][1]} == {
        ("-", 1),
        (0, 0),
    }
    backward = rows[0].slots[2][0]
    assert sum(backward in slot for slot in table[0]) == 1


def test_schedule_visualization_orders_colocated_input_and_output_endpoints() -> None:
    schedule = [
        {
            "op": "place",
            "filter": {"PIPE": 0, "A": "-", "F": "-"},
            "devices": [0, 2],
        },
        {"op": "place", "filter": {"A": "*"}, "devices": [0, 2]},
        {"op": "place", "filter": {"F": "*"}, "devices": [1, 3]},
        {
            "op": "place",
            "filter": {"PIPE": 1, "A": "-", "F": "-"},
            "devices": [0, 2],
        },
        {
            "op": "order",
            "filters": [
                [{"PIPE": 0, "A": "-", "F": "-", "MB": 0, "PASS": "F"}],
                [{"A": 0, "MB": 0, "PASS": "F"}],
                [{"A": 1, "MB": 0, "PASS": "F"}],
                [{"PIPE": 1, "A": "-", "F": "-", "MB": 0, "PASS": "F"}],
                [{"PIPE": 1, "A": "-", "F": "-", "MB": 0, "PASS": "B"}],
                [{"A": 1, "MB": 0, "PASS": "B"}],
                [{"A": 0, "MB": 0, "PASS": "B"}],
                [{"PIPE": 0, "A": "-", "F": "-", "MB": 0, "PASS": "B"}],
            ],
        },
        {
            "op": "order",
            "filters": [
                [{"F": 0, "MB": 0, "PASS": "F"}],
                [{"F": 0, "MB": 0, "PASS": "B"}],
            ],
        },
    ]

    rows = _rows_from_schedule_directives(schedule)

    assert [dict(region) for region in _infer_region_order(rows)] == [
        {"A": "-", "F": "-", "PIPE": 0},
        {"A": 0},
        {"F": 0},
        {"A": 1},
        {"A": "-", "F": "-", "PIPE": 1},
    ]
    columns = _compute_columns(rows)
    assert len(columns) == 10
    assert columns[(0, 0)] == 0
    assert columns[(0, 3)] > columns[(0, 2)]
    assert columns[(0, 4)] > columns[(0, 3)]
