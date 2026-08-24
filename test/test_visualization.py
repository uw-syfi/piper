from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.ordering import assign_dispatch_levels
from src.visualization import (
    _build_device_table,
    _compute_columns,
    _format_region,
    _infer_region_order,
    _placed_device_groups,
    _rows_from_schedule_directives,
    print_training_dag_order,
    render_training_dag,
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


def test_print_training_dag_order_uses_persisted_dispatch_levels(tmp_path) -> None:
    dag = TrainingDAG()
    for node in (
        TrainingDAGNode(
            uid="compute",
            node_kind="COMPUTE",
            compute_subkind="FWD",
            tag={"PASS": "F"},
            device=[0],
            stream="default_stream",
        ),
        TrainingDAGNode(
            uid="recv",
            node_kind="RECV_COMM",
            compute_subkind=None,
            tag={},
            device=[0],
            stream="pp_stream",
        ),
            TrainingDAGNode(
                uid="a2a",
            node_kind="A2A_COMM",
            compute_subkind=None,
            tag={},
            device=[0],
                stream="ep_stream",
            ),
            TrainingDAGNode(
                uid="consumer",
                node_kind="COMPUTE",
                compute_subkind="FWD",
                tag={"PASS": "F"},
                device=[0],
                stream="default_stream",
            ),
        ):
            dag.add_node(node)

    dag.nodes["recv"].node_meta["destination_compute_uid"] = "consumer"
    dag.add_edge(TrainingDAGEdge("compute", "a2a", "data"))
    dag.add_edge(TrainingDAGEdge("recv", "consumer", "data"))
    dag.add_edge(TrainingDAGEdge("a2a", "consumer", "data"))
    assign_dispatch_levels(dag)

    print_training_dag_order(dag, out_dir=str(tmp_path))
    lines = (tmp_path / "training_dag_order_rank0").read_text().splitlines()
    printed_uids = [
        next(
            uid
            for uid in ("compute", "a2a", "recv", "consumer")
            if f"uid={uid}" in line
        )
        for line in lines[1:5]
    ]
    assert printed_uids == ["compute", "a2a", "recv", "consumer"]
    assert "topo=0" in lines[3]
    assert "dispatch=1" in lines[3]


def test_render_training_dag_aligns_nodes_by_dispatch_level(
    tmp_path, monkeypatch
) -> None:
    import graphviz

    dag = TrainingDAG()
    for uid in ("early", "late"):
        dag.add_node(
            TrainingDAGNode(
                uid=uid,
                node_kind="COMPUTE",
                compute_subkind="FWD",
                tag={"PASS": "F"},
                device=[0],
                stream="default_stream",
            )
        )
    dag.nodes["early"].node_meta["dispatch_level"] = 0
    dag.nodes["late"].node_meta["dispatch_level"] = 7

    rendered: dict[str, str] = {}

    def _capture_render(dot, output_path, *, format, cleanup):
        rendered["source"] = dot.source
        return f"{output_path}.{format}"

    monkeypatch.setattr(graphviz.Digraph, "render", _capture_render)
    render_training_dag(dag, str(tmp_path / "dag"))

    source = rendered["source"]
    assert "subgraph dispatch_0" in source
    assert "subgraph dispatch_7" in source
    assert "subgraph topo_0" not in source
    assert "topo=" not in source
    assert "dispatch=" not in source


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
