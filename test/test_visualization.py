from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.visualization import print_training_dag_order


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
