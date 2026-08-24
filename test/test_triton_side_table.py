import torch.fx as fx

from src.dag import _collect_triton_side_table, restore_triton_side_table
from src.fx import _deserialize_graphmodule, _serialize_graphmodule


def _fake_triton_kernel_for_transport():
    return None


def _wrapper_graph(kernel_idx: int, constant_args_idx: int) -> fx.GraphModule:
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_mutation,
    )

    graph = fx.Graph()
    value = graph.placeholder("value")
    wrapped = graph.call_function(
        triton_kernel_wrapper_mutation,
        kwargs={
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": [(1, 1, 1)],
            "kwargs": {"value": value},
        },
    )
    graph.output(wrapped)
    return fx.GraphModule({}, graph)


def test_triton_side_table_round_trip_remaps_process_local_indices() -> None:
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    previous_kernels = dict(kernel_side_table.id_to_kernel)
    previous_kernel_ids = dict(kernel_side_table.kernel_to_id)
    previous_constants = dict(kernel_side_table.constant_args)
    try:
        kernel_side_table.reset_table()
        old_kernel_idx = kernel_side_table.add_kernel(
            _fake_triton_kernel_for_transport
        )
        old_constant_idx = kernel_side_table.add_constant_args({"BLOCK": 16})
        gm = _wrapper_graph(old_kernel_idx, old_constant_idx)
        transported = _collect_triton_side_table(gm)
        restored_gm = _deserialize_graphmodule(_serialize_graphmodule(gm))

        # Populate unrelated entries so a graph cannot accidentally work by
        # retaining driver-local integer IDs.
        kernel_side_table.reset_table()
        kernel_side_table.add_kernel(str)
        kernel_side_table.add_constant_args({"unrelated": True})
        restore_triton_side_table(
            [restored_gm],
            kernels=transported["kernels"],
            constant_args=transported["constant_args"],
        )

        wrapper = next(
            node
            for node in restored_gm.graph.nodes
            if node.op == "call_function"
        )
        new_kernel_idx = wrapper.kwargs["kernel_idx"]
        new_constant_idx = wrapper.kwargs["constant_args_idx"]
        restored_kernel = kernel_side_table.get_kernel(new_kernel_idx)
        assert restored_kernel.__name__ == _fake_triton_kernel_for_transport.__name__
        assert kernel_side_table.get_constant_args(new_constant_idx) == {
            "BLOCK": 16
        }
        assert new_kernel_idx != old_kernel_idx
        assert new_constant_idx != old_constant_idx
    finally:
        kernel_side_table.id_to_kernel = previous_kernels
        kernel_side_table.kernel_to_id = previous_kernel_ids
        kernel_side_table.constant_args = previous_constants
