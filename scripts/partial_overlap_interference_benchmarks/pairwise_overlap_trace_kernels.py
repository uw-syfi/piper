"""Trace-anchored Triton pointwise operations for the overlap suite.

These helpers reproduce the mathematical model operations while making the
launch geometry observed in the 12,288-token Qwen trace explicit.  Inductor's
pointwise heuristic can otherwise select a different warp count for the same
operation when it is compiled outside the dynamic model graph.
"""

from __future__ import annotations

from typing import Any, Callable


def make_swiglu_backward(
    torch: Any,
    *,
    numel: int,
) -> tuple[Callable[[], tuple[Any, Any]], list[Any], dict[str, Any]]:
    import triton
    import triton.language as tl

    @triton.jit
    def triton_poi_fused_mul_silu_silu_backward_0(
        gradient_ptr,
        gate_ptr,
        up_ptr,
        out_gate_ptr,
        out_up_ptr,
        xnumel: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        mask = offsets < xnumel
        gradient = tl.load(gradient_ptr + offsets, mask=mask).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
        sigmoid = tl.sigmoid(gate)
        silu = gate * sigmoid
        gate_gradient = gradient * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        up_gradient = gradient * silu
        tl.store(out_gate_ptr + offsets, gate_gradient, mask=mask)
        tl.store(out_up_ptr + offsets, up_gradient, mask=mask)

    gradient = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
    gate = torch.ones_like(gradient)
    up = torch.ones_like(gradient)
    out_gate = torch.empty_like(gradient)
    out_up = torch.empty_like(gradient)
    block = 1024
    grid = (triton.cdiv(numel, block),)

    def launch() -> tuple[Any, Any]:
        triton_poi_fused_mul_silu_silu_backward_0[grid](
            gradient,
            gate,
            up,
            out_gate,
            out_up,
            numel,
            XBLOCK=block,
            num_warps=8,
        )
        return out_gate, out_up

    return launch, [gradient, gate, up, out_gate, out_up], {
        "operation": "SwiGLU backward",
        "numel": numel,
        "xblock": block,
        "num_warps": 8,
        "expected_grid_x": grid[0],
    }


def make_bf16_add(
    torch: Any,
    *,
    numel: int,
) -> tuple[Callable[[], Any], list[Any], dict[str, Any]]:
    import triton
    import triton.language as tl

    @triton.jit
    def triton_poi_fused_add_1(
        lhs_ptr,
        rhs_ptr,
        output_ptr,
        xnumel: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        mask = offsets < xnumel
        lhs = tl.load(lhs_ptr + offsets, mask=mask).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask).to(tl.float32)
        tl.store(output_ptr + offsets, lhs + rhs, mask=mask)

    lhs = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
    rhs = torch.ones_like(lhs)
    output = torch.empty_like(lhs)
    block = 1024
    grid = (triton.cdiv(numel, block),)

    def launch() -> Any:
        triton_poi_fused_add_1[grid](
            lhs,
            rhs,
            output,
            numel,
            XBLOCK=block,
            num_warps=4,
        )
        return output

    return launch, [lhs, rhs, output], {
        "operation": "BF16 elementwise add",
        "numel": numel,
        "xblock": block,
        "num_warps": 4,
        "expected_grid_x": grid[0],
    }
