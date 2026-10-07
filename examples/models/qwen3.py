# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NVIDIA modifications:
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import warnings
from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.models.qwen3 import Qwen3Model, Qwen3ModelArgs
from torchtitan.models.qwen3.model.model import TransformerBlock
from torchtitan.models.moe import MoE, MoEArgs
from torchtitan.models.moe.moe import GroupedExperts
from torchtitan.protocols.model import AttentionMasksType
from src.piper import (
    annotate,
    mark_combine_payload,
    mark_dispatch_counts,
    mark_dispatch_payload,
)

PP_TAG = "PP"
EP_TAG = "EP"
ATTN_TAG = "ATTN"
FFN_TAG = "FFN"


@torch.library.custom_op(
    "piper_qwen::source_major_route_ids",
    mutates_args=(),
)
def _source_major_route_ids(
    routed_payload: torch.Tensor,
    route_counts: torch.Tensor,
) -> torch.Tensor:
    """Return one local-route id per runtime source-major payload row.

    The count tensor changes from ``[1, global_routes]`` before dispatch to
    ``[source_ranks, local_routes]`` after dispatch.  Keeping this operation
    opaque to Dynamo ensures its values use the latter runtime shape instead
    of capture-time Python integers.  This is model-side regrouping; Piper's
    transport still treats every payload row as opaque.
    """
    block_route_ids = torch.arange(
        route_counts.shape[1],
        dtype=torch.int64,
        device=route_counts.device,
    ).repeat(route_counts.shape[0])
    return torch.repeat_interleave(
        block_route_ids,
        route_counts.reshape(-1),
        output_size=routed_payload.shape[0],
    )


@_source_major_route_ids.register_fake
def _source_major_route_ids_fake(
    routed_payload: torch.Tensor,
    route_counts: torch.Tensor,
) -> torch.Tensor:
    del route_counts
    return routed_payload.new_empty(
        (routed_payload.shape[0],),
        dtype=torch.int64,
    )


@dataclass(frozen=True)
class Qwen3ConfigSpec:
    """Configuration fields for a Qwen model preset."""

    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    max_seq_len: int
    rope_theta: float
    vocab_size: int = 151936
    norm_eps: float = 1e-6
    qk_norm: bool = True
    depth_init: bool = True
    enable_weight_tying: bool = False
    moe_inter_dim: int | None = None
    num_experts: int = 0
    num_shared_experts: int = 0
    top_k: int = 0
    use_grouped_mm: bool = True
    score_func: str = "softmax"
    route_norm: bool = True
    route_scale: float = 1.0
    gate_bias: bool = False
    score_before_experts: bool = False
    load_balance_coeff: float | None = 1e-3
    exact_backbone: bool = True


_QWEN3_30B = Qwen3ConfigSpec(
    dim=2048,
    n_layers=8,
    n_heads=32,
    n_kv_heads=4,
    head_dim=128,
    hidden_dim=6144,
    moe_inter_dim=768,
    num_experts=128,
    top_k=8,
    max_seq_len=40960,
    rope_theta=1_000_000.0,
)
_QWEN35_397B = Qwen3ConfigSpec(
    dim=4096,
    n_layers=8,
    n_heads=32,
    n_kv_heads=2,
    head_dim=256,
    hidden_dim=1024,
    moe_inter_dim=1024,
    num_experts=512,
    top_k=10,
    max_seq_len=262144,
    rope_theta=10_000_000.0,
    vocab_size=248320,
    num_shared_experts=1,
    exact_backbone=False,
)

# Keep one representative preset per materially different workload used by
# Piper: small MoE correctness/benchmark cases, dense Qwen, standard and
# communication-balanced 30B MoE, and the large Qwen3.5 control-plane proxy.
QWEN3_CONFIG_SPECS = {
    "9M": Qwen3ConfigSpec(
        vocab_size=2048,
        dim=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=4,
        head_dim=32,
        hidden_dim=512,
        max_seq_len=393216,
        rope_theta=1_000_000.0,
        moe_inter_dim=128,
        num_experts=8,
        top_k=2,
        route_norm=False,
        load_balance_coeff=None,
    ),
    "1B": Qwen3ConfigSpec(
        dim=1024,
        n_layers=4,
        n_heads=16,
        n_kv_heads=8,
        head_dim=64,
        hidden_dim=3584,
        max_seq_len=32768,
        rope_theta=1_000_000.0,
        moe_inter_dim=3584,
        num_experts=4,
        top_k=2,
        route_norm=False,
        load_balance_coeff=None,
    ),
    "9B": Qwen3ConfigSpec(
        dim=2048,
        n_layers=4,
        n_heads=32,
        n_kv_heads=8,
        head_dim=64,
        hidden_dim=7168,
        max_seq_len=8192,
        rope_theta=1_000_000.0,
        moe_inter_dim=7168,
        num_experts=8,
        top_k=2,
        route_norm=False,
        load_balance_coeff=None,
    ),
    "72B": Qwen3ConfigSpec(
        vocab_size=152064,
        dim=8192,
        n_layers=4,  # Full model has 80 layers; keep the local reduced preset.
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=29568,
        norm_eps=1e-5,
        max_seq_len=131072,
        rope_theta=1_000_000.0,
    ),
    "Qwen3-30B-A3B": _QWEN3_30B,
    # Empirically calibrated from the normal PyTorch trace: the roofline width
    # of 864 measured about 0.645:1 for the full expert region, so use the
    # aligned width round(864 / 0.645) = 1344.
    "Qwen3-30B-A3B-empirical1to1-L4": replace(
        _QWEN3_30B, moe_inter_dim=1344, n_layers=4
    ),
    "Qwen3.5-397B-A17B": _QWEN35_397B,
}

QWEN_MODEL_NAMES = tuple(QWEN3_CONFIG_SPECS)


def qwen_expert_shard_degree(schedule_directives: list[dict]) -> int:
    """Return the uniform routed-expert shard degree used for Qwen capture.

    The current Qwen wrapper captures one expert implementation shared by all
    routed layers, so all ``shard`` directives must use the same degree.  The
    generic Piper schedule validator remains responsible for matching those
    directives to routed annotations and checking route ownership.
    """
    degrees = {
        len(directive.get("devices", directive.get("device", ())))
        for directive in schedule_directives
        if isinstance(directive, dict) and directive.get("op") == "shard"
    }
    if not degrees:
        return 1
    if 0 in degrees or len(degrees) != 1:
        raise ValueError(
            "Qwen routed expert capture requires one uniform, nonzero shard "
            f"degree across all shard directives, got {sorted(degrees)}"
        )
    return next(iter(degrees))


def create_qwen3_config(name: str) -> Qwen3ModelArgs:
    """Create a Qwen3 model config from the unified preset registry."""
    try:
        spec = QWEN3_CONFIG_SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown model config: {name}") from None

    if not spec.exact_backbone:
        warnings.warn(
            f"{name} uses hybrid linear/full attention; Piper's Qwen3 model "
            "uses full attention while preserving the checkpoint's model and "
            "MoE tensor shapes.",
            stacklevel=2,
        )

    model_kwargs = dict(
        vocab_size=spec.vocab_size,
        dim=spec.dim,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        n_kv_heads=spec.n_kv_heads,
        head_dim=spec.head_dim,
        hidden_dim=spec.hidden_dim,
        norm_eps=spec.norm_eps,
        qk_norm=spec.qk_norm,
        max_seq_len=spec.max_seq_len,
        rope_theta=spec.rope_theta,
        depth_init=spec.depth_init,
        enable_weight_tying=spec.enable_weight_tying,
        moe_enabled=spec.moe_inter_dim is not None,
    )
    if spec.moe_inter_dim is not None:
        model_kwargs.update(
            moe_inter_dim=spec.moe_inter_dim,
            moe_args=MoEArgs(
                num_experts=spec.num_experts,
                num_shared_experts=spec.num_shared_experts,
                top_k=spec.top_k,
                use_grouped_mm=spec.use_grouped_mm,
                num_expert_groups=None,
                num_limited_groups=None,
                score_func=spec.score_func,
                route_norm=spec.route_norm,
                route_scale=spec.route_scale,
                gate_bias=spec.gate_bias,
                score_before_experts=spec.score_before_experts,
                load_balance_coeff=spec.load_balance_coeff,
                _debug_force_load_balance=False,
            ),
        )
    return Qwen3ModelArgs(**model_kwargs)


class AnnotatedMoE(MoE):
    def __init__(
        self,
        moe_args: MoEArgs,
        dim: int,
        hidden_dim: int,
        *,
        num_local_experts: int | None = None,
    ) -> None:
        super().__init__(moe_args, dim=dim, hidden_dim=hidden_dim)
        self.num_global_experts = int(moe_args.num_experts)
        self.num_local_experts = int(
            self.num_global_experts
            if num_local_experts is None
            else num_local_experts
        )
        if (
            self.num_local_experts <= 0
            or self.num_global_experts % self.num_local_experts
        ):
            raise ValueError(
                "num_local_experts must be a positive divisor of "
                f"num_global_experts={self.num_global_experts}, got "
                f"{self.num_local_experts}"
            )
        if self.num_local_experts != self.num_global_experts:
            # Routing remains global, but the routed region captures only the
            # complete experts materialized by one shard rank.
            self.experts = GroupedExperts(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=self.num_local_experts,
                use_grouped_mm=moe_args.use_grouped_mm,
            )

    @staticmethod
    def _to_expert_local_order(
        routed_input: torch.Tensor,
        route_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Regroup an opaque source-major payload for local expert compute.

        Piper presents rows as ``[source][local route][row]`` and counts as
        ``[source, local route]``.  Expert implementations consume one
        contiguous block per local route, so the model (not Piper) performs a
        stable regrouping and retains the inverse permutation for combine.
        """
        row_route_ids = _source_major_route_ids(
            routed_input.detach(),
            route_counts,
        )
        expert_order = torch.argsort(row_route_ids, stable=True)
        transport_order = torch.argsort(expert_order)
        expert_input = routed_input.index_select(0, expert_order)
        local_counts = route_counts.sum(dim=0)
        return expert_input, local_counts, transport_order

    def _run_experts(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        with annotate(EP_TAG):
            # At source-side capture this tensor still describes all global
            # routes.  Routed transport replaces it with the compact local
            # descriptor before this segment runs, so this slice is a no-op at
            # runtime and gives the captured expert graph its local width.
            num_tokens_per_expert = num_tokens_per_expert[
                :, : self.num_local_experts
            ]
            expert_input, local_counts, transport_order = (
                self._to_expert_local_order(
                    routed_input,
                    num_tokens_per_expert,
                )
            )
            expert_output = self.experts(expert_input, local_counts)
            gathered_output = expert_output.index_select(0, transport_order)
            return mark_combine_payload(gathered_output)

    def _route_experts(self, x: torch.Tensor):
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

        # Match TorchTitan's default MoE bookkeeping. This count is distinct
        # from the reordered count vector consumed by GroupedExperts below.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted // self.router.top_k]
        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        routed_input = mark_dispatch_payload(routed_input)
        # Keep the source-block dimension explicit.  Piper replaces this
        # source-side ``[1, global routes]`` descriptor with destination-local
        # ``[num sources, local routes]`` counts after routed dispatch.
        num_tokens_per_expert = mark_dispatch_counts(
            num_tokens_per_expert.unsqueeze(0)
        )

        return (
            routed_input,
            num_tokens_per_expert,
            x,
            top_scores,
            token_indices_experts_sorted,
            bs,
            slen,
            dim,
        )

    def _combine_experts(
        self,
        gathered_output: torch.Tensor,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        bs,
        slen,
        dim,
    ) -> torch.Tensor:

        gathered_output = gathered_output.reshape(-1, dim)
        out = self.shared_experts(x) if self.shared_experts is not None else None

        routed_output_unsorted = torch.zeros(
            (bs * slen * self.router.top_k, dim),
            dtype=gathered_output.dtype,
            device=gathered_output.device,
        )
        routed_output_unsorted[token_indices_experts_sorted] = gathered_output
        routed_output_unsorted = routed_output_unsorted.reshape(
            -1, self.router.top_k, dim
        )

        if not self.score_before_experts:
            out_experts = (
                torch.bmm(
                    top_scores.reshape(-1, 1, self.router.top_k),
                    routed_output_unsorted.float(),
                )
                .to(x.dtype)
                .squeeze(1)
            )
        else:
            out_experts = routed_output_unsorted.sum(dim=1)

        if out is None:
            return out_experts.reshape(bs, slen, dim)
        return (out + out_experts).reshape(bs, slen, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (
            routed_input,
            num_tokens_per_expert,
            x,
            top_scores,
            token_indices_experts_sorted,
            bs,
            slen,
            dim,
        ) = self._route_experts(x)
        gathered_output = self._run_experts(routed_input, num_tokens_per_expert)
        return self._combine_experts(
            gathered_output,
            x,
            top_scores,
            token_indices_experts_sorted,
            bs,
            slen,
            dim,
        )


class AFDAnnotatedMoE(AnnotatedMoE):
    """MoE whose caller assigns only routed expert compute to an F region."""

    def _run_experts(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens_per_expert = num_tokens_per_expert[
            :, : self.num_local_experts
        ]
        expert_input, local_counts, transport_order = (
            self._to_expert_local_order(
                routed_input,
                num_tokens_per_expert,
            )
        )
        expert_output = self.experts(expert_input, local_counts)
        gathered_output = expert_output.index_select(0, transport_order)
        return mark_combine_payload(gathered_output)


class AnnotatedQwen3TransformerBlock(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        model_args: Qwen3ModelArgs,
        *,
        num_local_experts: int | None = None,
    ):
        super().__init__(layer_id, model_args)

        self.layer_id = layer_id
        if self.moe_enabled:
            self.moe = AnnotatedMoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
                num_local_experts=num_local_experts,
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class AFDAnnotatedQwen3TransformerBlock(AnnotatedQwen3TransformerBlock):
    """Qwen block split into alternating dense A and routed-expert F regions."""

    def __init__(
        self,
        layer_id: int,
        model_args: Qwen3ModelArgs,
        *,
        num_local_experts: int | None = None,
    ):
        super().__init__(
            layer_id,
            model_args,
            num_local_experts=num_local_experts,
        )
        if self.moe_enabled:
            self.moe = AFDAnnotatedMoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
                num_local_experts=num_local_experts,
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        # A_i contains every dense operation from the previous expert result
        # (assigned below by layer i - 1) through this layer's router.  This
        # makes the schedule-facing regions alternate A_i -> F_i -> A_{i+1}.
        with annotate(ATTN_TAG, index=self.layer_id):
            x = x + self.attention(
                self.attention_norm(x), freqs_cis, attention_masks, positions
            )
            if self.moe_enabled:
                (
                    routed_input,
                    num_tokens_per_expert,
                    flattened_x,
                    top_scores,
                    token_indices_experts_sorted,
                    bs,
                    slen,
                    dim,
                ) = self.moe._route_experts(self.ffn_norm(x))
            else:
                x = x + self.feed_forward(self.ffn_norm(x))

        if not self.moe_enabled:
            return x

        # F_i is only the routed expert calculation.  Dispatch/count and
        # combine-payload aliases remain at the two tensor boundaries.
        with annotate(FFN_TAG, index=self.layer_id):
            gathered_output = self.moe._run_experts(
                routed_input,
                num_tokens_per_expert,
            )

        def combine_with_residual():
            return x + self.moe._combine_experts(
                gathered_output,
                flattened_x,
                top_scores,
                token_indices_experts_sorted,
                bs,
                slen,
                dim,
            )

        # The post-expert dense tail starts A_{i+1}; the next block reuses that
        # logical index for its attention/router prefix.  The final layer
        # therefore has a terminal A_L tail before the output PP region.
        with annotate(ATTN_TAG, index=self.layer_id + 1):
            x = combine_with_residual()
        return x


class PiperQwen3Model(Qwen3Model):
    """
    Qwen3Model with pipeline stage annotations.

    Wraps torchtitan's Qwen3Model and adds stage boundary annotations
    in the forward pass for pipeline parallelism, and EP dispatch/combine
    annotations in the MoE layers.
    """

    transformer_block_cls = AnnotatedQwen3TransformerBlock

    def __init__(
        self,
        config: Qwen3ModelArgs,
        num_stages: int,
        expert_shard_degree: int = 1,
    ):
        if expert_shard_degree <= 0:
            raise ValueError(
                "expert_shard_degree must be positive, got "
                f"{expert_shard_degree}"
            )
        if (
            config.moe_enabled
            and config.moe_args.num_experts % expert_shard_degree
        ):
            raise ValueError(
                f"num_experts={config.moe_args.num_experts} must be divisible "
                f"by expert_shard_degree={expert_shard_degree}"
            )
        super().__init__(config)
        if num_stages is None:
            from src.state import piper_metadata

            schedule_info = piper_metadata.schedule_info or {}
            num_stages = int(schedule_info.get("num_stages", schedule_info.get("pp_degree", 2)))
        self.num_stages = num_stages
        self.expert_shard_degree = int(expert_shard_degree)
        num_local_experts = (
            config.moe_args.num_experts // self.expert_shard_degree
            if config.moe_enabled
            else None
        )

        # Replace TransformerBlock layers with the model's annotated block type.
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = self.transformer_block_cls(
                layer_id,
                config,
                num_local_experts=num_local_experts,
            )

    def route_parameter_axes(self) -> dict[str, int]:
        """Declare complete-expert parameters for deterministic eager parity."""
        specs: dict[str, int] = {}
        for layer_id, layer in self.layers.items():
            if not getattr(layer, "moe_enabled", False):
                continue
            for weight_name in ("w1", "w2", "w3"):
                specs[
                    f"layers.{layer_id}.moe.experts.{weight_name}"
                ] = 0
        return specs

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: Optional[AttentionMasksType] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_layers = len(self.layers)

        for stage_id in range(self.num_stages):
            layer_start = stage_id * num_layers // self.num_stages
            layer_end = (stage_id + 1) * num_layers // self.num_stages
            with annotate(PP_TAG):
                if stage_id == 0:
                    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

                for i in range(layer_start, layer_end):
                    layer = self.layers[str(i)]
                    h = layer(h, self.rope_cache, attention_masks, positions)

                if stage_id == self.num_stages - 1:
                    h = self.norm(h) if self.norm is not None else h
                    output = self.output(h) if self.output is not None else h

        return output


class AFDQwen3Model(PiperQwen3Model):
    """Qwen3 model with PP stages and per-layer ATTN/FFN annotations."""

    transformer_block_cls = AFDAnnotatedQwen3TransformerBlock
