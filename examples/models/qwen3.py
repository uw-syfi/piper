import warnings
from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.models.qwen3 import Qwen3Model, Qwen3ModelArgs
from torchtitan.models.qwen3.model.model import TransformerBlock
from torchtitan.models.moe import MoE, MoEArgs
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
_QWEN3_30B_BASE = replace(_QWEN3_30B, max_seq_len=32768)
_QWEN3_30B_LONG = replace(
    _QWEN3_30B,
    max_seq_len=262144,
    rope_theta=10_000_000.0,
)
_QWEN3_235B = Qwen3ConfigSpec(
    dim=4096,
    n_layers=94,
    n_heads=64,
    n_kv_heads=4,
    head_dim=128,
    hidden_dim=12288,
    moe_inter_dim=1536,
    num_experts=128,
    top_k=8,
    max_seq_len=40960,
    rope_theta=1_000_000.0,
)
_QWEN3_235B_LONG = replace(
    _QWEN3_235B,
    max_seq_len=262144,
    rope_theta=5_000_000.0,
)
_QWEN3_CODER_480B = Qwen3ConfigSpec(
    dim=6144,
    n_layers=62,
    n_heads=96,
    n_kv_heads=8,
    head_dim=128,
    hidden_dim=8192,
    moe_inter_dim=2560,
    num_experts=160,
    top_k=8,
    max_seq_len=262144,
    rope_theta=10_000_000.0,
    load_balance_coeff=0.0,
)
_QWEN3_NEXT_80B = Qwen3ConfigSpec(
    dim=2048,
    n_layers=8,
    n_heads=16,
    n_kv_heads=2,
    head_dim=256,
    hidden_dim=5120,
    moe_inter_dim=512,
    num_experts=512,
    top_k=10,
    max_seq_len=262144,
    rope_theta=10_000_000.0,
    num_shared_experts=1,
    exact_backbone=False,
)
_QWEN3_CODER_NEXT = replace(_QWEN3_NEXT_80B, rope_theta=5_000_000.0)
_QWEN35_35B = Qwen3ConfigSpec(
    dim=2048,
    n_layers=8,
    n_heads=16,
    n_kv_heads=2,
    head_dim=256,
    # Qwen3.5 has no dense intermediate_size. This field is unused by its
    # all-MoE layer proxy, so use the routed/shared expert width.
    hidden_dim=512,
    moe_inter_dim=512,
    num_experts=256,
    top_k=8,
    max_seq_len=262144,
    rope_theta=10_000_000.0,
    vocab_size=248320,
    num_shared_experts=1,
    exact_backbone=False,
)
_QWEN35_122B = Qwen3ConfigSpec(
    dim=3072,
    n_layers=8,
    n_heads=32,
    n_kv_heads=2,
    head_dim=256,
    hidden_dim=1024,
    moe_inter_dim=1024,
    num_experts=256,
    top_k=8,
    max_seq_len=262144,
    rope_theta=10_000_000.0,
    vocab_size=248320,
    num_shared_experts=1,
    exact_backbone=False,
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

# Synthetic H100 NVLink workload whose aggregate training expert GEMMs and
# dispatch/combine A2A payloads have approximately equal roofline times.  The
# depth-scaled variants preserve that per-layer ratio exactly.
_QWEN3_COMM_1TO1 = Qwen3ConfigSpec(
    vocab_size=4096,
    dim=256,
    n_layers=4,
    n_heads=8,
    n_kv_heads=4,
    head_dim=32,
    hidden_dim=512,
    max_seq_len=8192,
    rope_theta=1_000_000.0,
    moe_inter_dim=864,
    num_experts=8,
    top_k=8,
    route_norm=False,
    load_balance_coeff=None,
)

# Synthetic development presets and non-quantized checkpoints from Qwen's
# official model collections all use this one registry and construction path.
# Quantized releases reuse these same architecture shapes and are not separate
# presets.
QWEN3_CONFIG_SPECS = {
    "comm-1to1-L4": _QWEN3_COMM_1TO1,
    # Preserve Qwen3-30B-A3B's architecture and change only the routed expert
    # width from 768 to the H100-NVLink 1:1 roofline width of 864.  The local
    # proxy uses eight layers; the other variants bracket H100 capacity.
    "Qwen3-30B-A3B-comm1to1-L8": replace(
        _QWEN3_30B, moe_inter_dim=864
    ),
    "Qwen3-30B-A3B-comm1to1-L20": replace(
        _QWEN3_30B, moe_inter_dim=864, n_layers=20
    ),
    "Qwen3-30B-A3B-comm1to1-L22": replace(
        _QWEN3_30B, moe_inter_dim=864, n_layers=22
    ),
    "Qwen3-30B-A3B-comm1to1-L23": replace(
        _QWEN3_30B, moe_inter_dim=864, n_layers=23
    ),
    "Qwen3-30B-A3B-comm1to1-L24": replace(
        _QWEN3_30B, moe_inter_dim=864, n_layers=24
    ),
    # Empirically calibrated from the normal PyTorch trace: the roofline width
    # of 864 measured about 0.645:1 for the full expert region, so use the
    # aligned width round(864 / 0.645) = 1344 and re-bracket capacity.
    "Qwen3-30B-A3B-empirical1to1-L4": replace(
        _QWEN3_30B, moe_inter_dim=1344, n_layers=4
    ),
    "Qwen3-30B-A3B-empirical1to1-L12": replace(
        _QWEN3_30B, moe_inter_dim=1344, n_layers=12
    ),
    "Qwen3-30B-A3B-empirical1to1-L15": replace(
        _QWEN3_30B, moe_inter_dim=1344, n_layers=15
    ),
    "Qwen3-30B-A3B-empirical1to1-L16": replace(
        _QWEN3_30B, moe_inter_dim=1344, n_layers=16
    ),
    "Qwen3-30B-A3B-empirical1to1-L17": replace(
        _QWEN3_30B, moe_inter_dim=1344, n_layers=17
    ),
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
    "48B": Qwen3ConfigSpec(
        dim=4096,
        n_layers=4,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=14336,
        max_seq_len=131072,
        rope_theta=1_000_000.0,
        moe_inter_dim=14336,
        num_experts=8,
        top_k=2,
        route_norm=False,
        load_balance_coeff=None,
    ),
    "30B-A3B-half": Qwen3ConfigSpec(
        dim=2048,
        n_layers=24,
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
        hidden_dim=6144,
        max_seq_len=262144,
        rope_theta=1_000_000.0,
        moe_inter_dim=768,
        num_experts=64,
        top_k=8,
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
    "Qwen3-30B-A3B-Base": _QWEN3_30B_BASE,
    "Qwen3-30B-A3B-Instruct-2507": _QWEN3_30B_LONG,
    "Qwen3-30B-A3B-Thinking-2507": _QWEN3_30B_LONG,
    "Qwen3-235B-A22B": _QWEN3_235B,
    "Qwen3-235B-A22B-Instruct-2507": _QWEN3_235B_LONG,
    "Qwen3-235B-A22B-Thinking-2507": _QWEN3_235B_LONG,
    "Qwen3-Coder-30B-A3B-Instruct": _QWEN3_30B_LONG,
    "Qwen3-Coder-480B-A35B-Instruct": _QWEN3_CODER_480B,
    "Qwen3-Next-80B-A3B-Instruct": _QWEN3_NEXT_80B,
    "Qwen3-Next-80B-A3B-Thinking": _QWEN3_NEXT_80B,
    "Qwen3-Coder-Next": _QWEN3_CODER_NEXT,
    "Qwen3-Coder-Next-Base": _QWEN3_CODER_NEXT,
    "Qwen3.5-35B-A3B": _QWEN35_35B,
    "Qwen3.5-35B-A3B-Base": _QWEN35_35B,
    "Qwen3.5-122B-A10B": _QWEN35_122B,
    "Qwen3.5-397B-A17B": _QWEN35_397B,
    "Qwen3.6-35B-A3B": _QWEN35_35B,
    "Qwen-AgentWorld-35B-A3B": _QWEN35_35B,
}

QWEN_MODEL_NAMES = tuple(QWEN3_CONFIG_SPECS)


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
    def _run_experts(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        with annotate(EP_TAG):
            gathered_output = self.experts(routed_input, num_tokens_per_expert)
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
        routed_expert_indices = selected_experts_indices.reshape(-1)[
            token_indices_experts_sorted
        ]
        # Carry expert IDs through Piper's tensor all-to-all. Base-256 digits
        # remain exact in bfloat16 for the largest Qwen presets in this file.
        expert_id_low = torch.remainder(routed_expert_indices, 256)
        expert_id_high = torch.div(
            routed_expert_indices, 256, rounding_mode="floor"
        )
        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        routed_input = mark_dispatch_payload(routed_input)
        num_tokens_per_expert = mark_dispatch_counts(num_tokens_per_expert)

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
        gathered_output = self.experts(routed_input, num_tokens_per_expert)
        return mark_combine_payload(gathered_output)


class AnnotatedQwen3TransformerBlock(TransformerBlock):
    def __init__(self, layer_id: int, model_args: Qwen3ModelArgs):
        super().__init__(layer_id, model_args)

        self.layer_id = layer_id
        if self.moe_enabled:
            self.moe = AnnotatedMoE(
                model_args.moe_args, dim=model_args.dim, hidden_dim=model_args.moe_inter_dim
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

    def __init__(self, layer_id: int, model_args: Qwen3ModelArgs):
        super().__init__(layer_id, model_args)
        if self.moe_enabled:
            self.moe = AFDAnnotatedMoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
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

    def __init__(self, config: Qwen3ModelArgs, num_stages: int):
        super().__init__(config)
        if num_stages is None:
            from src.state import piper_metadata

            schedule_info = piper_metadata.schedule_info or {}
            num_stages = int(schedule_info.get("num_stages", schedule_info.get("pp_degree", 2)))
        self.num_stages = num_stages

        # Replace TransformerBlock layers with the model's annotated block type.
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = self.transformer_block_cls(layer_id, config)

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
