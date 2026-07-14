from typing import Optional
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from examples.models.qwen3_bmm_experts_patch import BmmExperts
from torchtitan.models.qwen3 import Qwen3Model, Qwen3ModelArgs
from torchtitan.models.qwen3.model.model import TransformerBlock
from torchtitan.models.moe import MoE, MoEArgs
from torchtitan.protocols.model import AttentionMasksType
from src.piper import annotate

PP_TAG = "PP"
EP_TAG = "EP"


def create_qwen3_config(name: str) -> Qwen3ModelArgs:
    """Create Qwen3 model config based on name."""
    match name:
        case '9M':
            return Qwen3ModelArgs(
                vocab_size=2048,
                dim=256,
                n_layers=4,
                n_heads=8,
                n_kv_heads=4,
                head_dim=32,
                hidden_dim=512,
                norm_eps=1e-6,
                qk_norm=True,
                max_seq_len=2048,
                rope_theta=1000000.0,
                depth_init=True,
                enable_weight_tying=False,
                moe_enabled=True,
                moe_inter_dim=128,
                moe_args=MoEArgs(
                    num_experts=8,
                    top_k=2,
                    use_grouped_mm=True,
                    num_expert_groups=None,
                    num_limited_groups=None,
                    score_func="softmax",
                    route_norm=False,
                    route_scale=1.0,
                    gate_bias=False,
                    score_before_experts=False,
                    num_shared_experts=0,
                    load_balance_coeff=None,
                    _debug_force_load_balance=False,
                ),
            )
        case '1B':
            return Qwen3ModelArgs(
                vocab_size=151936,
                dim=1024,
                n_layers=16,
                n_heads=16,
                n_kv_heads=8,
                head_dim=64,
                hidden_dim=3584,
                norm_eps=1e-6,
                qk_norm=True,
                max_seq_len=2048,
                rope_theta=1000000.0,
                depth_init=True,
                enable_weight_tying=False,
                moe_enabled=True,
                moe_inter_dim=3584,
                moe_args=MoEArgs(
                    num_experts=4,
                    top_k=2,
                    use_grouped_mm=True,
                    num_expert_groups=None,
                    num_limited_groups=None,
                    score_func="softmax",
                    route_norm=False,
                    route_scale=1.0,
                    gate_bias=False,
                    score_before_experts=False,
                    num_shared_experts=0,
                    load_balance_coeff=None,
                    _debug_force_load_balance=False,
                ),
            )
        case '9B':
            return Qwen3ModelArgs(
                vocab_size=151936,
                dim=2048,
                n_layers=24,
                n_heads=32,
                n_kv_heads=8,
                head_dim=64,
                hidden_dim=7168,
                norm_eps=1e-6,
                qk_norm=True,
                max_seq_len=2048,
                rope_theta=1000000.0,
                depth_init=True,
                enable_weight_tying=False,
                moe_enabled=True,
                moe_inter_dim=7168,
                moe_args=MoEArgs(
                    num_experts=8,
                    top_k=2,
                    use_grouped_mm=True,
                    num_expert_groups=None,
                    num_limited_groups=None,
                    score_func="softmax",
                    route_norm=False,
                    route_scale=1.0,
                    gate_bias=False,
                    score_before_experts=False,
                    num_shared_experts=0,
                    load_balance_coeff=None,
                    _debug_force_load_balance=False,
                ),
            )
        case '48B':
            return Qwen3ModelArgs(
                vocab_size=151936,
                dim=4096,
                n_layers=32,
                n_heads=32,
                n_kv_heads=8,
                head_dim=128,
                hidden_dim=14336,
                norm_eps=1e-6,
                qk_norm=True,
                max_seq_len=2048,
                rope_theta=1000000.0,
                depth_init=True,
                enable_weight_tying=False,
                moe_enabled=True,
                moe_inter_dim=14336,
                moe_args=MoEArgs(
                    num_experts=8,
                    top_k=2,
                    use_grouped_mm=True,
                    num_expert_groups=None,
                    num_limited_groups=None,
                    score_func="softmax",
                    route_norm=False,
                    route_scale=1.0,
                    gate_bias=False,
                    score_before_experts=False,
                    num_shared_experts=0,
                    load_balance_coeff=None,
                    _debug_force_load_balance=False,
                ),
            )
        case '30B-A3B':
            return Qwen3ModelArgs(
                vocab_size=151936,
                dim=2048,
                n_layers=48,
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                hidden_dim=6144,
                norm_eps=1e-6,
                qk_norm=True,
                max_seq_len=262144,
                rope_theta=1000000.0,
                enable_weight_tying=False,
                moe_enabled=True,
                moe_inter_dim=768,
                moe_args=MoEArgs(
                    num_experts=128,
                    num_shared_experts=0,
                    top_k=8,
                    score_func="softmax",
                    route_norm=True,
                    route_scale=1.0,
                    score_before_experts=False,
                    load_balance_coeff=None,
                ),
            )
        case '30B-A3B-half':
            return Qwen3ModelArgs(
                vocab_size=151936,
                dim=2048,
                n_layers=24,
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                hidden_dim=6144,
                norm_eps=1e-6,
                qk_norm=True,
                max_seq_len=262144,
                rope_theta=1000000.0,
                enable_weight_tying=False,
                moe_enabled=True,
                moe_inter_dim=768,
                moe_args=MoEArgs(
                    num_experts=64,
                    num_shared_experts=0,
                    top_k=8,
                    score_func="softmax",
                    route_norm=True,
                    route_scale=1.0,
                    score_before_experts=False,
                    load_balance_coeff=None,
                ),
            )
        case '72B':
            return Qwen3ModelArgs(
                vocab_size=152064,
                dim=8192,
                n_layers=4, #80,
                n_heads=64,
                n_kv_heads=8,
                head_dim=128,
                hidden_dim=29568,
                norm_eps=1e-5,
                qk_norm=True,
                max_seq_len=131072,
                rope_theta=1000000.0,
                depth_init=True,
                enable_weight_tying=False,
                moe_enabled=False,
            )
        case _:
            raise ValueError(f"Unknown model config: {name}")

class AnnotatedMoE(MoE):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__(moe_args, dim=dim, hidden_dim=hidden_dim)
        # Replace the grouped-mm (Triton) experts with a batched-matmul variant.
        self.experts = BmmExperts(
            dim=dim, hidden_dim=hidden_dim, num_experts=moe_args.num_experts
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
        ) = self.router(x, self.expert_bias)

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

        # expert component
        with annotate(EP_TAG):
            gathered_output = self.experts(routed_input, num_tokens_per_expert)

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

class PiperQwen3Model(Qwen3Model):
    """
    Qwen3Model with pipeline stage annotations.

    Wraps torchtitan's Qwen3Model and adds stage boundary annotations
    in the forward pass for pipeline parallelism, and EP dispatch/combine
    annotations in the MoE layers.
    """

    def __init__(self, config: Qwen3ModelArgs, num_stages: int):
        super().__init__(config)
        if num_stages is None:
            from src.state import piper_metadata

            schedule_info = piper_metadata.schedule_info or {}
            num_stages = int(schedule_info.get("num_stages", schedule_info.get("pp_degree", 2)))
        self.num_stages = num_stages

        # Replace TransformerBlock layers with AnnotatedQwen3TransformerBlock
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = AnnotatedQwen3TransformerBlock(layer_id, config)

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
