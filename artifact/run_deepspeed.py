
#!/usr/bin/env python3
"""
DeepSpeed training script for Qwen3 MoE models.

Parallelism layout (8 GPUs, 2 nodes x 4 GPUs/node):
  - PP = 4  (intra-node pipeline parallel)
  - EP = 2  (inter-node expert parallel, splits DP dim)
  - DP = 2  (data parallel, consumed by EP)

Config:
  - qwen3_1b: global_bs=128, micro_bs=8, seq_len=2048, grad_accum=8
  - qwen3_9b: global_bs=256, micro_bs=8, seq_len=2048, grad_accum=16

Topology design (see setup_custom_topology for details):
  We use ProcessTopology(axes=['data', 'pipe'], dims=[2, 4]) so that the
  'pipe' axis varies fastest → PP stages map to adjacent ranks (intra-node).
  Then we pre-register EP process groups with the correct cross-node rank
  pairs BEFORE deepspeed.initialize(), because DeepSpeed's default EP group
  creation assumes pipe-slowest layout and would produce wrong groups.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe.topology import ProcessTopology
from deepspeed.utils import groups as ds_groups

# ---------------------------------------------------------------------------
# Qwen3 architecture constants
# ---------------------------------------------------------------------------
ROPE_THETA = 1_000_000.0
MODEL_CONFIGS = {
    "qwen3_1b": {
        "vocab_size": 151936,
        "dim": 1024,
        "n_layers": 16,
        "n_heads": 16,
        "n_kv_heads": 8,
        "head_dim": 64,
        "moe_inter_dim": 3584,
        "num_experts": 4,
        "top_k": 2,
        "seq_len": 2048,
        "global_batch_size": 128,
        "micro_batch_size": 8,
    },
    "qwen3_9b": {
        "vocab_size": 151936,
        "dim": 2048,
        "n_layers": 24,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "moe_inter_dim": 7168,
        "num_experts": 8,
        "top_k": 2,
        "seq_len": 2048,
        "global_batch_size": 256,
        "micro_batch_size": 8,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim: int, seq_len: int, theta: float = 1e6):
    """Return (cos, sin) each of shape [seq_len, dim//2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [seq_len, dim//2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x: [B, n_heads, S, head_dim]. cos/sin: [S, head_dim//2]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)  # [1, 1, S, half]
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query head count."""
    if n_rep == 1:
        return x
    B, n_kv, S, D = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, S, D).reshape(B, n_kv * n_rep, S, D)


# ---------------------------------------------------------------------------
# Model layers (each takes and returns a single tensor for PipelineModule)
#
# Convention: the pipeline tensor is [B, S, D] for hidden states.
# The first layer converts input_ids → hidden, the last converts hidden → logits.
# ---------------------------------------------------------------------------

class EmbeddingLayer(nn.Module):
    """Token embedding: input_ids [B, S] → hidden [B, S, D]."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)

    def forward(self, input_ids):
        return self.tok_emb(input_ids)


class TransformerBlock(nn.Module):
    """
    Single Qwen3 MoE transformer block.
    GQA attention + MoE FFN, with pre-norm (RMSNorm) and residual connections.

    Takes hidden [B, S, D], returns hidden [B, S, D].
    """
    def __init__(self, model_config: dict[str, int], layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        self.dim = int(model_config["dim"])
        self.n_heads = int(model_config["n_heads"])
        self.n_kv_heads = int(model_config["n_kv_heads"])
        self.head_dim = int(model_config["head_dim"])

        # --- Attention ---
        self.attn_norm = RMSNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        # qk_norm
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        # RoPE cache (registered as buffer so it moves with .to(device))
        cos, sin = precompute_rope(self.head_dim, int(model_config["seq_len"]), ROPE_THETA)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # --- MoE FFN ---
        self.ffn_norm = RMSNorm(self.dim)
        # Build a single expert template, then wrap with DeepSpeed MoE
        expert = SwiGLUExpert(self.dim, int(model_config["moe_inter_dim"]))
        from deepspeed.moe.layer import MoE
        self.moe = MoE(
            hidden_size=self.dim,
            expert=expert,
            num_experts=int(model_config["num_experts"]),
            ep_size=int(model_config["ep_size"]),
            k=int(model_config["top_k"]),
            capacity_factor=1.0,
            eval_capacity_factor=1.0,
            min_capacity=4,
            noisy_gate_policy=None,
            drop_tokens=False,     # no token dropping for training stability
            use_residual=False,
            use_tutel=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        B, S, D = hidden.shape

        # ---------- Self-Attention ----------
        residual = hidden
        h = self.attn_norm(hidden)

        q = self.q_proj(h).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(h).view(B, S, self.n_kv_heads, self.head_dim)

        # qk_norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # [B, heads, S, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # GQA: repeat KV heads
        n_rep = self.n_heads // self.n_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        # Scaled dot-product attention (causal)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0
        )  # [B, heads, S, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        hidden = residual + self.o_proj(attn_out)

        # ---------- MoE FFN ----------
        residual = hidden
        h = self.ffn_norm(hidden)
        # DeepSpeed MoE expects [B*S, D] or [B, S, D]; it handles both
        moe_out, aux_loss, _ = self.moe(h)
        hidden = residual + moe_out

        return hidden


class SwiGLUExpert(nn.Module):
    """Single SwiGLU expert: gate_proj + up_proj → SiLU → down_proj."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FinalNorm(nn.Module):
    """Final RMSNorm before LM head."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden)


class LMHead(nn.Module):
    """Linear projection to vocab logits: [B, S, D] → [B*S, V]."""
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Flatten for cross-entropy: [B*S, V]
        return self.head(hidden).view(-1, self.vocab_size)


# ---------------------------------------------------------------------------
# Loss function for PipelineModule
# ---------------------------------------------------------------------------

def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss. logits: [B*S, V], labels: [B, S] or [B*S]."""
    labels = labels.view(-1)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Mock dataset
# ---------------------------------------------------------------------------

class MockTokenDataset(Dataset):
    """Random token dataset for benchmarking. No real data needed."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = torch.Generator()
        rng.manual_seed(self.seed + idx)
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=rng)
        return tokens[:-1], tokens[1:]  # (input_ids, labels)


# ---------------------------------------------------------------------------
# Pre-register EP groups for custom topology
# ---------------------------------------------------------------------------

def pre_register_ep_groups(ep_size: int, pp_size: int, dp_size: int):
    """
    Pre-register expert-parallel (EP) and expert-data-parallel process groups
    in DeepSpeed's global groups module BEFORE deepspeed.initialize() is called.

    Why this is needed:
    ------------------
    We use ProcessTopology(axes=['data', 'pipe'], dims=[dp, pp]) so that the
    'pipe' axis is fastest-varying → PP stages are adjacent ranks → intra-node.

    Rank layout with 2 nodes × 4 GPUs:
        rank 0: data=0, pipe=0  (node 0)
        rank 1: data=0, pipe=1  (node 0)
        rank 2: data=0, pipe=2  (node 0)
        rank 3: data=0, pipe=3  (node 0)
        rank 4: data=1, pipe=0  (node 1)
        rank 5: data=1, pipe=1  (node 1)
        rank 6: data=1, pipe=2  (node 1)
        rank 7: data=1, pipe=3  (node 1)

    PP groups (varying pipe, same data): [0,1,2,3] and [4,5,6,7]  → intra-node ✓
    DP groups (varying data, same pipe): [0,4], [1,5], [2,6], [3,7] → cross-node ✓

    EP should align with DP (same PP stage, different data ranks):
        EP groups: [0,4], [1,5], [2,6], [3,7]  → cross-node ✓

    However, DeepSpeed's _create_expert_and_data_parallel() assumes pipe is the
    SLOWEST axis and creates EP groups from consecutive ranks: [0,1], [2,3], ...
    which would be WRONG for our layout (those pair different PP stages on the
    same node).

    Solution: we pre-populate the _EXPERT_PARALLEL_GROUP and
    _EXPERT_DATA_PARALLEL_GROUP dicts. When DeepSpeed's MoE._create_process_groups
    runs, it checks `if group_name not in groups._get_expert_parallel_group_dict()`
    and skips creation if the groups already exist.
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == pp_size * dp_size

    group_name = f"ep_size_{ep_size}"

    # EP groups: ranks sharing the same pipe stage but different data index
    # With axes=['data', 'pipe'], rank = data_idx * pp_size + pipe_idx
    # So for a given pipe stage p, EP peers are: [p, p + pp_size, p + 2*pp_size, ...]
    # With dp_size=2, pp_size=4: EP groups = [0,4], [1,5], [2,6], [3,7]
    for pipe_stage in range(pp_size):
        ep_ranks = list(range(pipe_stage, world_size, pp_size))
        # ep_ranks has dp_size elements; chunk into groups of ep_size
        for start in range(0, len(ep_ranks), ep_size):
            ranks = ep_ranks[start : start + ep_size]
            group = dist.new_group(ranks)
            if rank in ranks:
                ds_groups._EXPERT_PARALLEL_GROUP[group_name] = group
                ds_groups._EXPERT_PARALLEL_GROUP_RANKS[group_name] = ranks
            if rank == 0:
                print(f"  EP group ({group_name}): {ranks}")

    # Expert-data-parallel groups: ranks with the same EP role but different
    # data-parallel shards.  With ep_size == dp_size (our case), each EP group
    # spans the entire DP dimension, so expert-data-parallel is trivially each
    # rank alone (no all-reduce needed for MoE params).
    # More generally: for each pipe stage, chunk DP ranks into EP groups,
    # then expert-data-parallel = ranks at the same position across EP groups.
    for pipe_stage in range(pp_size):
        dp_ranks = list(range(pipe_stage, world_size, pp_size))
        # With ep_size == dp_size, there's only 1 EP group per pipe stage,
        # so expert-data-parallel groups are just individual ranks.
        # With ep_size < dp_size, we'd interleave.
        num_ep_groups = dp_size // ep_size
        for pos_in_ep in range(ep_size):
            edp_ranks = [dp_ranks[g * ep_size + pos_in_ep] for g in range(num_ep_groups)]
            group = dist.new_group(edp_ranks)
            if rank in edp_ranks:
                ds_groups._EXPERT_DATA_PARALLEL_GROUP[group_name] = group
                ds_groups._EXPERT_DATA_PARALLEL_GROUP_RANKS[group_name] = edp_ranks
            if rank == 0:
                print(f"  Expert-DP group ({group_name}): {edp_ranks}")


# ---------------------------------------------------------------------------
# Build PipelineModule
# ---------------------------------------------------------------------------

def build_pipeline_model(model_config: dict[str, int], args):
    """Construct PipelineModule for the selected Qwen3 model."""
    layers = []

    # Embedding
    layers.append(LayerSpec(EmbeddingLayer, int(model_config["vocab_size"]), int(model_config["dim"])))

    for i in range(int(model_config["n_layers"])):
        layers.append(LayerSpec(TransformerBlock, model_config, layer_id=i))

    # Final norm + LM head
    layers.append(LayerSpec(FinalNorm, int(model_config["dim"])))
    layers.append(LayerSpec(LMHead, int(model_config["dim"]), int(model_config["vocab_size"])))

    # -----------------------------------------------------------------------
    # Topology: PP=4 intra-node, DP=2 cross-node (consumed by EP=2)
    # 8 GPUs total: 4 PP stages × 2 DP ranks
    #
    # We use ProcessTopology with axes=['data', 'pipe'] so that 'pipe' is the
    # fastest-varying axis.  This ensures PP stages are adjacent ranks and
    # thus intra-node (torchrun assigns rank = node_rank * nproc + local).
    #
    # Rank layout:
    #   rank 0: data=0, pipe=0  (node 0, GPU 0)
    #   rank 1: data=0, pipe=1  (node 0, GPU 1)
    #   rank 2: data=0, pipe=2  (node 0, GPU 2)
    #   rank 3: data=0, pipe=3  (node 0, GPU 3)
    #   rank 4: data=1, pipe=0  (node 1, GPU 0)
    #   rank 5: data=1, pipe=1  (node 1, GPU 1)
    #   rank 6: data=1, pipe=2  (node 1, GPU 2)
    #   rank 7: data=1, pipe=3  (node 1, GPU 3)
    #
    # PP groups (intra-node):  [0,1,2,3] and [4,5,6,7]
    # DP groups (cross-node):  [0,4], [1,5], [2,6], [3,7]
    # EP groups (cross-node):  [0,4], [1,5], [2,6], [3,7]  (pre-registered)
    # -----------------------------------------------------------------------
    topo = ProcessTopology(axes=['data', 'pipe'], dims=[args.dp, args.pp])

    model = PipelineModule(
        layers=layers,
        topology=topo,
        loss_fn=loss_fn,
        partition_method="parameters",
        activation_checkpoint_interval=0,
    )
    return model


def parameter_counts(model_config: dict[str, int]) -> tuple[int, int]:
    """Return total and active parameter counts for the local Qwen3 MoE model."""
    vocab_size = int(model_config["vocab_size"])
    dim = int(model_config["dim"])
    n_heads = int(model_config["n_heads"])
    n_kv_heads = int(model_config["n_kv_heads"])
    head_dim = int(model_config["head_dim"])
    n_layers = int(model_config["n_layers"])
    hidden_dim = int(model_config["moe_inter_dim"])
    num_experts = int(model_config["num_experts"])
    top_k = int(model_config["top_k"])

    embeddings_and_head = 2 * vocab_size * dim
    attention = (
        dim * (n_heads * head_dim)
        + dim * (n_kv_heads * head_dim)
        + dim * (n_kv_heads * head_dim)
        + (n_heads * head_dim) * dim
    )
    norms = dim + head_dim + head_dim + dim
    router = dim * num_experts
    expert = 3 * dim * hidden_dim
    total_block = attention + norms + router + num_experts * expert
    active_block = attention + norms + router + top_k * expert
    return embeddings_and_head + n_layers * total_block, embeddings_and_head + n_layers * active_block


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DeepSpeed Qwen3 MoE PP4+EP2")
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS), default="qwen3_1b")
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--zero-stage", type=int, default=1, choices=(0, 1, 2, 3))
    parser.add_argument("--schedule", choices=("1f1b",), default="1f1b")
    parser.add_argument("--micro-bs", type=int, default=None)
    parser.add_argument("--global-bs", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Initialize distributed
    deepspeed.init_distributed()
    torch.manual_seed(args.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if args.schedule != "1f1b":
        raise ValueError("DeepSpeed runner currently supports only 1f1b schedule")
    if args.pp > 1 and args.zero_stage > 1:
        raise ValueError("DeepSpeed pipeline parallelism is not compatible with ZeRO-2/3")
    if args.dp < args.ep or args.dp % args.ep != 0:
        raise ValueError("--dp must be divisible by --ep and at least as large as --ep")
    assert world_size == args.pp * args.dp, \
        f"Expected {args.pp * args.dp} GPUs (PP={args.pp} x DP={args.dp}), got {world_size}"
    model_config = dict(MODEL_CONFIGS[args.model])
    model_config["ep_size"] = args.ep
    if args.seq_len is not None:
        model_config["seq_len"] = args.seq_len
    if args.micro_bs is not None:
        model_config["micro_batch_size"] = args.micro_bs
    if args.global_bs is not None:
        model_config["global_batch_size"] = args.global_bs
    grad_accum_steps = int(model_config["global_batch_size"]) // (
        int(model_config["micro_batch_size"]) * args.dp
    )
    if grad_accum_steps < 1:
        raise ValueError("Derived gradient_accum_steps must be >= 1")

    if rank == 0:
        print(f"{'='*60}")
        print(f"{args.model} — DeepSpeed")
        print(f"  World size:   {world_size}")
        print(f"  PP stages:    {args.pp}")
        print(f"  EP size:      {args.ep}")
        print(f"  DP size:      {args.dp}")
        print(f"  ZeRO stage:   {args.zero_stage}")
        print(f"  Seq len:      {model_config['seq_len']}")
        print(f"  Global BS:    {model_config['global_batch_size']}")
        print(f"  Micro BS:     {model_config['micro_batch_size']}")
        print(f"  Grad accum:   {grad_accum_steps}")
        print(f"  Num layers:   {model_config['n_layers']}")
        print(f"  Num experts:  {model_config['num_experts']}")
        print(f"  Top-K:        {model_config['top_k']}")
        total_params, active_params = parameter_counts(model_config)
        print(f"  Total params: {total_params:,}")
        print(f"  Active params:{active_params:,}")
        print(f"{'='*60}")

    # Pre-register EP groups with correct cross-node topology BEFORE
    # deepspeed.initialize() so that the automatic (wrong) creation is skipped.
    if rank == 0:
        print("\nPre-registering EP process groups (cross-node):")
    pre_register_ep_groups(ep_size=args.ep, pp_size=args.pp, dp_size=args.dp)

    # Build model
    model = build_pipeline_model(model_config, args)

    # Dataset: enough samples for all steps
    # PipelineEngine handles batching internally
    num_samples = int(model_config["global_batch_size"]) * (args.steps + 2)
    dataset = MockTokenDataset(
        vocab_size=int(model_config["vocab_size"]),
        seq_len=int(model_config["seq_len"]),
        num_samples=num_samples,
        seed=args.seed,
    )

    ds_config = {
        "train_micro_batch_size_per_gpu": int(model_config["micro_batch_size"]),
        "train_batch_size": int(model_config["global_batch_size"]),
        "gradient_accumulation_steps": grad_accum_steps,
        "steps_per_print": 1,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 3.0e-4,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.1,
            },
        },
        "zero_optimization": {
            "stage": args.zero_stage,
        },
    }

    # Initialize DeepSpeed (auto-detects PipelineModule → PipelineEngine)
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        config=ds_config,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=dataset,
    )
    torch.cuda.reset_peak_memory_stats()

    if rank == 0:
        print(f"\nPipeline stage: {engine.stage_id}, "
              f"num micro-batches: {engine.micro_batches}")
        print(f"Starting training for {args.steps} steps...\n")

    # Training loop
    start_time = time.time()
    for step in range(1, args.steps + 1):
        step_start = time.time()
        loss = engine.train_batch()
        step_time = time.time() - step_start

        if rank == 0:
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            print(f"[Step {step:3d}/{args.steps}]  loss={loss_val:.4f}  "
                  f"step_time={step_time:.2f}s")

    total_time = time.time() - start_time
    torch.distributed.barrier()
    peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    local_peak = {
        "rank": rank,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
    }
    gathered_peaks = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered_peaks, local_peak)
    if rank == 0:
        for peak in sorted(gathered_peaks, key=lambda item: int(item["rank"])):
            print(
                f"[rank{peak['rank']}] "
                f"peak_memory_allocated_gb={float(peak['peak_allocated_gb']):.3f} "
                f"peak_memory_reserved_gb={float(peak['peak_reserved_gb']):.3f}",
                flush=True,
            )
    if rank == 0:
        print(f"\nTraining complete. Total time: {total_time:.2f}s "
              f"({total_time / args.steps:.2f}s/step)")


if __name__ == "__main__":
    main()
