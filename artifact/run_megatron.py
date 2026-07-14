################ Below is training script for Megatron ################



"""
Train a Qwen3-30B-A3B style MoE model using mock (synthetic) data.
No real dataset required — useful for debugging and benchmarking.

Architecture: Qwen3-30B-A3B MoE
  - 48 layers, hidden=2048, 32 attention heads (GQA: 4 query groups), kv_channels=128
  - FFN: dense ffn_hidden_size=6144 (unused for MoE layers), moe_ffn_hidden_size=768
  - 128 experts, top-8 routing, all layers are MoE (moe_layer_freq=1)
  - RoPE: base=1000000, position_embedding_type=rope
  - QK LayerNorm enabled

Parallelism notes:
  - EP (expert parallelism) is orthogonal to TP/PP/DP/CP.
  - Total GPUs = tp * pp * dp * cp * ep.
  - With alltoall dispatcher, EP tokens are exchanged across ep ranks within each dp group.
  - Sequence parallelism (--sp) requires --tp > 1.

Usage:
    python examples/train_moe_30b_mock.py --nnodes N --nproc-per-node N
        [--tensorboard-dir DIR]
        [--tp TP] [--pp PP] [--dp DP] [--cp CP] [--ep EP] [--sp]
        [--seq-length N] [--micro-bs N] [--global-bs N]
        [--use-tp-pp-dp-mapping]
        [--master-addr ADDR] [--master-port PORT]
        [--disable-background-mode]

Must be run from the Megatron-LM root directory.
"""

import argparse
import os
import subprocess
import sys

MODEL_CONFIGS = {
    "qwen3_1b": {
        "num_layers": 16,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_query_groups": 8,
        "kv_channels": 64,
        "ffn_hidden_size": 3584,
        "num_experts": 4,
        "moe_router_topk": 2,
        "moe_ffn_hidden_size": 3584,
        "seq_length": 512,
        "max_position_embeddings": 2048,
        "vocab_size": 151936,
    },
    "qwen3_9b": {
        "num_layers": 24,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "kv_channels": 64,
        "ffn_hidden_size": 7168,
        "num_experts": 8,
        "moe_router_topk": 2,
        "moe_ffn_hidden_size": 7168,
        "seq_length": 2048,
        "max_position_embeddings": 2048,
        "vocab_size": 151936,
    },
}

MEGATRON_OPTIONAL_FLAGS = {
    "--tensorboard-dir",
    "--profile",
    "--use-pytorch-profiler",
    "--profile-step-start",
    "--profile-step-end",
    "--pytorch-profiler-collect-shapes",
}


def _detect_supported_megatron_flags() -> set[str]:
    """Return optional CLI flags supported by this checkout's pretrain_gpt.py."""
    try:
        proc = subprocess.run(
            [sys.executable, "pretrain_gpt.py", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    help_text = f"{proc.stdout}\n{proc.stderr}"
    return {flag for flag in MEGATRON_OPTIONAL_FLAGS if flag in help_text}

def _setup_distributed(nnodes: int, master_addr: str, background_mode: bool):
    env_node_rank = os.environ.get("NODE_RANK") or os.environ.get("GROUP_RANK") or "0"

    if background_mode:
        resolved_master = master_addr or os.environ.get("MASTER_ADDR") or "127.0.0.1"
        return int(env_node_rank), resolved_master

    if not master_addr:
        raise ValueError("--master-addr is required when --disable-background-mode is set")
    if nnodes == 1:
        return 0, master_addr
    return int(env_node_rank), master_addr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorboard-dir", default="tensorboard/moe_30b_mock",
                        help="Directory for TensorBoard logs and profiler traces.")
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        default="qwen3_1b",
        help="Qwen model preset to run.",
    )
    parser.add_argument("--micro-bs", type=int, default=16,
                        help="Micro batch size.")
    parser.add_argument("--global-bs", type=int, default=64,
                        help="Global batch size.")
    parser.add_argument("--train-iters", type=int, default=8,
                        help="Training iterations.")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor model parallel size.")
    parser.add_argument("--pp", type=int, default=1,
                        help="Pipeline model parallel size.")
    parser.add_argument("--dp", type=int, default=1,
                        help="Data parallel size.")
    parser.add_argument("--cp", type=int, default=1,
                        help="Context parallel size.")
    parser.add_argument("--ep", type=int, default=1,
                        help="Expert model parallel size.")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Sequence length (default: 512).")
    parser.add_argument("--sp", action="store_true", default=False,
                        help="Enable sequence parallelism (requires --tp > 1).")
    parser.add_argument(
        "--schedule",
        choices=("1f1b", "interleaved1f1b"),
        default="1f1b",
        help="Pipeline schedule.",
    )
    parser.add_argument(
        "--zero-level",
        choices=("zero1", "zero2", "zero3"),
        default="zero1",
        help="Megatron data-parallel sharding mode.",
    )
    parser.add_argument("--use-tp-pp-dp-mapping", action="store_true", default=False,
                        help="Use tp-cp-ep-pp-dp rank ordering (PP intra-node) "
                             "instead of default tp-cp-ep-dp-pp (PP cross-node).")
    parser.add_argument("--nnodes", type=int, required=True,
                        help="Number of nodes.")
    parser.add_argument("--nproc-per-node", type=int, required=True,
                        help="GPUs per node.")
    parser.add_argument("--master-addr", default=None,
                        help="Master node address (required when --disable-background-mode).")
    parser.add_argument("--master-port", default="6000",
                        help="Master node port.")
    parser.add_argument("--disable-background-mode", action="store_true", default=False,
                        help="Disable hostfile-based rank discovery; requires --master-addr.")
    parser.add_argument("--nsight", action="store_true", default=False,
                        help="Enable Megatron/PyTorch profiler output under --tensorboard-dir.")
    parser.add_argument("--rerun-mode", choices=("disabled", "validate_results", "report_stats"), default="disabled",
                        help="Megatron rerun engine mode. Defaults to disabled for benchmark timing.")
    args = parser.parse_args()
    if "--seq-length" not in os.sys.argv:
        args.seq_length = MODEL_CONFIGS[args.model]["seq_length"]
    return args


def build_training_args(
    tensorboard_dir: str,
    micro_bs: int,
    global_bs: int,
    seq_length: int,
    tp: int,
    pp: int,
    dp: int,
    cp: int,
    ep: int,
    sp: bool,
    use_tp_pp_dp_mapping: bool,
    model: str,
    train_iters: int,
    schedule: str,
    zero_level: str,
    enable_profiler: bool,
    rerun_mode: str,
    supported_flags: set[str] | None = None,
) -> list:
    if sp and tp == 1:
        raise ValueError("--sp requires --tp > 1")

    model_config = MODEL_CONFIGS[model]
    supported_flags = supported_flags or set()
    args = [
        "--num-layers", str(model_config["num_layers"]),
        "--hidden-size", str(model_config["hidden_size"]),
        "--num-attention-heads", str(model_config["num_attention_heads"]),
        "--kv-channels", str(model_config["kv_channels"]),
        "--ffn-hidden-size", str(model_config["ffn_hidden_size"]),
        "--seq-length", str(seq_length),
        "--max-position-embeddings", str(model_config["max_position_embeddings"]),
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--position-embedding-type", "rope",
        "--rotary-percent", "1.0",
        "--rotary-base", "1000000",
        "--use-rotary-position-embeddings",
        "--swiglu",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--group-query-attention",
        "--num-query-groups", str(model_config["num_query_groups"]),
        "--qk-layernorm",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--use-mcore-models",
        "--transformer-impl", "transformer_engine",
        "--use-flash-attn",
        "--num-experts", str(model_config["num_experts"]),
        "--moe-router-topk", str(model_config["moe_router_topk"]),
        "--moe-ffn-hidden-size", str(model_config["moe_ffn_hidden_size"]),
        "--moe-layer-freq", "1",
        "--moe-router-load-balancing-type", "aux_loss",
        "--moe-aux-loss-coeff", "0.001",
        "--moe-token-dispatcher-type", "alltoall",
        "--moe-grouped-gemm",
        "--tensor-model-parallel-size", str(tp),
        "--pipeline-model-parallel-size", str(pp),
        "--context-parallel-size", str(cp),
        "--expert-model-parallel-size", str(ep),
        "--micro-batch-size", str(micro_bs),
        "--global-batch-size", str(global_bs),
        "--train-iters", str(train_iters),
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.95",
        "--init-method-std", "0.01",
        "--clip-grad", "1.0",
        "--bf16",
        "--lr", "3.0e-4",
        "--lr-decay-style", "cosine",
        "--min-lr", "3.0e-5",
        "--lr-warmup-iters", "0",
        "--lr-decay-iters", "2",
        "--mock-data",
        "--vocab-size", str(model_config["vocab_size"]),
        "--tokenizer-type", "NullTokenizer",
        "--log-interval", "1",
        "--eval-interval", "100000",
        "--eval-iters", "0",
        "--save-interval", "100000",
        "--no-gradient-accumulation-fusion",
        "--rerun-mode", rerun_mode,
    ]
    if "--tensorboard-dir" in supported_flags:
        args.extend(["--tensorboard-dir", tensorboard_dir])
    else:
        print("[megatron] pretrain_gpt.py does not support --tensorboard-dir; skipping")
    if enable_profiler:
        profiler_args: list[str] = []
        if "--profile" in supported_flags:
            profiler_args.append("--profile")
        if "--use-pytorch-profiler" in supported_flags:
            profiler_args.append("--use-pytorch-profiler")
        if "--profile-step-start" in supported_flags:
            profiler_args.extend(["--profile-step-start", "5"])
        if "--profile-step-end" in supported_flags:
            profiler_args.extend(["--profile-step-end", "8"])
        if "--pytorch-profiler-collect-shapes" in supported_flags:
            profiler_args.append("--pytorch-profiler-collect-shapes")
        if profiler_args:
            args.extend(profiler_args)
        else:
            print("[megatron] profiler flags unsupported by pretrain_gpt.py; running without profiler flags")
    if dp > 1:
        args.extend([
            "--use-distributed-optimizer",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
            "--data-parallel-sharding-strategy",
            {
                "zero1": "optim",
                "zero2": "optim_grads",
                "zero3": "optim_grads_params",
            }[zero_level],
        ])
    if zero_level in ("zero2", "zero3"):
        args.append("--use-megatron-fsdp")
    if schedule == "interleaved1f1b":
        args.extend(["--num-layers-per-virtual-pipeline-stage", "1"])
    if sp:
        args.append("--sequence-parallel")
    if use_tp_pp_dp_mapping:
        args.append("--use-tp-pp-dp-mapping")
    return args


def main():
    # TODO: 
    # os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    args = parse_args()
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    background_mode = not args.disable_background_mode
    node_rank, master_addr = _setup_distributed(
        args.nnodes, args.master_addr, background_mode,
    )

    print("=" * 60)
    print(f"Node rank:   {node_rank} / {args.nnodes}")
    print(f"Master:      {master_addr}:{args.master_port}")
    print(f"GPUs/node:   {args.nproc_per_node}")
    print("=" * 60)

    training_args = build_training_args(
        args.tensorboard_dir, args.micro_bs, args.global_bs, args.seq_length,
        args.tp, args.pp, args.dp, args.cp, args.ep, args.sp,
        args.use_tp_pp_dp_mapping, args.model, args.train_iters, args.schedule, args.zero_level,
        args.nsight, args.rerun_mode,
        _detect_supported_megatron_flags(),
    )

    from torch.distributed.run import main as torchrun_main

    # Use torchrun's static rendezvous path for multi-node runs so every node
    # consistently connects to the fixed master_addr/master_port pair.
    torchrun_args = [
        f"--nproc_per_node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--node_rank={node_rank}",
        "--rdzv_backend=static", # TODO:
        f"--master_addr={master_addr}",
        f"--master_port={args.master_port}",
    ]
    torchrun_main([
        *torchrun_args,
        "pretrain_gpt.py",
        *training_args,
    ])


if __name__ == "__main__":
    main()
