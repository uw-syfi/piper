"""End-to-end test for data-parallel weight consistency.

When a model layer is replicated across DP ranks, every replica must:
  1. Start with identical weights (initialization sync).
  2. End with identical weights after training (gradient all-reduce).

This test runs a simple model with dp_degree=2, pp_degree=1 and verifies
that both DP replicas produce the same final parameters.
"""

import json
import os
import tempfile

import ray
import torch
import torch.nn as nn

from src.compile import piper_setup
from src.coordinator import PiperProgramCoordinator
from src.piper import annotate, piper_exec_dag
from src.state import piper_metadata


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimpleModel(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with annotate("PP"):
            h = self.embed(x)
            return self.proj(h)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
DIM = 8
BATCH_SIZE = 2
SEQ_LEN = 4
NUM_STEPS = 3
LR = 0.01

# dp_degree=2 (two devices), pp_degree=1.
_DP2_SCHEDULE = [
    {"op": "place", "filter": {"PP": 0}, "devices": [0, 1], "stream": "pp_stream"},
    {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1]},
    {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
    {
        "op": "order",
        "filters": [
            [{"PP": 0, "MB": 0, "PASS": "F"}],
            [{"PP": 0, "MB": 0, "PASS": "B"}],
        ],
    },
]


def _write_schedule(schedule: list[dict], tmpdir: str) -> str:
    path = os.path.join(tmpdir, "schedule.json")
    with open(path, "w") as f:
        json.dump(schedule, f)
    return path


# ---------------------------------------------------------------------------
# Training function (runs inside each run_dp_rank worker)
# ---------------------------------------------------------------------------

def _training_main(args, pg):
    """Piper training function executed by each DP rank."""
    os.environ["PIPER_DEVICE"] = args.device

    torch.manual_seed(args.seed)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))

    _ce = torch.nn.CrossEntropyLoss()
    loss_fn = lambda output, labels: _ce(
        output.view(-1, output.size(-1)), labels.view(-1)
    )

    piper_setup(
        SimpleModel,
        model_args=(args.vocab_size, args.dim),
        optim_fn=lambda params: torch.optim.SGD(params, lr=args.lr),
        example_inputs=[x],
        example_outputs=y,
        model_dtype=torch.float32,
        pg=pg,
        schedule_directives_file=args.schedule_file,
        use_inductor=False,
    )

    actors = piper_metadata.actors

    # Snapshot initial parameters.
    initial_params = {}
    for pp_rank, actor in actors.items():
        initial_params[pp_rank] = ray.get(actor.get_params_cpu.remote())

    # Run training iterations.
    for _ in range(args.num_steps):
        piper_exec_dag(loss_fn)

    # Snapshot final parameters.
    final_params = {}
    for pp_rank, actor in actors.items():
        final_params[pp_rank] = ray.get(actor.get_params_cpu.remote())

    return {
        "dp_rank": int(os.environ["PIPER_DP_RANK"]),
        "initial_params": initial_params,
        "final_params": final_params,
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def _run_dp_piper(schedule: list[dict], *, seed: int = 42, device: str = "cpu") -> list[dict]:
    """Launch Piper with DP via the coordinator and return per-DP-rank results."""
    os.environ["PIPER_DEVICE"] = device

    with tempfile.TemporaryDirectory() as tmpdir:
        schedule_file = _write_schedule(schedule, tmpdir)

        from types import SimpleNamespace
        args = SimpleNamespace(
            vocab_size=VOCAB_SIZE,
            dim=DIM,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_steps=NUM_STEPS,
            lr=LR,
            seed=seed,
            schedule_file=schedule_file,
            device=device,
        )

        ray.init(
            namespace="test_dp_weights",
            log_to_driver=True,
            include_dashboard=False,
        )
        try:
            coordinator = PiperProgramCoordinator.remote(
                schedule_directives_file=schedule_file,
            )
            dp_results = ray.get(
                coordinator.run_program.remote(_training_main, None, args, None)
            )
            return dp_results
        finally:
            ray.shutdown()


def test_dp_weights_consistent():
    """DP replicas must start and end with identical weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = _run_dp_piper(_DP2_SCHEDULE, seed=42, device=device)
    assert len(results) == 2, f"Expected 2 DP results, got {len(results)}"

    dp0 = next(r for r in results if r["dp_rank"] == 0)
    dp1 = next(r for r in results if r["dp_rank"] == 1)

    # Both DP ranks have pp_rank=0 only.
    assert 0 in dp0["initial_params"] and 0 in dp1["initial_params"]

    init0 = dp0["initial_params"][0]
    init1 = dp1["initial_params"][0]
    final0 = dp0["final_params"][0]
    final1 = dp1["final_params"][0]

    # Check initial weights match across DP ranks.
    for name in init0:
        assert name in init1, f"Param {name} missing on DP rank 1"
        torch.testing.assert_close(
            init0[name], init1[name],
            atol=0, rtol=0,
            msg=f"Initial param {name} differs across DP ranks",
        )

    # Check final weights match across DP ranks.
    for name in final0:
        assert name in final1, f"Param {name} missing on DP rank 1"
        torch.testing.assert_close(
            final0[name], final1[name],
            atol=1e-5, rtol=1e-4,
            msg=f"Final param {name} differs across DP ranks after {NUM_STEPS} steps",
        )

    # Sanity: weights actually changed during training.
    any_changed = False
    for name in init0:
        if not torch.equal(init0[name], final0[name]):
            any_changed = True
            break
    assert any_changed, "No parameters changed during training — test is vacuous"


# ---------------------------------------------------------------------------
# PP=2 DP=2 model & schedule
# ---------------------------------------------------------------------------

class TwoStageModel(nn.Module):
    """A simple two-stage model for PP=2 testing."""

    def __init__(self, vocab_size: int = 32, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.hidden = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with annotate("PP"):
            h = self.embed(x)
            h = self.hidden(h)
        with annotate("PP"):
            return self.proj(h)


# PP=2, DP=2 → 4 devices. PP0 on {0,2}, PP1 on {1,3}.
_PP2_DP2_SCHEDULE = [
    {"op": "place", "filter": {"PP": 0}, "devices": [0, 2], "stream": "pp_stream"},
    {"op": "place", "filter": {"PP": 1}, "devices": [1, 3], "stream": "pp_stream"},
    {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 2]},
    {"op": "replicate", "filter": {"PP": 1}, "devices": [1, 3]},
    {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
    {
        "op": "order",
        "filters": [
            [{"PP": 0, "MB": 0, "PASS": "F"}],
            [{"PP": 0, "MB": 0, "PASS": "B"}],
        ],
    },
    {
        "op": "order",
        "filters": [
            [{"PP": 1, "MB": 0, "PASS": "F"}],
            [{"PP": 1, "MB": 0, "PASS": "B"}],
        ],
    },
]


def _pp2_dp2_training_main(args, pg):
    """Piper training function for PP=2, DP=2."""
    os.environ["PIPER_DEVICE"] = args.device

    torch.manual_seed(args.seed)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))

    _ce = torch.nn.CrossEntropyLoss()
    loss_fn = lambda output, labels: _ce(
        output.view(-1, output.size(-1)), labels.view(-1)
    )

    piper_setup(
        TwoStageModel,
        model_args=(args.vocab_size, args.dim),
        optim_fn=lambda params, **kw: torch.optim.SGD(params, lr=args.lr),
        example_inputs=[x],
        example_outputs=y,
        model_dtype=torch.float32,
        pg=pg,
        schedule_directives_file=args.schedule_file,
        use_inductor=False,
    )

    actors = piper_metadata.actors

    initial_params = {}
    for pp_rank, actor in actors.items():
        initial_params[pp_rank] = ray.get(actor.get_params_cpu.remote())

    for _ in range(args.num_steps):
        piper_exec_dag(loss_fn)

    final_params = {}
    for pp_rank, actor in actors.items():
        final_params[pp_rank] = ray.get(actor.get_params_cpu.remote())

    return {
        "dp_rank": int(os.environ["PIPER_DP_RANK"]),
        "initial_params": initial_params,
        "final_params": final_params,
    }


def test_pp2_dp2_weights_consistent():
    """PP=2 DP=2: replicas of each PP stage must have identical weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["PIPER_DEVICE"] = device

    with tempfile.TemporaryDirectory() as tmpdir:
        schedule_file = _write_schedule(_PP2_DP2_SCHEDULE, tmpdir)

        from types import SimpleNamespace
        args = SimpleNamespace(
            vocab_size=VOCAB_SIZE,
            dim=DIM,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            num_steps=NUM_STEPS,
            lr=LR,
            seed=42,
            schedule_file=schedule_file,
            device=device,
        )

        ray.init(
            namespace="test_pp2_dp2_weights",
            log_to_driver=True,
            include_dashboard=False,
        )
        try:
            coordinator = PiperProgramCoordinator.remote(
                schedule_directives_file=schedule_file,
            )
            results = ray.get(
                coordinator.run_program.remote(_pp2_dp2_training_main, None, args, None)
            )
        finally:
            ray.shutdown()

    assert len(results) == 2, f"Expected 2 DP results, got {len(results)}"

    dp0 = next(r for r in results if r["dp_rank"] == 0)
    dp1 = next(r for r in results if r["dp_rank"] == 1)

    # Both DP ranks should have PP stages 0 and 1.
    for pp_rank in [0, 1]:
        assert pp_rank in dp0["initial_params"], f"PP rank {pp_rank} missing on DP rank 0"
        assert pp_rank in dp1["initial_params"], f"PP rank {pp_rank} missing on DP rank 1"

        init0 = dp0["initial_params"][pp_rank]
        init1 = dp1["initial_params"][pp_rank]
        final0 = dp0["final_params"][pp_rank]
        final1 = dp1["final_params"][pp_rank]

        # Initial weights must match across DP ranks.
        for name in init0:
            assert name in init1, f"PP{pp_rank} param {name} missing on DP rank 1"
            torch.testing.assert_close(
                init0[name], init1[name],
                atol=0, rtol=0,
                msg=f"PP{pp_rank} initial param {name} differs across DP ranks",
            )

        # Final weights must match across DP ranks.
        for name in final0:
            assert name in final1, f"PP{pp_rank} param {name} missing on DP rank 1"
            torch.testing.assert_close(
                final0[name], final1[name],
                atol=1e-5, rtol=1e-4,
                msg=f"PP{pp_rank} final param {name} differs across DP ranks after {NUM_STEPS} steps",
            )

    # Sanity: weights actually changed during training for at least one PP stage.
    any_changed = False
    for pp_rank in [0, 1]:
        for name in dp0["initial_params"][pp_rank]:
            if not torch.equal(dp0["initial_params"][pp_rank][name], dp0["final_params"][pp_rank][name]):
                any_changed = True
                break
        if any_changed:
            break
    assert any_changed, "No parameters changed during training — test is vacuous"
