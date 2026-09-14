"""End-to-end tests for tied embedding support.

Tied embeddings share the same parameter tensor across pipeline stages (e.g.,
input embedding at PP=0 and output projection at PP=last).  These tests run
Piper end-to-end and verify that the final weights match a single-device
reference PyTorch run.

Two cases:
  1. Same device   -- both PP stages placed on the same device (V-layout).
  2. Different devices -- PP stages on separate devices (requires gradient sync
     for the shared weight).  Currently expected to fail until issue #13 lands.
"""

import json
import os
import tempfile

import pytest
import ray
import torch
import torch.nn as nn

from src.compile import piper_setup
from src.piper import annotate, piper_exec_dag
from src.schedule import derive_schedule_info
from src.state import piper_metadata


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TiedEmbeddingModel(nn.Module):
    """Minimal model with tied input/output embeddings across two PP stages."""

    def __init__(self, vocab_size: int = 32, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.hidden = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight  # weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with annotate("PP"):
            h = self.embed(x)
            h = torch.relu(self.hidden(h))
        with annotate("PP"):
            return self.out_proj(h)


# ---------------------------------------------------------------------------
# Schedule definitions
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
DIM = 8
BATCH_SIZE = 2
SEQ_LEN = 4
NUM_STEPS = 3
LR = 0.01

# Both PP stages on device 0 (V-layout, pp_degree=1).
_SAME_DEVICE_SCHEDULE = [
    {"op": "place", "filter": {"PP": 0}, "devices": [0], "stream": "pp_stream"},
    {"op": "place", "filter": {"PP": 1}, "devices": [0], "stream": "pp_stream"},
    {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
    {
        "op": "order",
        "filters": [
            [{"PP": 0, "MB": 0, "PASS": "F"}],
            [{"PP": 1, "MB": 0, "PASS": "F"}],
            [{"PP": 1, "MB": 0, "PASS": "B"}],
            [{"PP": 0, "MB": 0, "PASS": "B"}],
        ],
    },
]

# PP stages on different devices (pp_degree=2).
_DIFF_DEVICE_SCHEDULE = [
    {"op": "place", "filter": {"PP": 0}, "devices": [0], "stream": "pp_stream"},
    {"op": "place", "filter": {"PP": 1}, "devices": [1], "stream": "pp_stream"},
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


def _write_schedule(schedule: list[dict], tmpdir: str) -> str:
    path = os.path.join(tmpdir, "schedule.json")
    with open(path, "w") as f:
        json.dump(schedule, f)
    return path


# ---------------------------------------------------------------------------
# Reference single-device training
# ---------------------------------------------------------------------------

def _fx_name_to_module_name(fx_name: str) -> str | None:
    """Convert an FX placeholder name to a model parameter name.

    Dynamo generates names like ``l__self___embed_weight``,
    ``l_self_embed_weight``, or ``l_self_modules_embed_parameters_weight_``.
    Strip the prefix and known wrapper segments, then try all underscore-to-dot
    splits to find a valid parameter path.
    """
    for prefix in ("l__self___", "l_self_"):
        if fx_name.startswith(prefix):
            fx_name = fx_name[len(prefix):]
            break
    # Strip Dynamo wrapper segments: ``modules_`` prefix and ``_parameters`` infix.
    fx_name = fx_name.removeprefix("modules_")
    fx_name = fx_name.replace("_parameters_", "_")
    fx_name = fx_name.rstrip("_")
    # Try all possible underscore-to-dot splits to reconstruct ``module.param``.
    parts = fx_name.split("_")
    for i in range(1, len(parts)):
        candidate = ".".join(["_".join(parts[:i]), "_".join(parts[i:])])
        yield candidate


def _build_param_mapping(
    piper_params: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, str]:
    """Map Piper FX parameter names to nn.Module parameter names."""
    model_params = dict(model.named_parameters())
    mapping: dict[str, str] = {}
    for piper_name in piper_params:
        for candidate in _fx_name_to_module_name(piper_name):
            if candidate in model_params:
                mapping[piper_name] = candidate
                break
    return mapping


def _reference_training(initial_params: dict[int, dict], seed: int) -> dict:
    """Run reference PyTorch training with the same initial weights.

    ``initial_params`` maps pp_rank -> {param_name: tensor}.  We reconstruct
    a TiedEmbeddingModel on CPU, load those weights, and run the same forward /
    loss / backward / optimizer steps.

    Returns the final named parameters (CPU) after NUM_STEPS.
    """
    torch.manual_seed(seed)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    model = TiedEmbeddingModel(VOCAB_SIZE, DIM)

    # Merge all per-rank params into a single dict.
    piper_params: dict[str, torch.Tensor] = {}
    for pp_rank_params in initial_params.values():
        piper_params.update(pp_rank_params)

    mapping = _build_param_mapping(piper_params, model)

    # Load initial weights into the model.
    model_params = dict(model.named_parameters())
    with torch.no_grad():
        for piper_name, module_name in mapping.items():
            model_params[module_name].copy_(piper_params[piper_name])

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    ce = torch.nn.CrossEntropyLoss()

    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        out = model(x)
        loss = ce(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

    return {
        "params": {name: p.detach().cpu().clone() for name, p in model.named_parameters()},
        "mapping": mapping,
    }


# ---------------------------------------------------------------------------
# Piper runner
# ---------------------------------------------------------------------------

def _run_piper(schedule: list[dict], *, seed: int = 42, device: str = "cpu") -> dict:
    """Set up Piper with the given schedule, run training, and return results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        schedule_file = _write_schedule(schedule, tmpdir)
        info = derive_schedule_info(schedule, schedule_file)

        os.environ["PIPER_DEVICE"] = device
        os.environ["PIPER_DP_RANK"] = "0"
        os.environ["PIPER_DP_DEGREE"] = str(info["dp_degree"])
        os.environ["PIPER_PP_DEGREE"] = str(info["pp_degree"])
        os.environ["PIPER_WORLD_SIZE"] = str(info["pp_degree"] * info["dp_degree"])

        torch.manual_seed(seed)
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        _ce = torch.nn.CrossEntropyLoss()
        loss_fn = lambda output, labels: _ce(
            output.view(-1, output.size(-1)), labels.view(-1)
        )

        ray.init(
            namespace="test_tied_embeddings",
            log_to_driver=True,
            include_dashboard=False,
        )
        try:
            piper_setup(
                TiedEmbeddingModel,
                model_args=(VOCAB_SIZE, DIM),
                optim_fn=lambda params: torch.optim.SGD(params, lr=LR),
                example_inputs=[x],
                example_outputs=y,
                model_dtype=torch.float32,
                schedule_directives_file=schedule_file,
                use_inductor=False,
            )

            actors = piper_metadata.actors

            # Snapshot initial parameters from every actor.
            initial_params = {}
            for pp_rank, actor in actors.items():
                initial_params[pp_rank] = ray.get(actor.get_params_cpu.remote())

            # Run training iterations.
            for _ in range(NUM_STEPS):
                piper_exec_dag(loss_fn)

            # Snapshot final parameters.
            final_params = {}
            for pp_rank, actor in actors.items():
                final_params[pp_rank] = ray.get(actor.get_params_cpu.remote())

            return {
                "initial_params": initial_params,
                "final_params": final_params,
            }
        finally:
            ray.shutdown()


def _assert_params_match(piper_result: dict, ref: dict) -> None:
    """Assert that Piper's final params match the reference."""
    mapping = ref["mapping"]
    final_params = piper_result["final_params"]

    for piper_name, module_name in mapping.items():
        # Collect all copies of this param across ranks.
        copies = []
        for pp_rank, pp_rank_params in final_params.items():
            if piper_name in pp_rank_params:
                copies.append((pp_rank, pp_rank_params[piper_name]))
        assert copies, f"Piper param {piper_name} not found in final params"

        # All copies of the same param must agree (catches broken tied-weight sync).
        for pp_rank, val in copies[1:]:
            torch.testing.assert_close(
                val,
                copies[0][1],
                atol=1e-4,
                rtol=1e-4,
                msg=f"Param {piper_name} ({module_name}) diverged: "
                    f"rank {copies[0][0]} vs rank {pp_rank}",
            )

        ref_val = ref["params"][module_name]
        torch.testing.assert_close(
            copies[0][1],
            ref_val,
            atol=1e-4,
            rtol=1e-4,
            msg=f"Param {piper_name} ({module_name}) mismatch after {NUM_STEPS} steps",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tied_embedding_same_device():
    """Tied embedding with both PP stages on the same device.

    This is the V-layout case.  Since both stages share the same device,
    gradient accumulation into the tied weight happens naturally.  The final
    weights should match a single-device reference run.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = _run_piper(_SAME_DEVICE_SCHEDULE, seed=42, device=device)
    ref = _reference_training(result["initial_params"], seed=42)
    _assert_params_match(result, ref)


def test_tied_embedding_different_devices():
    """Tied embedding with PP stages on different devices.

    When the tied weight lives on two devices, Piper must insert gradient
    synchronization (e.g. all-reduce) so both copies see the full gradient
    before the optimizer step.  This test is expected to fail until issue #13
    is implemented.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = _run_piper(_DIFF_DEVICE_SCHEDULE, seed=42, device=device)
    ref = _reference_training(result["initial_params"], seed=42)
    _assert_params_match(result, ref)
