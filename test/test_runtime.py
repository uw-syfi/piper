"""End-to-end tests of the single-driver runtime on CPU (LLaMA debug, dp=2, pp=1)."""
import json
import os

os.environ.setdefault("PIPER_DEVICE", "cpu")

import pytest
import ray
import torch

from examples.models.llama3 import LLAMA_DEBUG, Transformer, precompute_freqs_cis
from src.compile import compile
from src.driver import StepFailed, setup
from src.topology import Topology

SEQ_LEN = 16
BATCH = 2


@pytest.fixture(scope="module")
def ray_session():
    ray.init(include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def schedule(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("schedule") / "dp2.json"
    path.write_text(json.dumps([
        {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
        {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1]},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2},
    ]))
    return str(path)


def _batch():
    torch.manual_seed(0)
    x = torch.randint(0, LLAMA_DEBUG.vocab_size, (BATCH, SEQ_LEN))
    y = torch.randint(0, LLAMA_DEBUG.vocab_size, (BATCH, SEQ_LEN))
    return x, y


def _loss_fn(output, labels):
    return torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), labels.view(-1))


def _const_attrs():
    cfg = LLAMA_DEBUG
    causal = torch.triu(torch.full((SEQ_LEN, SEQ_LEN), float("-inf")), diagonal=1)
    return {
        "freqs_cis": precompute_freqs_cis(cfg.dim // cfg.n_heads, SEQ_LEN, cfg.rope_theta),
        "mask": torch.hstack([torch.zeros((SEQ_LEN, 0)), causal]),
    }


def _build_runtime(schedule: str):
    x, _ = _batch()
    topology = Topology.from_schedule(schedule)
    program = compile(
        Transformer,
        model_args=(LLAMA_DEBUG, SEQ_LEN, 1),
        example_inputs=[x],
        schedule_file=schedule,
    )
    rt = setup(topology)
    rt.load(program, torch.optim.Adam, const_attrs=_const_attrs())
    return rt, program


def _params(rt, slot) -> dict:
    return ray.get(rt.actors[slot].get_params_cpu.remote())


def _assert_replicas_equal(rt) -> None:
    ref = _params(rt, (0, 0))
    other = _params(rt, (1, 0))
    assert ref.keys() == other.keys() and ref
    mismatched = [name for name in ref if not torch.equal(ref[name], other[name])]
    assert not mismatched, f"DP replicas differ on {mismatched[:5]}"


def test_setup_load_step(ray_session, schedule) -> None:
    rt, program = _build_runtime(schedule)
    assert program.kind == "dag" and program.pp == 1
    assert program.input_pp_ranks == [0] and program.loss_pp_ranks == [0]
    assert sorted(rt.actors) == [(0, 0), (1, 0)]
    assert rt.groups == {"dp": [[0, 1]], "pp": [[0], [1]]}
    assert rt.standbys == []

    # load broadcasts replicated parameters, so both replicas start identical.
    _assert_replicas_equal(rt)

    x, y = _batch()
    inputs, labels = [[x], [x]], [y, y]
    result = rt.step(0, inputs, labels, _loss_fn)
    assert result.next_iter == 1 and result.recovery is None
    assert len(result.losses) == 2
    # Two microbatches per slot; identical data and weights give identical losses.
    assert all(len(l) == 2 for l in result.losses)
    assert result.losses[0] == result.losses[1]
    assert all(torch.isfinite(torch.tensor(l)).all() for l in result.losses)

    second = rt.step(result.next_iter, inputs, labels, _loss_fn)
    assert second.next_iter == 2
    assert second.losses[0] == second.losses[1]
    assert second.losses[0][0] != result.losses[0][0], "optimizer step had no effect"
    _assert_replicas_equal(rt)

    with pytest.raises(ValueError, match="expects 2 input lists"):
        rt.step(2, [[x]], [y], _loss_fn)


def test_step_raises_on_dead_actor(ray_session, schedule) -> None:
    rt, _ = _build_runtime(schedule)
    ray.kill(rt.actors[(1, 0)])
    x, y = _batch()
    with pytest.raises(StepFailed) as info:
        rt.step(0, [[x], [x]], [y, y], _loss_fn)
    assert info.value.slots == [(1, 0)]
    assert info.value.kind == "actor_died"
