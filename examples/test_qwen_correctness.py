"""Qwen end-to-end parameter parity test for Piper schedules."""

from __future__ import annotations

import argparse
import os

import ray
import torch

from models.qwen3 import QWEN_MODEL_NAMES, PiperQwen3Model, create_qwen3_config
from src.compile import piper_setup
from src.model_state import (
    DEFAULT_MODEL_SEED,
    initialize_model_parameters_,
)
from src.piper import piper_exec_dag
from src.schedule import load_schedule_directives
from src.state import LOG_LEVEL, create_logger, piper_metadata


logger = create_logger("test_qwen_correctness", LOG_LEVEL)


def _create_optimizer(parameters, **kwargs):
    kwargs["fused"] = True
    return torch.optim.Adam(parameters, **kwargs)


def main(args, pg) -> dict:
    config = create_qwen3_config(args.model)
    num_stages = int(
        getattr(args, "num_stages", 0)
        or _derive_num_stages(args.schedule_directives_file)
    )
    dp_degree = int(os.environ["PIPER_DP_DEGREE"])

    input_generator = torch.Generator(device="cpu")
    input_generator.manual_seed(args.model_seed + 1)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.seq_len),
        generator=input_generator,
    )
    labels = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.seq_len),
        generator=input_generator,
    )

    device = torch.device("cuda")
    with torch.device(device):
        reference_model = PiperQwen3Model(config, num_stages).to(torch.bfloat16)
    initialize_model_parameters_(reference_model, args.model_seed)
    reference_model.train()

    loss_module = torch.nn.CrossEntropyLoss()

    def loss_fn(output, target):
        return loss_module(
            output.float().view(-1, output.size(-1)),
            target.view(-1),
        )

    piper_setup(
        PiperQwen3Model,
        model_args=(config, num_stages),
        optim_fn=_create_optimizer,
        example_inputs=[tokens],
        example_outputs=labels,
        model_dtype=torch.bfloat16,
        pg=pg,
        temp_dir=args.temp_dir,
        visualize_dag=args.viz,
        const_attrs={"rope_cache": reference_model.rope_cache.detach().cpu()},
        use_inductor=args.use_inductor,
        pp_outer=args.pp_outer,
        schedule_directives_file=args.schedule_directives_file,
        model_seed=args.model_seed,
    )
    schedule_info = dict(piper_metadata.schedule_info or {})
    num_microbatches = int(schedule_info["num_microbatches"])

    initial_piper_state = _export_piper_parameter_state()
    initial_reference_state = _eager_parameter_state(reference_model)
    _assert_parameter_states_close(
        initial_piper_state,
        initial_reference_state,
        rtol=0.0,
        atol=0.0,
        phase="initialization",
    )

    piper_loss_samples = []
    for _ in range(args.steps):
        piper_loss_samples.append(
            [float(loss) for loss in piper_exec_dag(loss_fn)]
        )

    final_piper_state = _export_piper_parameter_state()
    reference_loss_samples = _train_reference(
        reference_model,
        tokens.to(device),
        labels.to(device),
        loss_fn,
        steps=args.steps,
        num_microbatches=num_microbatches,
        dp_degree=dp_degree,
    )
    _assert_loss_samples_close(piper_loss_samples, reference_loss_samples)
    final_reference_state = _eager_parameter_state(reference_model)
    max_abs_error = _assert_parameter_states_close(
        final_piper_state,
        final_reference_state,
        rtol=args.rtol,
        atol=args.atol,
        phase=f"after {args.steps} training step(s)",
        baseline=initial_reference_state,
    )

    logger.info(
        "Qwen %s parameter parity passed: parameters=%d max_abs_error=%g",
        args.model,
        len(final_piper_state),
        max_abs_error,
    )
    return {
        "dp_rank": int(os.environ["PIPER_DP_RANK"]),
        "model": args.model,
        "schedule": schedule_info.get("name"),
        "pp": schedule_info.get("pp_degree"),
        "dp": dp_degree,
        "batch_size": args.batch_size,
        "num_microbatches": num_microbatches,
        "seq_len": args.seq_len,
        "iter_times_s": [],
        "peak_memory_by_rank": {},
        "parameter_count": len(final_piper_state),
        "max_parameter_abs_error": max_abs_error,
    }


def _train_reference(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_fn,
    *,
    steps: int,
    num_microbatches: int,
    dp_degree: int,
) -> list[list[float]]:
    optimizer = _create_optimizer(model.parameters())
    loss_samples: list[list[float]] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Piper's current split directive duplicates the scheduled graph and
        # reuses the loaded batch for every MB slot. Mirror that exact gradient
        # accumulation contract in the eager reference.
        step_losses = []
        for _ in range(num_microbatches):
            loss = loss_fn(model(tokens), labels)
            step_losses.append(float(loss.detach()))
            loss.backward()
        loss_samples.append(step_losses)
        # Piper's DP collectives are sums rather than averages. Every replica
        # receives the same deterministic batch in this correctness test.
        if dp_degree > 1:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(dp_degree)
        optimizer.step()
    torch.cuda.synchronize()
    return loss_samples


def _assert_loss_samples_close(
    actual: list[list[float]], expected: list[list[float]]
) -> None:
    if len(actual) != len(expected) or any(
        len(actual_step) != len(expected_step)
        for actual_step, expected_step in zip(actual, expected)
    ):
        raise AssertionError(
            f"loss sample shape mismatch: piper={actual} eager={expected}"
        )
    for step, (actual_step, expected_step) in enumerate(zip(actual, expected)):
        for microbatch, (actual_loss, expected_loss) in enumerate(
            zip(actual_step, expected_step)
        ):
            if not torch.isclose(
                torch.tensor(actual_loss),
                torch.tensor(expected_loss),
                rtol=1e-4,
                atol=1e-4,
            ):
                raise AssertionError(
                    f"loss mismatch at step={step} microbatch={microbatch}: "
                    f"piper={actual_loss:g} eager={expected_loss:g}"
                )


def _export_piper_parameter_state() -> dict[str, torch.Tensor]:
    actor_states = ray.get([
        actor.export_model_parameter_state.remote()
        for actor in piper_metadata.actors.values()
    ])
    merged: dict[str, torch.Tensor] = {}
    for actor_state in actor_states:
        overlap = set(merged).intersection(actor_state)
        if overlap:
            raise RuntimeError(
                f"Piper stages exported duplicate parameters: {sorted(overlap)[:8]}"
            )
        merged.update(actor_state)
    return merged


def _eager_parameter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _assert_parameter_states_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
    phase: str,
    baseline: dict[str, torch.Tensor] | None = None,
) -> float:
    missing = sorted(set(expected).difference(actual))
    unexpected = sorted(set(actual).difference(expected))
    if missing or unexpected:
        raise AssertionError(
            f"parameter-name mismatch during {phase}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    mismatches: list[tuple[float, str, str]] = []
    max_abs_error = 0.0
    for name in sorted(expected):
        actual_tensor = actual[name]
        expected_tensor = expected[name]
        if actual_tensor.shape != expected_tensor.shape:
            raise AssertionError(
                f"parameter {name!r} has shape {tuple(actual_tensor.shape)} "
                f"during {phase}, expected {tuple(expected_tensor.shape)}"
            )
        abs_error = (
            (actual_tensor.float() - expected_tensor.float()).abs().max().item()
            if actual_tensor.numel()
            else 0.0
        )
        max_abs_error = max(max_abs_error, abs_error)
        if not torch.allclose(
            actual_tensor,
            expected_tensor,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ):
            flat_error = (
                actual_tensor.float() - expected_tensor.float()
            ).abs().reshape(-1)
            worst_idx = int(flat_error.argmax().item())
            actual_value = float(actual_tensor.reshape(-1)[worst_idx])
            expected_value = float(expected_tensor.reshape(-1)[worst_idx])
            detail = (
                f"{name}[{worst_idx}]: max_abs_error={abs_error:g}, "
                f"piper={actual_value:g}, eager={expected_value:g}"
            )
            if baseline is not None:
                initial_value = float(baseline[name].reshape(-1)[worst_idx])
                detail += (
                    f", initial={initial_value:g}, "
                    f"piper_delta={actual_value - initial_value:g}, "
                    f"eager_delta={expected_value - initial_value:g}"
                )
            mismatches.append((abs_error, name, detail))

    if mismatches:
        mismatches.sort(reverse=True)
        details = ", ".join(
            detail for _error, _name, detail in mismatches[:8]
        )
        mismatch_names = [name for _error, name, _detail in mismatches]
        raise AssertionError(
            f"{len(mismatches)} parameter tensor(s) diverged during {phase} "
            f"(rtol={rtol}, atol={atol}); {details}; "
            f"mismatched_names={mismatch_names}"
        )
    return max_abs_error


def _derive_num_stages(schedule_directives_file: str) -> int:
    directives = load_schedule_directives(schedule_directives_file)
    num_stages = sum(
        1
        for directive in directives
        if isinstance(directive, dict) and directive.get("op") == "place"
    )
    if num_stages <= 0:
        raise ValueError(
            "schedule directives file must contain at least one place directive: "
            f"{schedule_directives_file}"
        )
    return num_stages


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare Qwen Piper training against one eager GPU"
    )
    parser.add_argument("--model", choices=QWEN_MODEL_NAMES, default="9M")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--schedule-directives-file",
        default="examples/base-schedules/pp2.json",
    )
    parser.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument(
        "--use-inductor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pp-outer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0 or args.seq_len <= 0:
        parser.error("--batch-size and --seq-len must be positive")
    if args.rtol < 0 or args.atol < 0:
        parser.error("--rtol and --atol must be non-negative")
    return args
