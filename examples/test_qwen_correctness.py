"""Qwen end-to-end parameter parity test for Piper schedules."""

from __future__ import annotations

import argparse
import os
from functools import partial

import ray
import torch

from models.qwen3 import (
    AFDQwen3Model,
    QWEN_MODEL_NAMES,
    PiperQwen3Model,
    create_qwen3_config,
    qwen_expert_shard_degree,
)
from src.compile import piper_setup
from src.model_state import (
    DEFAULT_MODEL_SEED,
    initialize_model_parameters_,
)
from src.piper import piper_exec_dag
from src.schedule import load_schedule_directives
from src.state import LOG_LEVEL, create_logger, piper_metadata


logger = create_logger("test_qwen_correctness", LOG_LEVEL)


def _create_optimizer(parameters, *, optimizer_name="adam", **kwargs):
    kwargs["fused"] = True
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if optimizer_name == "sgd":
        kwargs.setdefault("lr", 1e-2)
        return torch.optim.SGD(parameters, **kwargs)
    raise ValueError(f"unsupported optimizer {optimizer_name!r}")


def main(args, pg) -> dict:
    config = create_qwen3_config(args.model)
    num_stages = int(
        getattr(args, "num_stages", 0)
        or _derive_num_stages(args.schedule_directives_file)
    )
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
    initialize_model_parameters_(
        reference_model,
        args.model_seed,
        route_parameter_axes=reference_model.route_parameter_axes(),
    )
    reference_model.train()

    loss_module = torch.nn.CrossEntropyLoss()

    def loss_fn(output, target):
        return loss_module(
            output.float().view(-1, output.size(-1)),
            target.view(-1),
        )

    model_cls = _model_class_for_schedule(args.schedule_directives_file)
    shard_degree = qwen_expert_shard_degree(
        load_schedule_directives(args.schedule_directives_file)
    )
    logger.info("Using Qwen wrapper %s", model_cls.__name__)
    piper_setup(
        model_cls,
        model_args=(config, num_stages, shard_degree),
        optim_fn=partial(_create_optimizer, optimizer_name=args.optimizer),
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
        dynamic_input_dims={0: (1,)},
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
        actor_group_width=int(schedule_info["input_actor_group_size"]),
        optimizer_name=args.optimizer,
    )
    _assert_loss_samples_close(piper_loss_samples, reference_loss_samples)
    final_reference_state = _eager_parameter_state(reference_model)
    max_abs_error = _assert_parameter_states_close(
        final_piper_state,
        final_reference_state,
        rtol=args.rtol,
        atol=args.atol,
        expert_atol=args.expert_atol,
        dense_max_ulps=args.dense_max_ulps,
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
        "program_rank": 0,
        "model": args.model,
        "schedule": schedule_info.get("name"),
        "pp": schedule_info.get("pp_degree"),
        "actor_groups": schedule_info.get("actor_groups"),
        "world_size": schedule_info.get("world_size"),
        "input_actor_group_size": schedule_info.get(
            "input_actor_group_size", 1
        ),
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
    actor_group_width: int,
    optimizer_name: str,
) -> list[list[float]]:
    if tokens.shape[0] < actor_group_width or tokens.shape[0] % actor_group_width:
        raise ValueError(
            f"reference global microbatch size {tokens.shape[0]} must be at "
            f"least and divisible by ActorGroup width {actor_group_width}"
        )
    token_shards = tokens.chunk(actor_group_width, dim=0)
    label_shards = labels.chunk(actor_group_width, dim=0)
    route_owned_parameter_names = set(model.route_parameter_axes())
    optimizer = _create_optimizer(
        model.parameters(), optimizer_name=optimizer_name
    )
    loss_samples: list[list[float]] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Each ActorGroup local rank executes a distinct, equal slice of the
        # global microbatch. The split directive duplicates that same local
        # slice for each MB slot, so reproduce every rank/MB contribution.
        step_losses = []
        for _ in range(num_microbatches):
            local_losses = []
            for token_shard, label_shard in zip(token_shards, label_shards):
                loss = loss_fn(model(token_shard), label_shard)
                local_losses.append(float(loss.detach()))
                loss.backward()
            step_losses.append(sum(local_losses) / actor_group_width)
        # Piper all-reduces and averages replicated dense gradients. Complete
        # expert parameters are owned by exactly one shard rank, receive all
        # distinct routed contributions, and are intentionally not divided.
        if actor_group_width > 1:
            for name, parameter in model.named_parameters():
                if name in route_owned_parameter_names:
                    continue
                if parameter.grad is None:
                    raise RuntimeError(
                        f"replicated reference parameter {name!r} has no gradient"
                    )
                parameter.grad.div_(actor_group_width)
        loss_samples.append(step_losses)
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
    expert_atol: float | None = None,
    dense_max_ulps: int | None = None,
    phase: str,
    baseline: dict[str, torch.Tensor] | None = None,
) -> float:
    def bfloat16_spacing(tensor: torch.Tensor) -> torch.Tensor:
        as_float = tensor.float()
        negative_inf = torch.full_like(tensor, float("-inf"))
        positive_inf = torch.full_like(tensor, float("inf"))
        return torch.maximum(
            (torch.nextafter(tensor, negative_inf).float() - as_float).abs(),
            (torch.nextafter(tensor, positive_inf).float() - as_float).abs(),
        )

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
        parameter_atol = (
            expert_atol
            if expert_atol is not None and ".moe.experts." in name
            else atol
        )
        is_expert = ".moe.experts." in name
        if not torch.isfinite(actual_tensor).all() or not torch.isfinite(
            expected_tensor
        ).all():
            mismatches.append(
                (float("inf"), name, f"{name}: contains non-finite values")
            )
        elif (
            dense_max_ulps is not None
            and not is_expert
            and actual_tensor.dtype == torch.bfloat16
            and expected_tensor.dtype == torch.bfloat16
        ):
            abs_errors = (
                actual_tensor.float() - expected_tensor.float()
            ).abs()
            reference_spacing = bfloat16_spacing(expected_tensor)
            if baseline is not None:
                reference_spacing = torch.maximum(
                    reference_spacing,
                    bfloat16_spacing(baseline[name]),
                )
            allowed_error = torch.maximum(
                reference_spacing * dense_max_ulps,
                torch.full_like(reference_spacing, atol),
            )
            if bool((abs_errors <= allowed_error).all()):
                continue
            equivalent_ulps = torch.where(
                reference_spacing > 0,
                abs_errors / reference_spacing,
                torch.full_like(reference_spacing, float("inf")),
            )
            worst_idx = int(equivalent_ulps.reshape(-1).argmax().item())
            max_equivalent_ulps = float(
                equivalent_ulps.reshape(-1)[worst_idx]
            )
            actual_value = float(actual_tensor.reshape(-1)[worst_idx])
            expected_value = float(expected_tensor.reshape(-1)[worst_idx])
            worst_abs_error = abs(actual_value - expected_value)
            detail = (
                f"{name}[{worst_idx}]: reference_ulps={max_equivalent_ulps:g}, "
                f"abs_error={worst_abs_error:g}, max_abs_error={abs_error:g}, "
                f"piper={actual_value:g}, eager={expected_value:g}, "
                f"dense_max_ulps={dense_max_ulps}"
            )
            if baseline is not None:
                initial_value = float(baseline[name].reshape(-1)[worst_idx])
                detail += (
                    f", initial={initial_value:g}, "
                    f"piper_delta={actual_value - initial_value:g}, "
                    f"eager_delta={expected_value - initial_value:g}"
                )
            mismatches.append((abs_error, name, detail))
        elif not torch.allclose(
            actual_tensor,
            expected_tensor,
            rtol=rtol,
            atol=parameter_atol,
            equal_nan=False,
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
            f"(rtol={rtol}, atol={atol}, expert_atol={expert_atol}, "
            f"dense_max_ulps={dense_max_ulps}); "
            f"{details}; "
            f"mismatched_names={mismatch_names}"
        )
    return max_abs_error


def _derive_num_stages(schedule_directives_file: str) -> int:
    directives = load_schedule_directives(schedule_directives_file)
    pp_indices = {
        int(directive["filter"]["PP"])
        for directive in directives
        if isinstance(directive, dict)
        and directive.get("op") == "place"
        and isinstance(directive.get("filter"), dict)
        and isinstance(directive["filter"].get("PP"), int)
    }
    if not pp_indices:
        raise ValueError(
            "schedule directives file must contain a place directive with an "
            "exact PP filter: "
            f"{schedule_directives_file}"
        )
    if min(pp_indices) != 0:
        raise ValueError(
            f"schedule PP indices must start at 0, got {sorted(pp_indices)}"
        )
    return max(pp_indices) + 1


def _model_class_for_schedule(schedule_directives_file: str):
    """Select the AFD wrapper when placement addresses ATTN/FFN regions."""
    directives = load_schedule_directives(schedule_directives_file)
    uses_afd_regions = any(
        directive.get("op") == "place"
        and isinstance(directive.get("filter"), dict)
        and (
            "ATTN" in directive["filter"]
            or "FFN" in directive["filter"]
        )
        for directive in directives
    )
    return AFDQwen3Model if uses_afd_regions else PiperQwen3Model


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
    parser.add_argument("--expert-atol", type=float, default=None)
    parser.add_argument("--dense-max-ulps", type=int, default=None)
    parser.add_argument(
        "--optimizer", choices=("adam", "sgd"), default="adam"
    )
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
    if (
        args.rtol < 0
        or args.atol < 0
        or (args.expert_atol is not None and args.expert_atol < 0)
        or (args.dense_max_ulps is not None and args.dense_max_ulps < 0)
    ):
        parser.error("correctness tolerances must be non-negative")
    return args
