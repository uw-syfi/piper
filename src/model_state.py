"""Helpers for moving model parameters between Dynamo graphs and eager models."""

from __future__ import annotations

import hashlib

import torch


_MODULE_PREFIX = "l_self_modules_"
_ROOT_PARAMETER_PREFIX = "l_self_parameters_"
_ROOT_BUFFER_PREFIX = "l_self_buffers_"
_PARAMETER_SEPARATOR = "_parameters_"
_BUFFER_SEPARATOR = "_buffers_"
_MODULE_SEPARATOR = "_modules_"
DEFAULT_MODEL_SEED = 42


def dynamo_parameter_name_to_state_dict_key(name: str) -> str | None:
    """Translate a lifted Dynamo parameter placeholder to a ``state_dict`` key.

    Dynamo encodes module traversal in names such as
    ``l_self_modules_layers_modules_0_..._parameters_weight_``. Piper retains
    those placeholder names when it splits a model into stage and optimizer
    buckets, so this translation provides a stable bridge back to eager model
    parameter names.
    """

    if name.startswith(_MODULE_PREFIX):
        encoded = name[len(_MODULE_PREFIX) :].removesuffix("_")
        if _PARAMETER_SEPARATOR not in encoded:
            return None
        module_path, parameter_name = encoded.rsplit(_PARAMETER_SEPARATOR, 1)
        if not module_path or not parameter_name:
            return None
        return (
            module_path.replace(_MODULE_SEPARATOR, ".")
            + "."
            + parameter_name
        )

    if name.startswith(_ROOT_PARAMETER_PREFIX):
        parameter_name = name[len(_ROOT_PARAMETER_PREFIX) :].removesuffix("_")
        return parameter_name or None

    return None


def dynamo_buffer_name_to_state_dict_key(name: str) -> str | None:
    """Translate a lifted Dynamo buffer placeholder to a module buffer key."""
    if name.startswith(_MODULE_PREFIX):
        encoded = name[len(_MODULE_PREFIX) :].removesuffix("_")
        if _BUFFER_SEPARATOR not in encoded:
            return None
        module_path, buffer_name = encoded.rsplit(_BUFFER_SEPARATOR, 1)
        if not module_path or not buffer_name:
            return None
        return module_path.replace(_MODULE_SEPARATOR, ".") + "." + buffer_name

    if name.startswith(_ROOT_BUFFER_PREFIX):
        buffer_name = name[len(_ROOT_BUFFER_PREFIX) :].removesuffix("_")
        return buffer_name or None

    return None


def parameter_seed(model_seed: int, parameter_name: str) -> int:
    """Derive a stable per-parameter seed from one shared model seed."""
    payload = f"{int(model_seed)}:{parameter_name}".encode("utf-8")
    digest = hashlib.blake2b(
        payload, digest_size=8, person=b"piper-init"
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def initialize_parameter_(
    parameter: torch.Tensor,
    parameter_name: str,
    model_seed: int = DEFAULT_MODEL_SEED,
) -> torch.Tensor:
    """Apply Piper's deterministic normal initialization to one parameter."""
    generator = torch.Generator(device=parameter.device)
    generator.manual_seed(parameter_seed(model_seed, parameter_name))
    return torch.nn.init.normal_(
        parameter,
        mean=0.0,
        std=0.02,
        generator=generator,
    )


def initialize_model_parameters_(
    model: torch.nn.Module,
    model_seed: int = DEFAULT_MODEL_SEED,
) -> None:
    """Initialize an eager model exactly like Piper actors initialize tensors."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            initialize_parameter_(parameter, name, model_seed)
