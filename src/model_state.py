# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Helpers for moving model parameters between Dynamo graphs and eager models."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

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


def initialize_route_parameter_(
    parameter: torch.Tensor,
    parameter_name: str,
    *,
    global_route_start: int = 0,
    model_seed: int = DEFAULT_MODEL_SEED,
) -> torch.Tensor:
    """Initialize complete route slices independently and reproducibly.

    Independent per-route seeds let a rank initialize only its contiguous
    local experts while producing exactly the corresponding slice of an eager
    global parameter, without transiently materializing that global tensor.
    """
    if parameter.ndim < 1 or parameter.shape[0] <= 0:
        raise ValueError(
            "route-owned parameters require a nonempty leading route axis, "
            f"got shape={tuple(parameter.shape)}"
        )
    for local_route in range(int(parameter.shape[0])):
        global_route = int(global_route_start) + local_route
        initialize_parameter_(
            parameter[local_route],
            f"{parameter_name}[route={global_route}]",
            model_seed,
        )
    return parameter


def initialize_model_parameters_(
    model: torch.nn.Module,
    model_seed: int = DEFAULT_MODEL_SEED,
    *,
    route_parameter_axes: Mapping[str, int] | None = None,
) -> None:
    """Initialize an eager model exactly like Piper actors initialize tensors."""
    route_parameter_axes = dict(route_parameter_axes or {})
    named_parameters = dict(model.named_parameters())
    unknown = sorted(set(route_parameter_axes) - set(named_parameters))
    if unknown:
        raise ValueError(
            f"route_parameter_axes names are not model parameters: {unknown}"
        )
    unsupported = {
        name: axis for name, axis in route_parameter_axes.items() if axis != 0
    }
    if unsupported:
        raise ValueError(
            "only leading route-axis initialization is currently supported, "
            f"got {unsupported}"
        )
    with torch.no_grad():
        for name, parameter in named_parameters.items():
            if name in route_parameter_axes:
                initialize_route_parameter_(
                    parameter,
                    name,
                    model_seed=model_seed,
                )
            else:
                initialize_parameter_(parameter, name, model_seed)
