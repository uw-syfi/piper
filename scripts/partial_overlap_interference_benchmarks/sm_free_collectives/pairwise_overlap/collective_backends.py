"""Fail-closed collective backend selection for the overlap benchmark.

This module deliberately has no torch import.  The driver can therefore
validate and fingerprint an SM-aware configuration without creating a CUDA
context.  Backend modules are imported only inside benchmark workers.

An alternative backend selector has the form ``python.module:factory``.  The
factory is called once for each participating rank and operation as::

    operation = factory(context=context, options=options)

``context`` contains the exact tensors, split sizes, ranks, and process group
created by the normal benchmark workload.  A factory must return a mapping
with a nonblocking ``launch`` callable.  Optional fields are ``wait(result)``,
``close()``, ``owned`` (objects whose lifetime spans the case), ``metadata``,
and ``timing_domain`` (``"cuda_event"`` or ``"host"``).  A host-domain launch
must return promptly and defer completion to ``wait`` so both participants can
be dispatched before either is waited.

Factories may expose an ``allocate(context=..., options=...)`` attribute when
their transport requires a particular allocator (for example NCCL symmetric
window memory).  Backend modules may also expose ``worker_environment`` and
``prepare_worker_process`` hooks.  The former constructs Ray worker variables;
the latter runs before Torch is imported so a version-pinned communication
library can be loaded fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Mapping


COLLECTIVE_MODES = ("nccl", "sm-aware")
SM_AWARE_FAMILIES = ("all_gather", "balanced_a2a")
CATEGORY_TO_FAMILY = {
    "ag": "all_gather",
    "a2a": "balanced_a2a",
}
TIMING_DOMAINS = ("cuda_event", "host")
ZERO_CTA_ENVIRONMENT_NAMES = (
    "CUDA_COMPAT_PATH",
    "EXPECTED_NCCL_VERSION",
    "LD_LIBRARY_PATH",
    "NCCL4PY_OVERLAY",
    "PIPER_ZERO_CTA_OVERLAY",
    "PYTHONPATH",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BackendSpec:
    family: str
    selector: str
    module: str
    factory: str
    options: dict[str, Any]
    module_origin: str
    module_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "selector": self.selector,
            "module": self.module,
            "factory": self.factory,
            "options": self.options,
            "module_origin": self.module_origin,
            "module_sha256": self.module_sha256,
        }


@dataclass(frozen=True)
class CollectiveBackendSelection:
    mode: str
    backends: dict[str, BackendSpec]

    @property
    def uses_alternatives(self) -> bool:
        return self.mode == "sm-aware"

    def for_category(self, category: str) -> BackendSpec | None:
        family = CATEGORY_TO_FAMILY.get(category)
        return None if family is None else self.backends.get(family)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "normal_backend": "torch.distributed NCCL",
            "experiment_categories": ["compute", "ag", "a2a"],
            "backends": {
                family: spec.to_dict()
                for family, spec in sorted(self.backends.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CollectiveBackendSelection":
        if int(value.get("schema_version", -1)) != 1:
            raise ValueError("collective backend selection requires schema_version 1")
        mode = str(value.get("mode"))
        raw_backends = value.get("backends", {})
        if not isinstance(raw_backends, Mapping):
            raise ValueError("collective backend selection backends must be a mapping")
        specs: dict[str, BackendSpec] = {}
        for family, raw in raw_backends.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"backend {family!r} must be a mapping")
            specs[str(family)] = BackendSpec(
                family=str(raw["family"]),
                selector=str(raw["selector"]),
                module=str(raw["module"]),
                factory=str(raw["factory"]),
                options=dict(raw.get("options", {})),
                module_origin=str(raw["module_origin"]),
                module_sha256=(
                    None
                    if raw.get("module_sha256") is None
                    else str(raw["module_sha256"])
                ),
            )
        selection = cls(mode=mode, backends=specs)
        _validate_selection_shape(selection)
        return selection


def _parse_options(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {family: {} for family in SM_AWARE_FAMILIES}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("--sm-aware-options must contain a JSON object")
    unknown = sorted(set(document) - set(SM_AWARE_FAMILIES))
    if unknown:
        raise ValueError(f"unknown SM-aware option families: {unknown}")
    result: dict[str, dict[str, Any]] = {}
    for family in SM_AWARE_FAMILIES:
        options = document.get(family, {})
        if not isinstance(options, dict):
            raise ValueError(f"options for {family} must be a JSON object")
        result[family] = dict(options)
    return result


def _spec(family: str, selector: str, options: dict[str, Any]) -> BackendSpec:
    module, separator, factory = selector.partition(":")
    if not separator or not module or not factory or ":" in factory:
        raise ValueError(
            f"{family} backend must use python.module:factory syntax; got {selector!r}"
        )
    found = importlib.util.find_spec(module)
    if found is None:
        raise ValueError(f"cannot find backend module {module!r} for {family}")
    origin = found.origin or "namespace-or-built-in"
    source_path = Path(origin)
    source_hash = _sha256(source_path) if source_path.is_file() else None
    return BackendSpec(
        family=family,
        selector=selector,
        module=module,
        factory=factory,
        options=dict(options),
        module_origin=origin,
        module_sha256=source_hash,
    )


def _validate_selection_shape(selection: CollectiveBackendSelection) -> None:
    if selection.mode not in COLLECTIVE_MODES:
        raise ValueError(
            f"collective mode must be one of {COLLECTIVE_MODES}; got {selection.mode!r}"
        )
    observed = set(selection.backends)
    expected = set(SM_AWARE_FAMILIES) if selection.mode == "sm-aware" else set()
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{selection.mode} collective selection is incomplete: "
            f"missing={missing}, extra={extra}"
        )
    for family, spec in selection.backends.items():
        if spec.family != family:
            raise ValueError(
                f"backend dictionary key {family!r} disagrees with {spec.family!r}"
            )


def build_selection(
    *,
    mode: str,
    all_gather: str | None = None,
    a2a: str | None = None,
    options_path: Path | None = None,
) -> CollectiveBackendSelection:
    selectors = {
        "all_gather": all_gather,
        "balanced_a2a": a2a,
    }
    if mode not in COLLECTIVE_MODES:
        raise ValueError(f"collective mode must be one of {COLLECTIVE_MODES}")
    if mode == "nccl":
        set_values = sorted(family for family, value in selectors.items() if value)
        if set_values or options_path is not None:
            raise ValueError(
                "SM-aware backend selectors/options are invalid in nccl mode: "
                f"{set_values}"
            )
        return CollectiveBackendSelection(mode=mode, backends={})
    missing = sorted(family for family, value in selectors.items() if not value)
    if missing:
        raise ValueError(
            "sm-aware mode requires an explicit backend for every replaceable "
            f"collective; missing={missing}"
        )
    options = _parse_options(options_path)
    selection = CollectiveBackendSelection(
        mode=mode,
        backends={
            family: _spec(family, str(selectors[family]), options[family])
            for family in SM_AWARE_FAMILIES
        },
    )
    _validate_selection_shape(selection)
    return selection


def _resolve_module(spec: BackendSpec) -> Any:
    found = importlib.util.find_spec(spec.module)
    if found is None:
        raise ValueError(f"backend module {spec.module!r} disappeared on the worker")
    worker_origin = found.origin or "namespace-or-built-in"
    worker_path = Path(worker_origin)
    worker_hash = _sha256(worker_path) if worker_path.is_file() else None
    if worker_origin != spec.module_origin or worker_hash != spec.module_sha256:
        raise RuntimeError(
            f"backend module provenance changed for {spec.selector!r}: "
            f"driver=({spec.module_origin}, {spec.module_sha256}), "
            f"worker=({worker_origin}, {worker_hash})"
        )
    return importlib.import_module(spec.module)


def resolve_factory(spec: BackendSpec) -> Callable[..., Mapping[str, Any]]:
    module = _resolve_module(spec)
    factory = getattr(module, spec.factory, None)
    if not callable(factory):
        raise ValueError(
            f"backend selector {spec.selector!r} does not resolve to a callable"
        )
    return factory


def collective_backend_worker_environment(
    selection: CollectiveBackendSelection,
    base_environment: Mapping[str, str],
) -> dict[str, str]:
    """Return worker variables requested by the selected backend modules.

    Hooks are evaluated on the driver without importing Torch.  Each hook sees
    the environment accumulated from earlier families, and conflicting values
    therefore remain explicit and deterministic.
    """

    environment = dict(base_environment)
    requested: dict[str, str] = {}
    for family in SM_AWARE_FAMILIES:
        spec = selection.backends.get(family)
        if spec is None:
            continue
        module = _resolve_module(spec)
        hook = getattr(module, "worker_environment", None)
        if hook is None:
            continue
        if not callable(hook):
            raise TypeError(f"backend module {spec.module!r} worker_environment is not callable")
        raw = hook(
            family=family,
            options=dict(spec.options),
            environment=dict(environment),
        )
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"backend module {spec.module!r} worker_environment must return a mapping"
            )
        for name, value in raw.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError("backend worker environment must contain string pairs")
            environment[name] = value
            requested[name] = value
    return requested


def configured_zero_cta_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Propagate an explicitly sourced Zero-CTA environment in either mode."""

    if not environment.get("PIPER_ZERO_CTA_OVERLAY"):
        return {}
    return {
        name: str(environment[name])
        for name in ZERO_CTA_ENVIRONMENT_NAMES
        if environment.get(name)
    }


def prepare_collective_backend_worker_process(
    selection: CollectiveBackendSelection,
) -> None:
    """Run selected-module setup before a worker imports Torch."""

    for family in SM_AWARE_FAMILIES:
        spec = selection.backends.get(family)
        if spec is None:
            continue
        module = _resolve_module(spec)
        hook = getattr(module, "prepare_worker_process", None)
        if hook is None:
            continue
        if not callable(hook):
            raise TypeError(
                f"backend module {spec.module!r} prepare_worker_process is not callable"
            )
        hook(family=family, options=dict(spec.options))


def create_plugin_buffers(
    spec: BackendSpec,
    context: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Let a plugin allocate transport-specific source/output tensors."""

    factory = resolve_factory(spec)
    allocator = getattr(factory, "allocate", None)
    if allocator is None:
        return None
    if not callable(allocator):
        raise TypeError(f"backend {spec.selector!r} allocate attribute is not callable")
    raw = allocator(context=dict(context), options=dict(spec.options))
    if not isinstance(raw, Mapping):
        raise TypeError(f"backend {spec.selector!r} allocator must return a mapping")
    allocation = dict(raw)
    missing = [name for name in ("source", "output") if name not in allocation]
    if missing:
        raise ValueError(f"backend {spec.selector!r} allocator is missing {missing}")
    owned = allocation.get("owned", [])
    metadata = allocation.get("metadata", {})
    if not isinstance(owned, list):
        raise ValueError(f"backend {spec.selector!r} allocator owned must be a list")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"backend {spec.selector!r} allocator metadata must be a mapping")
    allocation["owned"] = owned
    allocation["metadata"] = dict(metadata)
    return allocation


def create_plugin_operation(
    spec: BackendSpec,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    factory = resolve_factory(spec)
    raw = factory(context=dict(context), options=dict(spec.options))
    if not isinstance(raw, Mapping):
        raise TypeError(f"backend {spec.selector!r} factory must return a mapping")
    operation = dict(raw)
    if not callable(operation.get("launch")):
        raise ValueError(f"backend {spec.selector!r} operation lacks launch()")
    timing_domain = str(operation.get("timing_domain", "cuda_event"))
    if timing_domain not in TIMING_DOMAINS:
        raise ValueError(
            f"backend {spec.selector!r} timing_domain must be one of {TIMING_DOMAINS}"
        )
    if timing_domain == "host" and not callable(operation.get("wait")):
        raise ValueError(
            f"host-timed backend {spec.selector!r} must provide wait(result)"
        )
    for name in ("wait", "close"):
        if name in operation and operation[name] is not None and not callable(operation[name]):
            raise ValueError(f"backend {spec.selector!r} field {name} must be callable")
    owned = operation.get("owned", [])
    if not isinstance(owned, list):
        raise ValueError(f"backend {spec.selector!r} field owned must be a list")
    metadata = operation.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError(f"backend {spec.selector!r} metadata must be a mapping")
    operation["timing_domain"] = timing_domain
    operation["owned"] = owned
    operation["metadata"] = {
        **dict(metadata),
        "collective_backend_family": spec.family,
        "collective_backend_selector": spec.selector,
        "collective_backend_module_sha256": spec.module_sha256,
    }
    return operation


def wait_for_result(result: Any) -> None:
    """Wait for torch Work objects while accepting stream-ordered results."""

    values = result if isinstance(result, (list, tuple)) else [result]
    for value in values:
        waiter = getattr(value, "wait", None)
        if callable(waiter):
            waiter()
