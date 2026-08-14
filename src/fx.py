import torch
import torch.fx as fx
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import inspect
import importlib
import json
import operator

from .state import create_logger, LOG_LEVEL

logger = create_logger("fx", LOG_LEVEL)
PIPER_ANNOTATIONS_META_KEY = "piper_annotations"
PIPER_EP_BOUNDARY_META_KEY = "piper_ep_boundary"
EP_DISPATCH_PAYLOAD = "dispatch_payload"
EP_DISPATCH_COUNTS = "dispatch_counts"
EP_COMBINE_PAYLOAD = "combine_payload"
_EP_BOUNDARY_ROLES = frozenset({
    EP_DISPATCH_PAYLOAD,
    EP_DISPATCH_COUNTS,
    EP_COMBINE_PAYLOAD,
})
_EP_BOUNDARY_APIS = {
    EP_DISPATCH_PAYLOAD: "mark_dispatch_payload",
    EP_DISPATCH_COUNTS: "mark_dispatch_counts",
    EP_COMBINE_PAYLOAD: "mark_combine_payload",
}


def _encode_arg(a):
    if isinstance(a, fx.Node):
        return {"__node__": a.name}
    if isinstance(a, torch.device):
        return {"__device__": str(a)}
    if isinstance(a, torch.dtype):
        return {"__dtype__": str(a).replace("torch.", "")}
    if isinstance(a, slice):
        return {
            "__slice__": True,
            "start": _encode_arg(a.start),
            "stop": _encode_arg(a.stop),
            "step": _encode_arg(a.step),
        }
    if a is Ellipsis:
        return {"__ellipsis__": True}
    if isinstance(a, tuple):
        return {"__tuple__": [_encode_arg(x) for x in a]}
    if isinstance(a, list):
        return [_encode_arg(x) for x in a]
    if isinstance(a, dict):
        return {k: _encode_arg(v) for k, v in a.items()}
    return a


def _decode_arg(a, name_to_node):
    if isinstance(a, dict):
        if "__node__" in a:
            return name_to_node[a["__node__"]]
        if "__device__" in a:
            return torch.device(a["__device__"])
        if "__dtype__" in a:
            return getattr(torch, a["__dtype__"])
        if "__slice__" in a:
            return slice(
                _decode_arg(a["start"], name_to_node),
                _decode_arg(a["stop"], name_to_node),
                _decode_arg(a["step"], name_to_node),
            )
        if "__ellipsis__" in a:
            return Ellipsis
        if "__tuple__" in a:
            return tuple(_decode_arg(x, name_to_node) for x in a["__tuple__"])
        return {k: _decode_arg(v, name_to_node) for k, v in a.items()}
    if isinstance(a, list):
        return [_decode_arg(x, name_to_node) for x in a]
    return a


def _is_op_overload(obj):
    return (
        obj.__class__.__module__.startswith("torch._ops")
        or obj.__class__.__name__.startswith("OpOverload")
    )


def _serialize_torch_op(path, target):
    """Serialize an operator and, for custom ops, its registration module.

    A custom ``torch.library`` operator does not exist in a fresh actor process
    until the Python module that defines it has been imported.  Carry that
    module generically instead of teaching Piper about individual model ops.
    """
    payload = {"kind": "torch_op", "path": path}
    try:
        from torch._library.custom_ops import _maybe_get_opdef

        opdef = _maybe_get_opdef(target)
    except (ImportError, AttributeError, TypeError):
        opdef = None
    init_fn = getattr(opdef, "_init_fn", None)
    registration_module = getattr(init_fn, "__module__", None)
    if registration_module:
        payload["registration_module"] = registration_module
    return payload


def _serialize_target(t):
    if isinstance(t, str):
        return {"kind": "string", "value": t}

    target_str = str(t)
    if target_str.startswith("torch.ops."):
        path = target_str[len("torch.ops."):]
        return _serialize_torch_op(path, t)

    if getattr(t, "__module__", "") == "torch._VariableFunctionsClass":
        public_name = t.__name__
        if hasattr(torch, public_name):
            return {"kind": "py_func", "module": "torch", "qualname": public_name}
        raise ValueError(f"No public torch alias for {t}")

    if _is_op_overload(t) or getattr(t, "__module__", "").startswith("torch._ops"):
        return _serialize_torch_op(str(t), t)

    if inspect.isfunction(t) or inspect.isbuiltin(t):
        mod = inspect.getmodule(t)

        if mod is not None:
            mod_name = mod.__name__
            if mod_name.startswith("torch.ops"):
                target_str = str(t)
                if target_str.startswith("torch.ops."):
                    path = target_str[len("torch.ops."):]
                    return _serialize_torch_op(path, t)
                module_parts = mod_name.split(".")
                if len(module_parts) >= 3:
                    namespace = module_parts[2]
                    path = f"{namespace}.{t.__name__}"
                    return _serialize_torch_op(path, t)

        if t.__name__ == "apply":
            qualname = getattr(t, "__qualname__", "")
            mod = inspect.getmodule(t)
            if "." in qualname:
                class_name = qualname.split(".")[0]
                if mod and hasattr(mod, class_name):
                    func_class = getattr(mod, class_name)
                    if isinstance(func_class, type) and issubclass(func_class, torch.autograd.Function):
                        return {
                            "kind": "py_obj",
                            "module": func_class.__module__,
                            "qualname": func_class.__qualname__ + ".apply",
                        }

                import sys

                for module_obj in sys.modules.values():
                    if module_obj is not None and hasattr(module_obj, class_name):
                        try:
                            func_class = getattr(module_obj, class_name)
                            if isinstance(func_class, type) and issubclass(func_class, torch.autograd.Function):
                                return {
                                    "kind": "py_obj",
                                    "module": func_class.__module__,
                                    "qualname": func_class.__qualname__ + ".apply",
                                }
                        except (TypeError, AttributeError):
                            continue

                if class_name.startswith("AllToAll"):
                    func_class = globals().get(class_name)
                    if isinstance(func_class, type) and issubclass(func_class, torch.autograd.Function):
                        return {
                            "kind": "py_obj",
                            "module": func_class.__module__,
                            "qualname": func_class.__qualname__ + ".apply",
                        }

        mod = inspect.getmodule(t)
        if mod is None:
            raise ValueError(f"Cannot serialize function without module: {t}")
        return {"kind": "py_func", "module": mod.__name__, "qualname": t.__name__}

    if inspect.isclass(t):
        mod = t.__module__
        return {"kind": "py_obj", "module": mod, "qualname": t.__qualname__}

    if t in operator.__dict__.values():
        return {"kind": "py_func", "module": "operator", "qualname": t.__name__}

    mod = getattr(t, "__module__", None)
    name = getattr(t, "__name__", None)
    if mod and name:
        return {"kind": "py_func", "module": mod, "qualname": name}

    raise NotImplementedError(f"Unsupported target type: {t} ({type(t)})")


def _resolve_qualname(mod, qualname):
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _deserialize_target(payload):
    kind = payload["kind"]

    if kind == "string":
        return payload["value"]

    if kind == "py_func":
        if payload["module"].startswith("torch.ops"):
            module_parts = payload["module"].split(".")
            if len(module_parts) >= 3:
                namespace = module_parts[2]
                path = f"{namespace}.{payload['qualname']}"
                return _deserialize_target({"kind": "torch_op", "path": path})
        mod = importlib.import_module(payload["module"])
        return _resolve_qualname(mod, payload["qualname"])

    if kind == "py_obj":
        mod = importlib.import_module(payload["module"])
        try:
            return _resolve_qualname(mod, payload["qualname"])
        except AttributeError as e:
            raise AttributeError(
                f"Failed to resolve qualname '{payload['qualname']}' "
                f"in module '{payload['module']}': {e}"
            ) from e

    if kind == "torch_op":
        obj = torch.ops
        path = payload["path"]
        registration_module = payload.get("registration_module")
        if registration_module:
            importlib.import_module(registration_module)
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            if path.startswith("higher_order."):
                op_name = path.split(".", 1)[1]
                op_to_module = {
                    "triton_kernel_wrapper_mutation": "torch._higher_order_ops.triton_kernel_wrap",
                    "triton_kernel_wrapper_functional": "torch._higher_order_ops.triton_kernel_wrap",
                }
                if op_name in op_to_module:
                    try:
                        importlib.import_module(op_to_module[op_name])
                        obj = torch.ops
                        for part in path.split("."):
                            obj = getattr(obj, part)
                        return obj
                    except (ImportError, AttributeError):
                        pass
            raise AttributeError(
                f"Failed to resolve torch.ops path '{path}'. "
                f"Available attributes: {[x for x in dir(obj) if not x.startswith('_')][:20]}"
            )

    raise NotImplementedError(f"Unknown target kind: {kind}")


def _serialize_graphmodule(gm: fx.GraphModule) -> str:
    nodes = []
    for n in gm.graph.nodes:
        nodes.append({
            "name": n.name,
            "op": n.op,
            "target": (
                _serialize_target(n.target)
                if n.op in ("call_function", "call_method", "call_module", "get_attr")
                else None
            ),
            "args": _encode_arg(n.args),
            "kwargs": _encode_arg(n.kwargs),
        })

    submodules = {}
    for name, module in gm.named_children():
        if isinstance(module, fx.GraphModule):
            submodules[name] = _serialize_graphmodule(module)

    data = {
        "nodes": nodes,
        "state_dict": {k: v.detach().cpu().tolist() for k, v in gm.state_dict().items()},
        "param_devices": {k: str(v.device) for k, v in gm.state_dict().items()},
        "submodules": submodules,
    }
    serialized = json.dumps(data, ensure_ascii=False)

    del data
    del nodes

    return serialized


def _unwrap_output_arg(decoded):
    if isinstance(decoded, (tuple, list)) and len(decoded) == 1 and isinstance(decoded[0], (tuple, list)):
        return decoded[0]
    return decoded


def _deserialize_graphmodule(s: str) -> fx.GraphModule:
    data = json.loads(s)
    g = fx.Graph()
    name_to_node = {}

    for n in data["nodes"]:
        op = n["op"]
        if op == "placeholder":
            node = g.placeholder(n["name"])
        elif op == "output":
            decoded = _decode_arg(n["args"], name_to_node)
            node = g.output(_unwrap_output_arg(decoded))
        elif op == "call_function":
            target = _deserialize_target(n["target"])
            args = _decode_arg(n["args"], name_to_node)
            kwargs = _decode_arg(n["kwargs"], name_to_node)
            node = g.call_function(target, tuple(args), kwargs)
        elif op == "call_method":
            target = _deserialize_target(n["target"])
            args = _decode_arg(n["args"], name_to_node)
            kwargs = _decode_arg(n["kwargs"], name_to_node)
            node = g.call_method(target, tuple(args), kwargs)
        elif op == "call_module":
            target = _deserialize_target(n["target"])
            args = _decode_arg(n["args"], name_to_node)
            kwargs = _decode_arg(n["kwargs"], name_to_node)
            node = g.call_module(target, tuple(args), kwargs)
        elif op == "get_attr":
            target = _deserialize_target(n["target"])
            node = g.get_attr(target)
        else:
            raise NotImplementedError(f"op {op} not handled")
        name_to_node[n["name"]] = node

    root_module = torch.nn.Module()
    submodules = data.get("submodules", {})
    for module_name, serialized_submodule in submodules.items():
        submodule_gm = _deserialize_graphmodule(serialized_submodule)
        parts = module_name.split(".")
        if len(parts) == 1:
            root_module.add_module(module_name, submodule_gm)
        else:
            current = root_module
            for part in parts[:-1]:
                if not hasattr(current, part):
                    current.add_module(part, torch.nn.Module())
                current = getattr(current, part)
            current.add_module(parts[-1], submodule_gm)

    gm = fx.GraphModule(root_module, g)
    state = {k: torch.tensor(v) for k, v in data["state_dict"].items()}
    gm.load_state_dict(state, strict=False)
    return gm


@dataclass
class AnnotationSegment:
    segment_id: int
    stage_id: int
    tag: dict[str, int]
    gm: fx.GraphModule
    input_idxs: list[int]
    param_idxs: list[int]
    graphargs: list[Any]
    placeholders: list[fx.Node]
    a2a_boundary_after: dict[str, Any] | None = None


def _inject_piper_annotation(node: fx.Node, name: str, index: int, uid: int) -> None:
    custom = node.meta.setdefault("custom", {})
    annotation = {"name": name, "index": int(index), "uid": int(uid)}
    custom[PIPER_ANNOTATIONS_META_KEY] = (annotation,)
    custom["name"] = name
    custom["index"] = int(index)


def _meta_tensor_like(example, *, requires_grad: bool, as_parameter: bool):
    sym_int_type = getattr(torch, "SymInt", ())
    sym_float_type = getattr(torch, "SymFloat", ())
    sym_bool_type = getattr(torch, "SymBool", ())

    if isinstance(example, torch.Tensor):
        t = torch.empty(example.shape, dtype=example.dtype, device="meta")
        t.requires_grad_(requires_grad)
    elif isinstance(example, (sym_int_type, int)):
        # Symbolic/python integers do not have .shape/.dtype; represent them as
        # singleton meta tensors so downstream graph-arg handling remains uniform.
        t = torch.empty((1,), dtype=torch.int64, device="meta")
    elif isinstance(example, (sym_float_type, float)):
        t = torch.empty((1,), dtype=torch.float32, device="meta")
    elif isinstance(example, (sym_bool_type, bool)):
        t = torch.empty((1,), dtype=torch.bool, device="meta")
    else:
        # Fallback for unknown scalar-like values.
        t = torch.empty((1,), dtype=torch.float32, device="meta")

    if as_parameter:
        return torch.nn.Parameter(t, requires_grad=requires_grad)
    return t


def _iter_arg_nodes(arg: Any) -> list[fx.Node]:
    nodes: list[fx.Node] = []

    def _walk(v: Any) -> None:
        if isinstance(v, fx.Node):
            nodes.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                _walk(item)
        elif isinstance(v, dict):
            for item in v.values():
                _walk(item)

    _walk(arg)
    return nodes


def _example_value(node: fx.Node) -> Any:
    return node.meta.get("example_value", node.meta.get("val"))


def _is_symbolic_scalar(value: Any) -> bool:
    return isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool))


def _symbolic_expression_key(value: Any) -> tuple[type, Any] | None:
    if not _is_symbolic_scalar(value):
        return None
    expr = getattr(getattr(value, "node", None), "expr", None)
    if expr is None:
        return None
    # SymPy expressions are immutable/hashable. Structural equality identifies
    # two occurrences of the same captured ShapeEnv value without parsing
    # names or depending on a printed representation.
    return type(value), expr


def _placeholder_is_runtime_input(node: fx.Node) -> bool:
    """Return whether a Dynamo placeholder must be supplied for every run.

    Dynamo marks user inputs with ``grapharg`` metadata too, so that marker
    cannot distinguish lifted module state from runtime inputs.  Lifted
    parameters and buffers use the ``l_self_`` prefix; every other tensor
    placeholder is a user input, including inputs that require gradients.
    """
    ex = _example_value(node)
    if not isinstance(ex, torch.Tensor):
        return False
    if isinstance(ex, torch.nn.Parameter):
        return False
    if node.name.startswith("l_self_"):
        return False
    return True


def _grapharg_for_placeholder(node: fx.Node) -> Any:
    ex = _example_value(node)
    requires_grad = bool(getattr(ex, "requires_grad", False))
    as_parameter = isinstance(ex, torch.nn.Parameter) or requires_grad
    return _meta_tensor_like(ex, requires_grad=requires_grad, as_parameter=as_parameter)


def _grapharg_for_cross_value(node: fx.Node) -> Any:
    ex = _example_value(node)
    requires_grad = bool(getattr(ex, "requires_grad", False))
    return _meta_tensor_like(ex, requires_grad=requires_grad, as_parameter=False)


def _validate_and_get_annotation_stack(node: fx.Node) -> tuple[dict[str, int], ...]:
    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return ()

    if PIPER_ANNOTATIONS_META_KEY not in custom:
        return ()

    raw_stack = custom[PIPER_ANNOTATIONS_META_KEY]
    if not isinstance(raw_stack, (list, tuple)) or not raw_stack:
        raise ValueError(
            "Piper annotation metadata must be a non-empty list/tuple of "
            f"{{'name': str, 'index': int, 'uid': int}} entries. "
            f"Node {node.name!r} has {raw_stack!r}."
        )

    stack: list[dict[str, int]] = []
    for annotation in raw_stack:
        if not isinstance(annotation, dict):
            raise ValueError(
                f"Piper annotation entry on node {node.name!r} must be a dict, "
                f"got {annotation!r}."
            )
        if set(annotation) - {"name", "index", "uid"}:
            raise ValueError(
                "Piper annotation entries may only contain 'name', 'index', and 'uid'. "
                f"Node {node.name!r} has {annotation!r}."
            )
        name = annotation.get("name")
        index = annotation.get("index")
        uid = annotation.get("uid")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Piper annotation 'name' must be a non-empty string on node {node.name!r}: "
                f"{annotation!r}."
            )
        if not isinstance(index, int):
            raise ValueError(
                f"Piper annotation 'index' must be an int on node {node.name!r}: "
                f"{annotation!r}."
            )
        if not isinstance(uid, int):
            raise ValueError(
                f"Piper annotation 'uid' must be an int on node {node.name!r}: "
                f"{annotation!r}."
            )
        stack.append({"name": name, "index": int(index), "uid": int(uid)})
    return tuple(stack)


def _annotation_stack_key(stack: tuple[dict[str, int], ...]) -> tuple[tuple[str, int, int], ...]:
    return tuple((a["name"], int(a["index"]), int(a["uid"])) for a in stack)


def _tag_from_stack(stack: tuple[dict[str, int], ...]) -> dict[str, int]:
    tag: dict[str, int] = {}
    for annotation in stack:
        tag[annotation["name"]] = int(annotation["index"])
    return tag


def _routed_boundary_role(node: fx.Node) -> str | None:
    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return None
    role = custom.get(PIPER_EP_BOUNDARY_META_KEY)
    if role is None:
        return None
    if role not in _EP_BOUNDARY_ROLES:
        raise ValueError(
            f"FX node {node.name!r} has invalid Piper routed boundary role {role!r}"
        )
    if (
        node.op != "call_function"
        or node.target is not torch.ops.aten.alias.default
    ):
        raise ValueError(
            f"Piper routed boundary role {role!r} must annotate an aten.alias node; "
            f"got node {node.name!r} with op={node.op!r}, target={node.target!r}"
        )
    return role


def _require_routed_boundary_roles(
    nodes: list[fx.Node],
    required_roles: tuple[str, ...],
    *,
    from_tag: dict[str, int],
    to_tag: dict[str, int],
) -> dict[str, int]:
    marked: dict[str, list[tuple[int, str]]] = {}
    for idx, node in enumerate(nodes):
        role = _routed_boundary_role(node)
        if role is not None:
            marked.setdefault(role, []).append((idx, node.name))

    expected = set(required_roles)
    unexpected = {
        role: [name for _, name in entries]
        for role, entries in marked.items()
        if role not in expected
    }
    if unexpected:
        raise ValueError(
            f"Routed boundary {from_tag}->{to_tag} has marker roles in the wrong "
            f"location: {unexpected}"
        )

    indices: dict[str, int] = {}
    for role in required_roles:
        matches = marked.get(role, [])
        if len(matches) != 1:
            raise ValueError(
                f"Routed boundary {from_tag}->{to_tag} requires exactly one "
                f"{_EP_BOUNDARY_APIS[role]}(...) tensor, found {len(matches)}"
            )
        indices[role] = matches[0][0]
    return indices


def _validate_routed_marker_tensor(node: fx.Node, role: str) -> torch.Tensor:
    """Validate the placement-independent tensor contract of a route marker."""
    value = _example_value(node)
    api_name = _EP_BOUNDARY_APIS[role]
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"{api_name}(...) on FX node {node.name!r} is missing tensor "
            "example metadata; routed marker schemas must be known while "
            "building the annotated graph"
        )

    if role == EP_DISPATCH_COUNTS:
        if value.ndim != 2:
            raise ValueError(
                f"{api_name}(...) requires a 2-D "
                "[source_blocks, local_routes] count tensor, got "
                f"shape={tuple(value.shape)} on FX node {node.name!r}"
            )
        if (
            value.dtype == torch.bool
            or value.dtype.is_floating_point
            or value.dtype.is_complex
        ):
            raise ValueError(
                f"{api_name}(...) requires an integer count tensor, got "
                f"dtype={value.dtype} on FX node {node.name!r}"
            )
        if value.requires_grad:
            raise ValueError(
                f"{api_name}(...) requires non-differentiable route counts, "
                f"but FX node {node.name!r} has requires_grad=True"
            )
        shape = _concrete_tensor_shape(value)
        if shape is None or shape[1] <= 0:
            raise ValueError(
                f"{api_name}(...) requires a statically known, positive route-axis "
                f"width, got shape={tuple(value.shape)} on FX node {node.name!r}"
            )
        # Dynamo normally records FakeTensor metadata, which intentionally has
        # no data. Shape propagation and hand-built FX graphs may retain a
        # concrete tensor; reject negative counts as early as possible there.
        if value.device.type != "meta" and not hasattr(value, "fake_mode"):
            try:
                has_negative = bool(torch.any(value < 0).item())
            except (RuntimeError, TypeError):
                has_negative = False
            if has_negative:
                raise ValueError(
                    f"{api_name}(...) requires nonnegative route counts on FX "
                    f"node {node.name!r}"
                )
    else:
        if value.ndim != 2:
            raise ValueError(
                f"{api_name}(...) requires a 2-D [tokens, hidden] payload "
                f"tensor, got shape={tuple(value.shape)} on FX node "
                f"{node.name!r}"
            )
    return value


def _concrete_tensor_shape(value: torch.Tensor) -> tuple[int, ...] | None:
    """Return a shape only when comparing it cannot create symbolic guards."""
    shape = tuple(value.shape)
    if not all(type(dim) is int for dim in shape):
        return None
    return tuple(int(dim) for dim in shape)


def _routed_tensor_schema(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": _concrete_tensor_shape(value),
        "dtype": str(value.dtype),
        "requires_grad": bool(value.requires_grad),
    }


def _is_fx_ancestor(ancestor: fx.Node, descendant: fx.Node) -> bool:
    pending = list(descendant.all_input_nodes)
    seen: set[fx.Node] = set()
    while pending:
        node = pending.pop()
        if node is ancestor:
            return True
        if node in seen:
            continue
        seen.add(node)
        pending.extend(node.all_input_nodes)
    return False


def _validate_routed_region_interfaces(
    *,
    nodes: list[fx.Node],
    seg_cross_in: list[list[fx.Node]],
    segments: list[AnnotationSegment],
    routed_segment_ids: set[int],
    node_seg: dict[fx.Node, int],
) -> None:
    """Validate the route-marker grammar before placement/DAG lowering.

    A routed region has one immediately adjacent dispatch boundary and one
    immediately adjacent combine boundary. Its payload/count ranks and graph
    dependency are fixed by the routed collective runtime.

    These constraints follow solely from FX annotation and marker placement,
    so reporting them here is both earlier and independent of schedule tags.
    Buffer-layout constraints used only by cross-group communication are
    deliberately validated later, once placement is known.
    """
    marker_boundaries: dict[fx.Node, list[int]] = defaultdict(list)
    for boundary_idx, crossing_nodes in enumerate(seg_cross_in[1:]):
        for node in crossing_nodes:
            if _routed_boundary_role(node) is not None:
                marker_boundaries[node].append(boundary_idx)

    marker_nodes = [node for node in nodes if _routed_boundary_role(node) is not None]
    for node in marker_nodes:
        boundary_ids = marker_boundaries.get(node, [])
        role = _routed_boundary_role(node)
        assert role is not None
        if len(boundary_ids) != 1:
            raise ValueError(
                f"{_EP_BOUNDARY_APIS[role]}(...) on FX node {node.name!r} must "
                "cross exactly one adjacent annotation boundary, but crosses "
                f"{len(boundary_ids)} boundaries"
            )
        _validate_routed_marker_tensor(node, role)

    for seg in sorted(routed_segment_ids):
        incoming = segments[seg - 1].a2a_boundary_after
        outgoing = segments[seg].a2a_boundary_after
        assert isinstance(incoming, dict) and isinstance(outgoing, dict)
        route = (
            f"{incoming.get('from_tag')}->{segments[seg].tag}->"
            f"{outgoing.get('to_tag')}"
        )
        inputs = seg_cross_in[seg]
        outputs = seg_cross_in[seg + 1]
        dispatch_idx = incoming.get("tensor_idx")
        counts_idx = incoming.get("routing_counts_idx")
        combine_idx = outgoing.get("tensor_idx")
        assert isinstance(dispatch_idx, int) and not isinstance(dispatch_idx, bool)
        assert isinstance(counts_idx, int) and not isinstance(counts_idx, bool)
        assert isinstance(combine_idx, int) and not isinstance(combine_idx, bool)
        dispatch_node = inputs[dispatch_idx]
        counts_node = inputs[counts_idx]
        combine_node = outputs[combine_idx]
        dispatch_value = _validate_routed_marker_tensor(
            dispatch_node, EP_DISPATCH_PAYLOAD
        )
        counts_value = _validate_routed_marker_tensor(
            counts_node, EP_DISPATCH_COUNTS
        )
        combine_value = _validate_routed_marker_tensor(
            combine_node, EP_COMBINE_PAYLOAD
        )

        incoming["payload_schema"] = _routed_tensor_schema(dispatch_value)
        incoming["routing_counts_schema"] = _routed_tensor_schema(counts_value)
        outgoing["payload_schema"] = _routed_tensor_schema(combine_value)
        if not _is_fx_ancestor(dispatch_node, combine_node):
            raise ValueError(
                f"Routed region {route} must compute its combine payload from "
                f"its dispatch payload; FX node {dispatch_node.name!r} is not "
                f"an ancestor of {combine_node.name!r}"
            )
        if not _is_fx_ancestor(counts_node, combine_node):
            raise ValueError(
                f"Routed region {route} must compute its combine payload from "
                f"its route counts; FX node {counts_node.name!r} is not an "
                f"ancestor of {combine_node.name!r}"
            )

        # Bypassing values may cross either physical boundary, but the routed
        # segment itself has a deliberately closed dynamic activation ABI.
        # Parameters, buffers, and constants are lifted separately and do not
        # appear in these crossing-value lists.
        consumed_inputs = {
            node
            for node in inputs
            if any(node_seg.get(user) == seg for user in node.users)
        }
        expected_inputs = {dispatch_node, counts_node}
        if consumed_inputs != expected_inputs:
            unexpected = sorted(
                node.name for node in consumed_inputs - expected_inputs
            )
            missing = sorted(node.name for node in expected_inputs - consumed_inputs)
            raise ValueError(
                f"Routed region {route} must have exactly "
                "mark_dispatch_payload(...) and mark_dispatch_counts(...) as "
                "its dynamic activation inputs; "
                f"unexpected={unexpected}, missing={missing}"
            )

        produced_outputs = {
            node for node in outputs if node_seg.get(node) == seg
        }
        expected_outputs = {combine_node}
        if produced_outputs != expected_outputs:
            unexpected = sorted(
                node.name for node in produced_outputs - expected_outputs
            )
            missing = sorted(node.name for node in expected_outputs - produced_outputs)
            raise ValueError(
                f"Routed region {route} must have exactly "
                "mark_combine_payload(...) as its dynamic activation output; "
                f"unexpected={unexpected}, missing={missing}"
            )

        dispatch_shape = _concrete_tensor_shape(dispatch_value)
        counts_shape = _concrete_tensor_shape(counts_value)
        combine_shape = _concrete_tensor_shape(combine_value)
        if (
            dispatch_shape is not None
            and combine_shape is not None
            and dispatch_shape != combine_shape
        ):
            raise ValueError(
                f"Routed region {route} must preserve the concrete payload "
                "shape across dispatch/combine, got "
                f"{dispatch_shape} and {combine_shape}"
            )
        if dispatch_value.dtype != combine_value.dtype:
            raise ValueError(
                f"Routed region {route} must preserve payload dtype across "
                f"dispatch/combine, got {dispatch_value.dtype} and "
                f"{combine_value.dtype}"
            )
        if dispatch_value.requires_grad and not combine_value.requires_grad:
            raise ValueError(
                f"Routed region {route} has a differentiable dispatch payload "
                "but a non-differentiable combine payload"
            )

        # Value-dependent checks are possible only for concrete FX metadata.
        if (
            counts_shape is not None
            and dispatch_shape is not None
            and counts_value.device.type != "meta"
            and not hasattr(counts_value, "fake_mode")
        ):
            try:
                count_sum = int(counts_value.sum().item())
            except (RuntimeError, TypeError):
                count_sum = None
            if count_sum is not None and count_sum != dispatch_shape[0]:
                raise ValueError(
                    f"Routed region {route} requires dispatch payload rows to "
                    "equal the sum of route counts, got "
                    f"rows={dispatch_shape[0]}, count_sum={count_sum}"
                )


def split_gm_by_annotations(gm: fx.GraphModule) -> tuple[fx.GraphModule, list[AnnotationSegment]]:
    """Split a GraphModule into contiguous subgraphs by Piper annotation stack.

    Piper annotations are carried in ``node.meta['custom']['piper_annotations']``.
    The full stack, including nested annotations, is part of the split key. The
    schedule-facing tag for each segment is the stack projected to
    ``{annotation_name: auto_index}``.
    """
    nodes = list(gm.graph.nodes)
    operation_nodes = [
        node for node in nodes
        if node.op not in ("placeholder", "get_attr", "output")
    ]

    node_stacks: dict[fx.Node, tuple[dict[str, int], ...]] = {}
    for node in nodes:
        stack = _validate_and_get_annotation_stack(node)
        if node in operation_nodes:
            node_stacks[node] = stack
            _routed_boundary_role(node)

    # Dynamo emits shape-only scalar helpers outside fx.traceback annotation
    # scopes when an input dimension is marked dynamic.  They are not model
    # computation and must not become an extra routed ABI input: each split
    # subgraph rematerializes the needed expression from one of its tensor
    # placeholders below.
    symbolic_shape_nodes = {
        node
        for node in operation_nodes
        if not node_stacks.get(node)
        and _is_symbolic_scalar(_example_value(node))
    }
    compute_nodes = [
        node for node in operation_nodes if node not in symbolic_shape_nodes
    ]

    annotated_compute = [node for node in compute_nodes if node_stacks.get(node)]
    if not annotated_compute:
        return gm, []

    unannotated = [node.name for node in compute_nodes if not node_stacks.get(node)]
    if unannotated:
        raise ValueError(
            "Found compute nodes without Piper annotations in a graph that uses "
            f"piper.annotate: {unannotated[:16]}. Wrap all model compute in "
            "piper.annotate(...)."
        )

    segment_keys: list[tuple[tuple[str, int, int], ...]] = []
    segment_stacks: list[tuple[dict[str, int], ...]] = []
    node_seg: dict[fx.Node, int] = {}
    current_key: tuple[tuple[str, int, int], ...] | None = None
    for node in compute_nodes:
        stack = node_stacks[node]
        key = _annotation_stack_key(stack)
        if key != current_key:
            segment_keys.append(key)
            segment_stacks.append(stack)
            current_key = key
        node_seg[node] = len(segment_keys) - 1

    n_segs = len(segment_stacks)
    if n_segs == 0:
        return gm, []

    runtime_placeholders = {
        node for node in nodes
        if node.op == "placeholder" and _placeholder_is_runtime_input(node)
    }
    grapharg_placeholders = {
        node for node in nodes
        if node.op == "placeholder" and node not in runtime_placeholders
    }

    value_seg: dict[fx.Node, int] = dict(node_seg)
    for node in runtime_placeholders:
        value_seg[node] = 0

    def _consumer_seg(user: fx.Node) -> int | None:
        if user.op == "output":
            return n_segs - 1
        return node_seg.get(user)

    node_max_user_seg: dict[fx.Node, int] = {}
    for node, seg in value_seg.items():
        user_segs = [
            user_seg for user in node.users
            for user_seg in [_consumer_seg(user)]
            if user_seg is not None
        ]
        node_max_user_seg[node] = max(user_segs) if user_segs else seg

    seg_cross_in: list[list[fx.Node]] = [[] for _ in range(n_segs)]
    for node in nodes:
        if node not in value_seg:
            continue
        if _is_symbolic_scalar(_example_value(node)):
            # Shape scalars are rematerialized from a tensor placeholder in
            # each consumer segment; they are not part of the dynamic routed
            # activation ABI.
            continue
        seg = value_seg[node]
        max_seg = node_max_user_seg[node]
        for target_seg in range(seg + 1, max_seg + 1):
            seg_cross_in[target_seg].append(node)

    segments: list[AnnotationSegment] = []
    output_node = next((node for node in nodes if node.op == "output"), None)

    for seg in range(n_segs):
        sub_g = fx.Graph()
        remap: dict[fx.Node, fx.Node] = {}
        new_input_idxs: list[int] = []
        new_param_idxs: list[int] = []
        new_graphargs: list[Any] = []
        pos = 0

        def _add_placeholder(node: fx.Node, name: str, *, is_runtime_input: bool) -> None:
            nonlocal pos
            if node in remap:
                return
            new_ph = sub_g.placeholder(name)
            new_ph.type = node.type
            new_ph.meta.update(node.meta)
            remap[node] = new_ph
            if is_runtime_input:
                new_input_idxs.append(pos)
                new_graphargs.append(_grapharg_for_cross_value(node))
            else:
                new_param_idxs.append(pos)
                new_graphargs.append(_grapharg_for_placeholder(node))
            pos += 1

        if seg == 0:
            for node in nodes:
                if node in runtime_placeholders:
                    _add_placeholder(node, node.name, is_runtime_input=True)
        else:
            for node in seg_cross_in[seg]:
                _add_placeholder(node, f"_xseg_{node.name}", is_runtime_input=True)

        seg_compute_nodes = [
            node for node in nodes
            if node.op not in ("placeholder", "get_attr", "output")
            and node_seg.get(node) == seg
        ]
        needed_grapharg_placeholders: set[fx.Node] = set()
        needed_getattrs: set[fx.Node] = set()
        for node in seg_compute_nodes:
            for dep in [*node.all_input_nodes, *_iter_arg_nodes(node.kwargs)]:
                if dep.op == "placeholder" and dep in grapharg_placeholders:
                    needed_grapharg_placeholders.add(dep)
                elif dep.op == "get_attr":
                    needed_getattrs.add(dep)

        if output_node is not None and seg == n_segs - 1:
            for dep in _iter_arg_nodes(output_node.args):
                if dep.op == "placeholder" and dep in grapharg_placeholders:
                    needed_grapharg_placeholders.add(dep)
                elif dep.op == "get_attr":
                    needed_getattrs.add(dep)

        for node in nodes:
            if node in needed_grapharg_placeholders:
                if _is_symbolic_scalar(_example_value(node)):
                    continue
                _add_placeholder(node, node.name, is_runtime_input=False)

        for node in nodes:
            if node.op == "get_attr" and node in needed_getattrs and node not in remap:
                new_ga = sub_g.get_attr(node.target)
                new_ga.type = node.type
                new_ga.meta.update(node.meta)
                remap[node] = new_ga

        def _rematerialize_symbolic(node: fx.Node) -> fx.Node:
            if node in remap:
                return remap[node]
            value_key = _symbolic_expression_key(_example_value(node))
            if value_key is not None:
                candidates = list(remap.items())
                candidates.sort(
                    key=lambda item: (
                        0
                        if _routed_boundary_role(item[0])
                        in {EP_DISPATCH_PAYLOAD, EP_COMBINE_PAYLOAD}
                        else 1
                    )
                )
                for original, copied in candidates:
                    example = _example_value(original)
                    if not isinstance(example, torch.Tensor):
                        continue
                    for dim, size in enumerate(example.shape):
                        if _symbolic_expression_key(size) == value_key:
                            size_node = sub_g.call_method("size", (copied, dim))
                            size_node.meta.update(node.meta)
                            remap[node] = size_node
                            return size_node

            if value_key is not None and node.op != "placeholder":
                for dep in node.all_input_nodes:
                    if dep not in remap:
                        _rematerialize_symbolic(dep)
                copied = sub_g.node_copy(node, arg_transform=lambda x: remap[x])
                remap[node] = copied
                return copied
            raise KeyError(node)

        def _ensure_dependency(node: fx.Node) -> None:
            if node in remap or node_seg.get(node) == seg:
                return
            if _is_symbolic_scalar(_example_value(node)):
                _rematerialize_symbolic(node)

        for node in seg_compute_nodes:
            for dep in [*node.all_input_nodes, *_iter_arg_nodes(node.kwargs)]:
                _ensure_dependency(dep)
            missing = [dep.name for dep in node.all_input_nodes if dep not in remap and node_seg.get(dep) != seg]
            if missing:
                raise ValueError(
                    f"Cannot split annotated graph at segment {seg}; node {node.name!r} "
                    f"has unmapped external dependencies {missing}."
                )
            remap[node] = sub_g.node_copy(node, arg_transform=lambda x, r=remap: r[x])

        if seg == n_segs - 1:
            if output_node is None:
                sub_g.output(())
            else:
                for dep in _iter_arg_nodes(output_node.args):
                    _ensure_dependency(dep)
                sub_g.output(fx.map_arg(output_node.args[0], lambda x: remap[x]))
            boundary_after = None
        else:
            out_nodes = [remap[node] for node in seg_cross_in[seg + 1]]
            sub_g.output(tuple(out_nodes) if len(out_nodes) != 1 else out_nodes[0])
            from_tag = _tag_from_stack(segment_stacks[seg])
            to_tag = _tag_from_stack(segment_stacks[seg + 1])
            boundary_after = {
                "reshape_input": None,
                "reshape_output": None,
                "from_tag": from_tag,
                "to_tag": to_tag,
            }
            crossing_nodes = seg_cross_in[seg + 1]
            marker_roles = {
                role
                for node in crossing_nodes
                for role in [_routed_boundary_role(node)]
                if role is not None
            }
            dispatch_roles = {
                EP_DISPATCH_PAYLOAD,
                EP_DISPATCH_COUNTS,
            }
            has_dispatch = bool(marker_roles & dispatch_roles)
            has_combine = EP_COMBINE_PAYLOAD in marker_roles
            if has_dispatch and has_combine:
                raise ValueError(
                    f"Routed boundary {from_tag}->{to_tag} mixes dispatch and "
                    f"combine markers: {sorted(marker_roles)}"
                )
            if has_dispatch:
                indices = _require_routed_boundary_roles(
                    crossing_nodes,
                    (EP_DISPATCH_PAYLOAD, EP_DISPATCH_COUNTS),
                    from_tag=from_tag,
                    to_tag=to_tag,
                )
                boundary_after["routed_boundary"] = "dispatch"
                boundary_after["tensor_idx"] = indices[EP_DISPATCH_PAYLOAD]
                boundary_after["routing_counts_idx"] = indices[
                    EP_DISPATCH_COUNTS
                ]
            elif has_combine:
                indices = _require_routed_boundary_roles(
                    crossing_nodes,
                    (EP_COMBINE_PAYLOAD,),
                    from_tag=from_tag,
                    to_tag=to_tag,
                )
                boundary_after["routed_boundary"] = "combine"
                boundary_after["tensor_idx"] = indices[EP_COMBINE_PAYLOAD]
            else:
                _require_routed_boundary_roles(
                    crossing_nodes,
                    (),
                    from_tag=from_tag,
                    to_tag=to_tag,
                )

        sub_g.lint()
        seg_gm = fx.GraphModule(gm, sub_g)
        placeholders = [node for node in seg_gm.graph.nodes if node.op == "placeholder"]
        tag = _tag_from_stack(segment_stacks[seg])
        segments.append(
            AnnotationSegment(
                segment_id=seg,
                stage_id=int(tag.get("PP", seg)),
                tag=tag,
                gm=seg_gm,
                input_idxs=new_input_idxs,
                param_idxs=new_param_idxs,
                graphargs=new_graphargs,
                placeholders=placeholders,
                a2a_boundary_after=boundary_after,
            )
        )

    dispatch_targets = {
        seg + 1
        for seg, segment in enumerate(segments[:-1])
        if isinstance(segment.a2a_boundary_after, dict)
        and segment.a2a_boundary_after.get("routed_boundary") == "dispatch"
    }
    combine_sources = {
        seg
        for seg, segment in enumerate(segments[:-1])
        if isinstance(segment.a2a_boundary_after, dict)
        and segment.a2a_boundary_after.get("routed_boundary") == "combine"
    }
    if dispatch_targets != combine_sources:
        raise ValueError(
            "Every routed region must have adjacent dispatch and combine "
            "boundaries marked by mark_dispatch_payload(...), "
            "mark_dispatch_counts(...), and mark_combine_payload(...); dispatch targets="
            f"{sorted(dispatch_targets)}, combine sources={sorted(combine_sources)}"
        )

    _validate_routed_region_interfaces(
        nodes=nodes,
        seg_cross_in=seg_cross_in,
        segments=segments,
        routed_segment_ids=dispatch_targets,
        node_seg=node_seg,
    )

    return gm, segments
