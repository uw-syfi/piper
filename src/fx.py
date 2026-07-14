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


def _serialize_target(t):
    if isinstance(t, str):
        return {"kind": "string", "value": t}

    target_str = str(t)
    if target_str.startswith("torch.ops."):
        path = target_str[len("torch.ops."):]
        return {"kind": "torch_op", "path": path}

    if getattr(t, "__module__", "") == "torch._VariableFunctionsClass":
        public_name = t.__name__
        if hasattr(torch, public_name):
            return {"kind": "py_func", "module": "torch", "qualname": public_name}
        raise ValueError(f"No public torch alias for {t}")

    if _is_op_overload(t) or getattr(t, "__module__", "").startswith("torch._ops"):
        return {"kind": "torch_op", "path": str(t)}

    if inspect.isfunction(t) or inspect.isbuiltin(t):
        mod = inspect.getmodule(t)

        if mod is not None:
            mod_name = mod.__name__
            if mod_name.startswith("torch.ops"):
                target_str = str(t)
                if target_str.startswith("torch.ops."):
                    path = target_str[len("torch.ops."):]
                    return {"kind": "torch_op", "path": path}
                module_parts = mod_name.split(".")
                if len(module_parts) >= 3:
                    namespace = module_parts[2]
                    path = f"{namespace}.{t.__name__}"
                    return {"kind": "torch_op", "path": path}

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
        if path == "piper_artifact.bmm_experts":
            # The Qwen artifact model defines this model-specific custom op.
            for module_name in ("models.qwen3", "examples.models.qwen3"):
                try:
                    importlib.import_module(module_name)
                    break
                except ImportError:
                    pass
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


def _placeholder_is_runtime_input(node: fx.Node) -> bool:
    ex = _example_value(node)
    if not isinstance(ex, torch.Tensor):
        return False
    if isinstance(ex, torch.nn.Parameter):
        return False
    if bool(getattr(ex, "requires_grad", False)):
        return False
    if "self" in node.name or "grapharg" in node.meta:
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


def _select_boundary_tensor_idx(nodes: list[fx.Node]) -> int | None:
    best_idx = None
    best_score = -1
    for idx, node in enumerate(nodes):
        ex = _example_value(node)
        if not isinstance(ex, torch.Tensor):
            continue
        score = 1
        if ex.is_floating_point():
            score += 2
        if bool(getattr(ex, "requires_grad", False)):
            score += 1
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is not None:
        return best_idx
    return 0 if nodes else None


def split_gm_by_annotations(gm: fx.GraphModule) -> tuple[fx.GraphModule, list[AnnotationSegment]]:
    """Split a GraphModule into contiguous subgraphs by Piper annotation stack.

    Piper annotations are carried in ``node.meta['custom']['piper_annotations']``.
    The full stack, including nested annotations, is part of the split key. The
    schedule-facing tag for each segment is the stack projected to
    ``{annotation_name: auto_index}``.
    """
    nodes = list(gm.graph.nodes)
    compute_nodes = [
        node for node in nodes
        if node.op not in ("placeholder", "get_attr", "output")
    ]

    node_stacks: dict[fx.Node, tuple[dict[str, int], ...]] = {}
    for node in nodes:
        stack = _validate_and_get_annotation_stack(node)
        if node in compute_nodes:
            node_stacks[node] = stack

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
            and node_seg[node] == seg
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
                _add_placeholder(node, node.name, is_runtime_input=False)

        for node in nodes:
            if node.op == "get_attr" and node in needed_getattrs and node not in remap:
                new_ga = sub_g.get_attr(node.target)
                new_ga.type = node.type
                new_ga.meta.update(node.meta)
                remap[node] = new_ga

        for node in seg_compute_nodes:
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
                sub_g.output(fx.map_arg(output_node.args[0], lambda x: remap[x]))
            boundary_after = None
        else:
            out_nodes = [remap[node] for node in seg_cross_in[seg + 1]]
            sub_g.output(tuple(out_nodes) if len(out_nodes) != 1 else out_nodes[0])
            boundary_after = {
                "tensor_idx": _select_boundary_tensor_idx(seg_cross_in[seg + 1]),
                "reshape_input": None,
                "reshape_output": None,
                "from_tag": _tag_from_stack(segment_stacks[seg]),
                "to_tag": _tag_from_stack(segment_stacks[seg + 1]),
            }

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

    return gm, segments
