"""Piper driver API.

``annotate`` tags model regions; ``compile`` lowers a schedule into a
``DagProgram``; ``setup`` builds a ``Runtime`` whose ``load`` and ``step``
drive training. Names resolve lazily so light modules import without Ray.
"""
import importlib

_EXPORTS = {
    "annotate": ".piper",
    "compile": ".compile",
    "DagProgram": ".compile",
    "setup": ".driver",
    "Runtime": ".driver",
    "StepResult": ".driver",
    "StepFailed": ".driver",
    "Topology": ".topology",
    "ActorConfig": ".config",
    "FTConfig": ".config",
}


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)
