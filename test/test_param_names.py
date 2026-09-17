"""Lock the one place that knows Dynamo's lifted-attribute spelling.

These strings were read off a real traced graph; if Dynamo renames lifted
attributes, param_overrides silently stops matching and _load_stage falls back
to random initialization, which is exactly the failure mode the override path
exists to avoid.
"""
from src.compile import dynamo_param_placeholder_name as name


def test_top_level_parameter() -> None:
    assert name("pre.weight") == "l_self_modules_pre_parameters_weight_"


def test_nested_parameter() -> None:
    assert name("layers.0.attention.wq.weight") == (
        "l_self_modules_layers_modules_0_modules_attention_modules_wq_parameters_weight_"
    )


def test_bias_and_other_attrs() -> None:
    assert name("fc.bias") == "l_self_modules_fc_parameters_bias_"
