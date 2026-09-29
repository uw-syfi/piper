"""piper_setup refuses options it does not implement, before it starts any actor."""
import pytest

from src.compile import piper_setup


@pytest.mark.parametrize("kwargs", [{"activation_checkpointing": True}, {"num_checkpoints": 2}])
def test_activation_checkpointing_is_refused(kwargs) -> None:
    with pytest.raises(NotImplementedError, match="activation checkpointing"):
        piper_setup(object, schedule_directives_file="unused.json", **kwargs)
