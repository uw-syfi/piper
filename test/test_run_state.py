"""Per-run state must not survive into a second piper_setup."""
from src.compile import _reset_run_state
from src.state import piper_metadata


def test_reset_clears_the_pushed_loss_function() -> None:
    """piper_exec_dag skips the push while the cached object is identical.

    A second setup builds new actors, which would then never receive it and hit a
    None callable at the loss node.
    """
    sentinel = lambda out, labels: out
    piper_metadata.installed_loss_fn = sentinel
    piper_metadata.training_dag = object()
    piper_metadata.per_pp_training_dags = [object()]

    _reset_run_state()

    assert piper_metadata.installed_loss_fn is None
    assert piper_metadata.training_dag is None
    assert piper_metadata.per_pp_training_dags is None
