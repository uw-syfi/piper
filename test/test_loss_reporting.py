"""The loss buffer must be drained by value, not aliased then cleared."""
import torch

from src.executors import _drain_losses


def test_drain_losses_returns_values_and_empties_the_buffer() -> None:
    buf = [torch.tensor(1.5), torch.tensor(2.5), 3.5]

    losses = _drain_losses(buf)

    assert losses == [1.5, 2.5, 3.5]
    assert buf == []


def test_drain_losses_does_not_alias_the_buffer() -> None:
    """`losses = loss_buffer; loss_buffer.clear()` returned an empty list."""
    buf = [torch.tensor(0.25)]

    losses = _drain_losses(buf)
    buf.append(torch.tensor(9.0))

    assert losses == [0.25]
