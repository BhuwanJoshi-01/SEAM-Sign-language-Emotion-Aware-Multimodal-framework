"""Gradient reversal, applied in both directions.

A gradient reversal layer is the standard way to make one branch *unlearnable* from
the other's representation, and it is used here in both directions because the
requirement is mutual: `z_A` must not contain what only `z_L` knows, and `z_L` must
not contain what only `z_A` knows. One direction leaves the other factor free to
smuggle the information through.

The implementation is the forward-identity / backward-negate form. Its cost per step
is a negation, which is not why it is written this way - it is written this way so
that the forward pass can never see the sign, because a forward pass that knows it
is reversing a gradient is a class of bug that produces a plausible model and a
silently wrong loss curve.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _GradientReversalFn(torch.autograd.Function):
    """Identity forward, negate the gradient backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Reverse gradients flowing back through ``x`` by ``lambd``."""
    return _GradientReversalFn.apply(x, lambd)


class GradientReversal(nn.Module):
    """Module form, so the reversal shows up in a printed architecture.

    ``lambd`` is a buffer rather than a forward argument so a training loop can ramp
    it with a scheduler instead of rebuilding the graph, and so the value that
    actually ran is recoverable from ``state_dict``.
    """

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("lambd", torch.tensor(float(lambd)))

    def lambda_value(self) -> float:
        """The reversal strength, read from the buffer."""
        buf = self._buffers["lambd"]
        return float(buf.item()) if buf is not None else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambda_value())

    def extra_repr(self) -> str:
        return f"lambda={self.lambda_value():.3f}"
