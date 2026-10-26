"""Regression tests for masked PPO overflow and empty action masks."""

import pytest
import torch

from orz.ppo.actors import PolicyLoss
from orz.ppo.utils import masked_mean


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("invalid_padding", ["overflow", "negative_infinity"])
def test_policy_loss_ignores_nonfinite_padding(dtype, invalid_padding):
    old = torch.tensor([[-1001.0, -1.0]], dtype=dtype)
    if invalid_padding == "negative_infinity":
        old[0, 0] = float("-inf")
        new = torch.tensor([[float("-inf"), -1.0]], dtype=dtype, requires_grad=True)
    else:
        new = torch.tensor([[-1.0, -1.0]], dtype=dtype, requires_grad=True)

    advantage = torch.ones_like(old)
    mask = torch.tensor([[0, 1]], dtype=torch.bool)
    loss = PolicyLoss()(new, old, advantage, mask)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(-1.0, dtype=dtype))
    torch.testing.assert_close(new.grad, torch.tensor([[0.0, -1.0]], dtype=dtype))
    assert torch.isfinite(new.grad).all()


def test_empty_action_mask_has_zero_loss_and_zero_gradients():
    log_probs = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    old_log_probs = log_probs.detach().clone()
    mask = torch.zeros_like(log_probs, dtype=torch.bool)
    loss = PolicyLoss()(log_probs, old_log_probs, torch.ones_like(log_probs), mask)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs))


def test_masked_mean_keeps_finite_unmasked_values():
    values = torch.tensor([[9.0, 2.0, 4.0]])
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(masked_mean(values, mask, dim=-1), torch.tensor([3.0]))
