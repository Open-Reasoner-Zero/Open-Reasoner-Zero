import pytest
import torch

from orz.ppo.actors import PolicyLoss


@pytest.mark.parametrize("excluded_old", [-1.0, -1001.0])
def test_masked_log_ratio_does_not_change_policy_loss_or_gradient(excluded_old):
    current = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    old = torch.tensor([[-1.0, excluded_old]])
    mask = torch.tensor([[True, False]])
    loss = PolicyLoss()(current, old, -torch.ones_like(current), mask)
    (gradient,) = torch.autograd.grad(loss, current)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(gradient, torch.tensor([[1.0, 0.0]]))
