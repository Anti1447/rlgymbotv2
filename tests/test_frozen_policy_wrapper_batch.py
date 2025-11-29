import numpy as np
import torch
import torch.nn as nn

from rlgymbotv2.training.frozen_policy_wrapper import FrozenPolicyWrapper


class DummyNet(nn.Module):
    def __init__(self, in_dim=10, out_dim=5):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def test_act_and_act_batch_shapes():
    net = DummyNet()
    wrapper = FrozenPolicyWrapper(net, device="cpu", profile=False)

    obs_single = np.zeros(10, dtype=np.float32)
    obs_batch = np.zeros((4, 10), dtype=np.float32)

    a1 = wrapper.act(obs_single)
    ab = wrapper.act_batch(obs_batch)

    assert isinstance(a1, int)
    assert ab.shape == (4,)
    assert ab.dtype == np.int64 or np.issubdtype(ab.dtype, np.integer)


def test_act_batch_consistency_with_act():
    net = DummyNet()
    wrapper = FrozenPolicyWrapper(net, device="cpu", profile=False)

    obs_batch = np.random.randn(4, 10).astype(np.float32)

    ab = wrapper.act_batch(obs_batch)
    a_list = [wrapper.act(obs_batch[i]) for i in range(obs_batch.shape[0])]

    assert np.allclose(ab, np.array(a_list))
