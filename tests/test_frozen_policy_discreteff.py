# rlgymbotv2/tests/test_frozen_policy_discreteff.py

import torch
import numpy as np

from rlgym_ppo.ppo.discrete_policy import DiscreteFF
from rlgymbotv2.training.frozen_policy_wrapper import FrozenPolicyWrapper

def test_frozen_policy_wrapper_with_discreteff():
    obs_dim = 10
    n_actions = 5
    device = torch.device("cpu")

    policy = DiscreteFF(
        input_shape=obs_dim,
        n_actions=n_actions,
        layer_sizes=[32, 32],
        device=device,
    )

    wrapper = FrozenPolicyWrapper(policy, device="cpu")

    obs = np.zeros(obs_dim, dtype=np.float32)
    idx = wrapper.act(obs)

    assert isinstance(idx, int)
    assert 0 <= idx < n_actions
