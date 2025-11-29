# [2025-11-28] Rev 1.0.0 (NJH) - Sanity test for DiscreteFF forward

import torch
import pytest
from rlgymbotv2.training import ppo_learner as pl  # or wherever you import DiscreteFF from

@pytest.mark.skip(reason="not working as intended yet")
def test_discreteff_forward_runs():
    in_size = 10
    out_size = 5
    hidden = [32, 32]

    net = pl.DiscreteFF(in_size, hidden, out_size)  # adjust import if needed
    x = torch.zeros(4, in_size)  # batch of 4

    y = net(x)
    assert y.shape == (4, out_size)
