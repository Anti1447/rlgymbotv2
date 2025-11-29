# [2025-11-28] Rev 1.0.1 (NJH) - Simplify milestone tests to rely on
#              pl.crossed_milestone instead of a dummy wrapped_save stub.


import numpy as np
import types
from rlgymbotv2.training import ppo_learner as pl


def test_milestones_every_5M():
    """
    We should only see milestones at exact 5M, 10M, ... boundaries when
    saving every SAVE_EVERY steps.
    """
    interval = pl.MILESTONE_INTERVAL       # e.g. 5_000_000
    save_every = pl.SAVE_EVERY            # e.g. 100_000

    prev = 0
    milestones = []
    step = 0

    # Simulate saves from 0 up to 10.1M in SAVE_EVERY increments
    while step <= 10_100_000:
        step += save_every
        if pl.crossed_milestone(prev, step, interval):
            milestones.append(step)
        prev = step

    # Only 5M and 10M boundaries should count
    assert milestones == [5_000_000, 10_000_000]

