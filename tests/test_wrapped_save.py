# [2025-11-28] Rev 1.0.1 (NJH) - Update resume guard test to use
#              pl.crossed_milestone instead of poking _next_milestone /
#              wrapped_save internals.


import types
import numpy as np
from rlgymbotv2.training.ppo_config import SAVE_EVERY, CHECKPOINT_ROOT, MILESTONE_INTERVAL, SELFPLAY_ENABLED, LOG_PHASE
import pytest
from rlgymbotv2.training import ppo_learner as pl

def make_dummy_learner():
    class DummyPPO:
        last_policy_entropy = 0.5
        last_kl_divergence = 0.01
        policy = object()

    class DummyAgent:
        average_reward = [1.0, 2.0, 3.0]

    class DummyLearner:
        def __init__(self):
            self.agent = DummyAgent()
            self.ppo_learner = DummyPPO()
            self.checkpoints_save_folder = CHECKPOINT_ROOT
            self._next_milestone = MILESTONE_INTERVAL

    return DummyLearner()

def test_wrapped_save_no_milestone(monkeypatch):
    from rlgymbotv2.training import ppo_learner

    learner = make_dummy_learner()
    ppo_learner.learner = learner
    ppo_learner.SELFPLAY_ENABLED = SELFPLAY_ENABLED
    ppo_learner.SELFPLAY_UPDATE_MODE = "manual"
    ppo_learner.MILESTONE_INTERVAL = MILESTONE_INTERVAL
    ppo_learner.phase_name = LOG_PHASE

    calls = []

    monkeypatch.setattr(ppo_learner, "make_milestone_dir", lambda root, phase, steps: f"/fake/{steps}")
    monkeypatch.setattr(ppo_learner.shutil, "copytree", lambda *args, **kwargs: calls.append(("copytree", args)))
    monkeypatch.setattr(ppo_learner.torch, "save", lambda *args, **kwargs: calls.append(("save")))
    monkeypatch.setattr(ppo_learner, "write_meta", lambda *args, **kwargs: calls.append(("meta", args)))

    # wrap without orig_save
    orig_save = None
    def wrapped_save(cumulative_timesteps: int):
        return ppo_learner.wrapped_save(cumulative_timesteps)

    # Below next_milestone → nothing should happen
    wrapped_save(MILESTONE_INTERVAL - 100_000)

    assert calls == []
    assert learner._next_milestone == MILESTONE_INTERVAL

def test_wrapped_save_at_milestone(monkeypatch):
    from rlgymbotv2.training import ppo_learner

    learner = make_dummy_learner()
    ppo_learner.learner = learner
    ppo_learner.SELFPLAY_ENABLED = SELFPLAY_ENABLED
    ppo_learner.SELFPLAY_UPDATE_MODE = "manual"
    ppo_learner.MILESTONE_INTERVAL = MILESTONE_INTERVAL
    ppo_learner.CHECKPOINT_ROOT = CHECKPOINT_ROOT
    ppo_learner.phase_name = LOG_PHASE
    ppo_learner.obs_dim = 150
    ppo_learner.POLICY_LAYERS = [256, 256]
    ppo_learner.CRITIC_LAYERS = [256, 256]

    events = []

    monkeypatch.setattr(ppo_learner, "make_milestone_dir", lambda root, phase, steps: f"/fake/{steps}")
    monkeypatch.setattr(ppo_learner.shutil, "copytree", lambda *args, **kwargs: events.append("copytree"))
    monkeypatch.setattr(ppo_learner.torch, "save", lambda *args, **kwargs: events.append("save"))
    monkeypatch.setattr(ppo_learner, "write_meta", lambda *args, **kwargs: events.append("meta"))

    # Make orig_save a no-op
    ppo_learner.orig_save = lambda steps: events.append("save")

    # Call wrapped_save at exactly the milestone
    ppo_learner.wrapped_save(MILESTONE_INTERVAL)

    # Should have called orig_save + copytree + save + meta
    # assert "orig_save_5000000" in events
    assert "copytree" in events
    assert "save" in events
    assert "meta" in events

    # Next milestone moved forward
    assert learner._next_milestone == 10_000_000

def test_wrapped_save_resume_guard():
    """
    Resume semantics we care about now:

    - Simply being at 12M (already past one or more 5M intervals) should NOT
      magically trigger an old milestone; milestones only happen when a
      SAVE_EVERY window actually crosses an interval boundary.

    - Crossing 15M from below (e.g. 14.9M -> 15.0M) should trigger.
    """
    interval = pl.MILESTONE_INTERVAL  # e.g. 5_000_000

    # Resumed training; last save was at 12M. We are already between 10M and 15M.
    # There should be NO new milestone just for being at 12M.
    prev_steps = 10_000_000
    total_steps = 12_000_000
    assert not pl.crossed_milestone(prev_steps, total_steps, interval)

    # Later we continue saving, and eventually cross the 15M boundary:
    prev_steps = 14_900_000
    total_steps = 15_000_000
    assert pl.crossed_milestone(prev_steps, total_steps, interval)