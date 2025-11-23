import torch
from pathlib import Path
from typing import Tuple
import numpy as np

from rlgym_ppo.ppo.ppo_learner import PPOLearner  # internal class used by Learner.agent
from rlgymbotv2.training.ppo_learner import build_rocketsim_env
from rlgymbotv2.mysim.training_utils.selfplay_pool import SelfPlayOpponentPool


class FrozenPolicy:
    """
    Lightweight wrapper around a PPO policy network loaded from a checkpoint.
    Only does inference, no gradients, no updates.
    """

    def __init__(self, ckpt_folder: Path, device: str = "cpu"):
        self.device = device
        self.ckpt_folder = ckpt_folder
        self._load()

    def _load(self):
        # We re-use PPOLearner internals to reconstruct the policy network
        # shape; you already rely on this structure via Learner.
        policy_path = self.ckpt_folder / "PPO_POLICY.pt"
        if not policy_path.exists():
            raise FileNotFoundError(str(policy_path))

        # torch.load returns state dict; we need a dummy PPOLearner to get arch.
        # Easiest: load via torch.jit or direct state dict into the same policy
        # architecture you use in your Learner.agent.policy.
        self.state_dict = torch.load(policy_path, map_location=self.device)
        # NOTE: We can't fully reconstruct PPOLearner here without
        # all hyperparams; instead we'll be conservative and treat this
        # as a stub; the real network is already in your running Learner
        # during training. For pure evaluation, you'd normally spin up a
        # second Learner instance and call its agent directly.

        # So for now, think of this as a placeholder; we won't run this
        # class in isolation inside training. Instead, we'll talk about
        # how to hook into an actual second Learner below.


def run_selfplay_eval(
    current_ckpt: Path,
    opponent_ckpt: Path,
    n_games: int = 5,
    device: str = "cuda"
) -> Tuple[int, int]:
    """
    Pits current policy vs opponent policy in n_games 1v1 matches.
    Returns (current_wins, opponent_wins).

    This function is intentionally high-level / pseudo-code because
    proper multi-agent control depends on how you wired rlgym_sim Match.
    """

    # PSEUDO-CODE SKETCH (you'll need to adapt to your existing match code):

    # 1) Build a special self-play env factory that creates a 1v1 env where:
    #    - Agent 0 uses "current" policy.
    #    - Agent 1 uses "opponent" policy.
    #
    # 2) For each game:
    #       reset env
    #       while not done:
    #           obs_current, obs_opponent = split_obs(...)
    #           a0 = current_policy(obs_current)
    #           a1 = opponent_policy(obs_opponent)
    #           obs, rewards, done, info = env.step([a0, a1])
    #    Track which team scored more.
    #
    # Because your current training stack is single-agent PPO, wiring this
    # cleanly requires exposing both agents' obs + actions. That's a
    # bigger architectural change, so here we focus on the management
    # layer (pool + evaluation hook) and let you fill in the exact match
    # wiring you prefer.

    raise NotImplementedError("Wire this into your multi-agent match/rlgym_sim setup.")
