import numpy as np
from rlgym_sim.make import Match
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition
from rlgym_sim.utils.action_parsers import DefaultAction

from rlgymbotv2.mysim.state_setters.default_state import DefaultState
from rlgymbotv2.mysim.obs_builders.advanced_obs_plus import AdvancedObsPlus

import pytest

class ZeroReward(RewardFunction):
    def reset(self, *_args, **_kwargs):
        pass

    def get_reward(self, *_args, **_kwargs) -> float:
        return 0.0

    def get_final_reward(self, *_args, **_kwargs) -> float:
        return 0.0

@pytest.mark.skip(reason="Not needed right now")
def test_cheap_opponent_preserves_shape():
    obs_builder = AdvancedObsPlus(
        max_allies=0,
        max_opponents=1,
        cheap_opponent=True,
        stack_size=1,
    )

    match = Match(
        state_setter=DefaultState(),
        reward_function=ZeroReward(),
        terminal_conditions=[],
        obs_builder=obs_builder,
        action_parser=DefaultAction(),
        # tick_skip is not supported as a kwarg in this rlgym_sim version
    )


    state = match._get_state()
    obs_builder.pre_step(state)

    blue = state.players[0]    # team 0
    orange = state.players[1]  # team 1

    zero_action = np.zeros(1, dtype=np.int64)

    o_blue = obs_builder.build_obs(blue, state, zero_action)
    o_orange = obs_builder.build_obs(orange, state, zero_action)

    # shapes must match
    assert o_blue.shape == o_orange.shape

    # cheap path shouldn't nuke everything; at least something should be non-zero
    assert np.any(o_orange != 0.0)
