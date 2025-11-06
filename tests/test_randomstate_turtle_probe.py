import numpy as np
import pytest
import rlgym_sim

from mysim.state_setters.random_state import RandomState
from mysim.action_parsers.simple_discrete_hybrid_action import SimpleHybridDiscreteAction
from mysim.obs_builders.advanced_obs_plus import AdvancedObsPlus

def _is_turtled(car, thresh=-0.2):
    return bool(getattr(car, "on_ground", False)) and float(car.up()[2]) < thresh

@pytest.mark.xfail(strict=False, reason="Diagnostic until recovery is reliable in RandomState.")
def test_randomstate_has_at_least_one_turtle_recovery():
    env = rlgym_sim.make(
        state_setter=RandomState(True, True, False),
        action_parser=SimpleHybridDiscreteAction(),
        obs_builder=AdvancedObsPlus(),
        team_size=1,
        spawn_opponents=False,
        tick_skip=1,
    )
    try:
        env.reset()
        events = succ = 0
        window = 60
        horizon = 1200
        t = 0
        while t < horizon:
            a = env.action_space.sample()
            env.step(a)
            me = env._prev_state.players[0].car_data
            if _is_turtled(me):
                events += 1
                ok = False
                for _ in range(window):
                    env.step(env.action_space.sample())
                    t += 1
                    me2 = env._prev_state.players[0].car_data
                    if bool(getattr(me2, "on_ground", False)) and float(me2.up()[2]) > 0.6:
                        succ += 1
                        ok = True
                        break
            t += 1
        if events == 0:
            pytest.xfail("No turtle events observed in horizon (random).")
        assert succ > 0, "Observed turtling but never self-righted within the window."
    finally:
        env.close()
