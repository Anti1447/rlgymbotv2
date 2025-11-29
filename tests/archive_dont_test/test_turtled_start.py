import numpy as np
import pytest
import rlgym_sim

from rlgymbotv2.mysim.state_setters.turtled_start import TurtledStart
from rlgymbotv2.mysim.action_parsers.simple_discrete_hybrid_action import SimpleHybridDiscreteAction
from rlgymbotv2.mysim.obs_builders.advanced_obs_plus import AdvancedObsPlus

# [thr, steer, pitch, yaw, roll, jump, boost, handbrake]

def _find(tbl, row, atol=1e-6):
    r = np.array(row, dtype=np.float32)
    idx = np.where(np.all(np.isclose(tbl, r, atol=atol), axis=1))[0]
    return int(idx[0]) if idx.size else None

def _upright(car, up_thresh=0.6):
    return float(car.up()[2]) > up_thresh

def _uprighted_on_ground(car, up_thresh=0.6):
    return bool(getattr(car, "on_ground", False)) and float(car.up()[2]) > up_thresh

@pytest.mark.skip(reason="Not needed right now")
def test_turtled_start_recovers_quickly():
    ap  = SimpleHybridDiscreteAction()
    tbl = ap.lookup_table

    rows = {
        "idle":                    [0,0,0,0,0,0,0,0],
        "pitch_only_pos":          [0,0, 1,0,0,0,0,0],
        "pitch_only_neg":          [0,0,-1,0,0,0,0,0],
        "yaw_only_pos":            [0,0,0, 1,0,0,0,0],
        "yaw_only_neg":            [0,0,0,-1,0,0,0,0],
        "roll_only_pos":           [0,0,0,0, 1,0,0,0],
        "roll_only_neg":           [0,0,0,0,-1,0,0,0],
        "throttle_pitch_forward":  [1,0, 1,0,0,0,0,0],
        "throttle_pitch_backward": [1,0,-1,0,0,0,0,0],
        # optional boost torque helps some physics builds
        "roll_only_pos_boost":     [0,0,0,0, 1,0,1,0],
        "roll_only_neg_boost":     [0,0,0,0,-1,0,1,0],
    }
    idx = {k: _find(tbl, v) for k, v in rows.items() if v is not None}
    missing = [k for k, v in idx.items() if v is None]
    assert not missing, f"Missing primitives required for test: {missing}"

    env = rlgym_sim.make(
        state_setter=TurtledStart(z=14.0, yaw_random=False, spawn_opponents=False),
        action_parser=ap,
        obs_builder=AdvancedObsPlus(),
        team_size=1,
        spawn_opponents=False,
        tick_skip=1,
    )

    try:
        env.reset()

        # S: settle a few ticks to hit the floor
        for _ in range(6):
            env.step(idx["idle"])

        # B: verify torques change angular velocity (mapping sanity)
        moved = False
        for key in ["pitch_only_pos","pitch_only_neg","yaw_only_pos","yaw_only_neg","roll_only_pos","roll_only_neg"]:
            me_b = env._prev_state.players[0].car_data
            w_b  = np.asarray(me_b.angular_velocity, dtype=float)
            env.step(idx[key])
            me_a = env._prev_state.players[0].car_data
            w_a  = np.asarray(me_a.angular_velocity, dtype=float)
            if np.linalg.norm(w_a - w_b) > 1e-3:
                moved = True
                break
        assert moved, "Torque-only rows did not change angular velocity; control mapping/order likely wrong."

        # C: cycles mixing torques, boosted roll, and throttle+pitch rocking
        cycle = (
            [idx["pitch_only_pos"], idx["roll_only_pos"], idx["yaw_only_pos"],
             idx["pitch_only_neg"], idx["roll_only_neg"], idx["yaw_only_neg"],
             idx["roll_only_pos_boost"], idx["roll_only_neg_boost"],
             idx["throttle_pitch_forward"], idx["throttle_pitch_backward"]]
        )
        for a in cycle * 10:  # give it ample time
            env.step(a)
            me = env._prev_state.players[0].car_data
            if _upright(me):
                # landing window to register wheel contact
                for _ in range(60):
                    env.step(idx["idle"])
                    me2 = env._prev_state.players[0].car_data
                    if _uprighted_on_ground(me2):
                        return
                # if still mid-air, continue the cycle

        me = env._prev_state.players[0].car_data
        pytest.fail(f"Never uprighted; final on_ground={bool(getattr(me,'on_ground',False))} up_z={float(me.up()[2]):.2f}")
    finally:
        env.close()
