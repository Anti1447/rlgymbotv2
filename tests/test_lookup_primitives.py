import numpy as np
import pytest

from rlgymbotv2.mysim.action_parsers.simple_discrete_hybrid_action import SimpleHybridDiscreteAction

# Control order (as implemented in your parser):
# [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

def _has_row(tbl, row, atol=1e-6):
    r = np.array(row, dtype=np.float32)
    return np.any(np.all(np.isclose(tbl, r, atol=atol), axis=1))

def test_lookup_primitives_present():
    tbl = SimpleHybridDiscreteAction().lookup_table

    must_haves = {
        "pure_jump":               [0, 0, 0, 0, 0, 1, 0, 0],
        "jump_plus_handbrake":     [0, 0, 0, 0, 0, 1, 0, 1],
        "throttle_pitch_forward":  [1, 0, 1, 0, 0, 0, 0, 0],
        "throttle_pitch_backward": [1, 0,-1, 0, 0, 0, 0, 0],
        "roll_only_pos":           [0, 0, 0, 0, 1, 0, 0, 0],
        "roll_only_neg":           [0, 0, 0, 0,-1, 0, 0, 0],
        "pitch_only_pos":          [0, 0, 1, 0, 0, 0, 0, 0],
        "pitch_only_neg":          [0, 0,-1, 0, 0, 0, 0, 0],
        "yaw_only_pos":            [0, 0, 0, 1, 0, 0, 0, 0],
        "yaw_only_neg":            [0, 0, 0,-1, 0, 0, 0, 0],
    }

    missing = [name for name, row in must_haves.items() if not _has_row(tbl, row)]
    assert not missing, f"Missing key recovery primitives: {missing}"
