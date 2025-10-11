# Minimal local copy to avoid rlgym_sim dependency at runtime.
import numpy as np

class DiscreteAction:
    """
    Simple discrete action space. Analog actions have 3 bins: -1, 0, 1.
    Layout: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    """
    def __init__(self, n_bins=3):
        assert n_bins % 2 == 1, "n_bins must be an odd number"
        self._n_bins = n_bins

    def parse_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = actions.reshape((-1, 8)).astype(np.float32)
        # map bins {0..n_bins-1} → {-1 .. 1} for first 5
        actions[..., :5] = actions[..., :5] / (self._n_bins // 2) - 1
        return actions
