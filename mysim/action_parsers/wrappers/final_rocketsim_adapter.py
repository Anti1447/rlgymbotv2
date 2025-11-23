"""
mysim/action_parsers/wrappers/final_rocketsim_adapter.py

FinalRocketSimAdapter
=====================
Ensures RocketSim receives a correctly flattened (N,9) control array.

Expected control order (RLBot layout, matching rlgym_sim):
[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

RocketSim ultimately receives:
[spectator_id, throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
"""
from typing import Optional, Dict, Any
import numpy as np
from ._compat import safe_get_action_space
from rlgymbotv2.mysim.debug_config import global_debug_mode, debug_actions


class FinalRocketSimAdapter:
    """
    Outermost adapter ensuring RocketSim receives the correct flattened control array.

    It does NOT modify control order (assumes already RLBot-compatible).
    Only appends spectator IDs and stores the full 9-wide controls for RocketSim.
    """

    def __init__(self, inner):
        self.inner = inner
        self.last_full_controls: Optional[np.ndarray] = None
        self._warned_non_integer = False  # avoid spam

    def _coerce_indices(self, actions):
        arr = np.asarray(actions)
        if arr.dtype.kind in "fc":  # float/complex
            rounded = np.rint(arr)
            # only warn if any element is meaningfully non-integer
            if not np.allclose(arr, rounded, atol=1e-6):
                if not self._warned_non_integer:
                    print("[FinalRocketSimAdapter] WARNING: received non-integer discrete actions; rounding to nearest index.")
                    self._warned_non_integer = True
            arr = rounded.astype(int)
        return arr

    def get_action_space(self, agent=None):
        return safe_get_action_space(self.inner, agent)

    def reset(self, *args, **kwargs):
        if hasattr(self.inner, "reset"):
            self.inner.reset(*args, **kwargs)
        self.last_full_controls = None

    def parse_actions(self, actions, state, shared_info=None, *_, **__):
        # If the policy head outputs float dtypes, coerce to ints (Discrete)
        actions = self._coerce_indices(actions)

        seq = self.inner.parse_actions(actions, state, shared_info)
        arr = np.asarray(seq, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] == 9:
            self.last_full_controls = arr
            return arr[:, 1:]  # drop spectator_id for Match

        if arr.shape[1] == 8:
            # convert RocketSim (thr, steer, yaw, pitch, roll, jump, boost, handbrake)
            # -> RLBot/RLGym order (thr, steer, pitch, yaw, roll, jump, boost, handbrake)
            arr2 = arr.copy()
            arr2[:, [2, 3]] = arr2[:, [3, 2]]
            ids = getattr(state, "spectator_ids", None) or list(range(1, arr2.shape[0] + 1))
            out = np.zeros((arr2.shape[0], 9), dtype=np.float32)
            for i, spec_id in enumerate(ids[: arr2.shape[0]]):
                out[i, 0] = spec_id
                out[i, 1:] = arr2[i]
            self.last_full_controls = out
            return arr2

        raise ValueError(f"Unexpected parser output shape: {arr.shape}")

    def get_controls_for_rocketsim(self) -> np.ndarray:
        """
        Returns the most recent (N,9) control array flattened for RocketSim.
        """
        if self.last_full_controls is None:
            raise RuntimeError("No full controls cached yet.")
        return self.last_full_controls.flatten().astype(np.float32)
