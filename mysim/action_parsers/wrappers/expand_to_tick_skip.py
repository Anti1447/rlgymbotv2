from typing import Optional, Dict, Any, List
import numpy as np
from ._compat import safe_get_action_space

class ExpandToTickSkip:
    """
    Outer wrapper that expands per-agent actions to a flat list sized (agents * repeats),
    arranged per-agent then per-tick, as RocketSim expects.

    Input from inner: list of per-agent actions, each (8,) or (T,8)
    Output: flat list:
      [agent0_tick0, agent0_tick1, ... agent0_tick{repeats-1},
       agent1_tick0, agent1_tick1, ... agent1_tick{repeats-1}, ...]
    """

    def __init__(self, inner, repeats: int):
        self.inner = inner
        self.repeats = int(repeats)

    def get_action_space(self, agent=None):
        return safe_get_action_space(self.inner, agent)

    def reset(self, *args, **kwargs):
        if hasattr(self.inner, "reset"):
            self.inner.reset(*args, **kwargs)

    def parse_actions(self, actions, state, shared_info: Optional[Dict[str, Any]] = None, *_, **__) -> List[np.ndarray]:
        seq = self.inner.parse_actions(actions, state, shared_info)  # list per agent
        out: List[np.ndarray] = []

        for a in seq:
            arr = np.asarray(a, dtype=np.float32)
            # normalize to (1,8)
            if arr.ndim == 1:      # (8,)
                arr = arr[None, :] # (1,8)
            elif arr.ndim == 2 and arr.shape[1] == 8:
                # If a sequence (T,8) arrives, use the LAST row for this step.
                arr = arr[-1: , :]  # (1,8)
            else:
                raise ValueError(f"ExpandToTickSkip: unsupported inner action shape {arr.shape}")

            # expand to (repeats, 8) by repeating this row
            arr = np.repeat(arr, self.repeats, axis=0)  # (repeats, 8)

            # append per-agent then per-tick
            for t in range(self.repeats):
                out.append(arr[t])

        return out
