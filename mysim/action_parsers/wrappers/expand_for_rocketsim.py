from typing import Optional, Dict, Any, List
import numpy as np
from ._compat import safe_get_action_space


class ExpandForRocketSim:
    """
    Expands each agent's (8,) control vector into `tick_skip` repeats
    and flattens them into the control list layout RocketSim expects:
      [agent0_tick0, agent0_tick1, ..., agent0_tick{n-1},
       agent1_tick0, agent1_tick1, ..., agent1_tick{n-1}, ...]
    """

    def __init__(self, inner, tick_skip: int = 8):
        self.inner = inner
        self.tick_skip = int(tick_skip)

    def get_action_space(self, agent=None):
        return safe_get_action_space(self.inner, agent)

    def reset(self, *args, **kwargs):
        if hasattr(self.inner, "reset"):
            self.inner.reset(*args, **kwargs)

    def parse_actions(self, actions, state, shared_info: Optional[Dict[str, Any]] = None, *_, **__) -> List[np.ndarray]:
        seq = self.inner.parse_actions(actions, state, shared_info)
        # inner returns a 2D array (N_agents, 8)
        arr = np.asarray(seq, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]  # (1, 8)
        elif arr.ndim != 2 or arr.shape[1] != 8:
            raise ValueError(f"ExpandForRocketSim expected (N_agents,8), got {arr.shape}")

        out: List[np.ndarray] = []
        for agent_idx in range(arr.shape[0]):
            act = arr[agent_idx]
            # repeat the same control tick_skip times
            for _ in range(self.tick_skip):
                out.append(act.copy())
        return out
