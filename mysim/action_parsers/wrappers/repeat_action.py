# mysim/action_parsers/wrappers/repeat_action.py
from typing import Optional, Dict, Any, List
import numpy as np
from ._compat import safe_get_action_space

class RepeatAction:
    """
    Ensure each agent's action is repeated for `repeats` ticks per env step,
    returning a list of (repeats, 8) arrays (one per controlled agent).
    """
    def __init__(self, parser, repeats: int = 8):
        self.parser = parser
        self.repeats = int(repeats)

    def get_action_space(self, agent=None):
        return safe_get_action_space(self.parser, agent)

    def reset(self, *args, **kwargs):
        if hasattr(self.parser, "reset"):
            self.parser.reset(*args, **kwargs)

    def parse_actions(self, actions, state, shared_info: Optional[Dict[str, Any]] = None, *_, **__) -> List[np.ndarray]:
        seq = self.parser.parse_actions(actions, state, shared_info)  # list of per-agent actions
        out: List[np.ndarray] = []
        for a in seq:
            arr = np.asarray(a, dtype=np.float32)
            if arr.ndim == 1:           # (8,)
                arr = arr[None, :]      # (1,8)
            if arr.shape[0] == 1:
                arr = np.repeat(arr, self.repeats, axis=0)
            elif arr.shape[0] > self.repeats:
                arr = arr[: self.repeats, :]
            elif arr.shape[0] < self.repeats:
                # Pad by repeating last row
                pad = np.repeat(arr[-1:], self.repeats - arr.shape[0], axis=0)
                arr = np.vstack([arr, pad])
            out.append(arr)
        return out
