from typing import Dict, Any, Optional, List
import numpy as np
from ._compat import safe_get_action_space

class StickyButtonsWrapper:
    def __init__(self, inner, hold_ticks: int = 3):
        self.inner = inner
        self.hold = int(hold_ticks)
        self._counters: Dict[int, np.ndarray] = {}  # key by agent index (order)

    def get_action_space(self, agent=None):
        return safe_get_action_space(self.inner, agent)

    def reset(self, *args, **kwargs):
        self._counters.clear()
        if hasattr(self.inner, "reset"):
            self.inner.reset(*args, **kwargs)

    def parse_actions(self, actions, state, shared_info: Optional[Dict[str, Any]] = None, *_, **__) -> List[np.ndarray]:
        seq = self.inner.parse_actions(actions, state, shared_info)
        assert isinstance(seq, list), f"StickyButtonsWrapper expected list, got {type(seq)}"

        result: List[np.ndarray] = []
        for i, a in enumerate(seq):
            arr = np.asarray(a, dtype=np.float32)
            if arr.ndim == 1:       # (8,)
                arr = arr[None, :]  # (1,8)
            assert arr.shape[1] == 8, f"Expected (*,8), got {arr.shape}"

            counters = self._counters.get(i)
            if counters is None:
                counters = np.zeros(3, dtype=int)  # [jump, boost, handbrake]

            for t in range(arr.shape[0]):
                for j, idx in enumerate((5, 6, 7)):
                    if arr[t, idx] > 0.5:
                        counters[j] = self.hold
                    else:
                        counters[j] = max(0, counters[j] - 1)
                    arr[t, idx] = 1.0 if counters[j] > 0 else 0.0

            self._counters[i] = counters
            # Return compact (8,) when T==1
            result.append(arr[0] if arr.shape[0] == 1 else arr)
        return result
