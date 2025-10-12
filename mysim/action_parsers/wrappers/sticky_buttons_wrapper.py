# Usage: Could be chained with ClipActionWrapper
# action_parser = StickyButtonsWrapper(ClipActionWrapper(ContinuousAction()), hold_ticks=3)


import numpy as np

class StickyButtonsWrapper:
    def __init__(self, inner, hold_ticks=3):
        self.inner = inner
        self.hold = int(hold_ticks)
        self._counters = {}  # agent_id -> [jump, boost, handbrake]

    def get_action_space(self): return self.inner.get_action_space()

    def parse_actions(self, actions, state):
        out = self.inner.parse_actions(actions, state)
        out = np.asarray(out, dtype=np.float32)
        n = out.shape[0]
        for i in range(n):
            cnt = self._counters.get(i, [0,0,0])
            for j, idx in enumerate((5,6,7)):     # jump, boost, handbrake
                if out[i, idx] > 0.5:
                    cnt[j] = self.hold
                else:
                    cnt[j] = max(0, cnt[j]-1)
                out[i, idx] = 1.0 if cnt[j] > 0 else 0.0
            self._counters[i] = cnt
        return out
