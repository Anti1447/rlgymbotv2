import numpy as np

class EMASmoothingWrapper:
    def __init__(self, inner, alpha=0.5):
        self.inner = inner
        self.alpha = float(alpha)
        self._prev = None

    def get_action_space(self): return self.inner.get_action_space()

    def parse_actions(self, actions, state):
        out = self.inner.parse_actions(actions, state)
        out = np.asarray(out, dtype=np.float32)
        if self._prev is None: self._prev = out.copy()
        out[..., :5] = self.alpha*out[..., :5] + (1.0-self.alpha)*self._prev[..., :5]
        self._prev = out.copy()
        return out
