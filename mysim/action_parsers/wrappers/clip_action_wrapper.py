# Usage:
# from mysim.action_parsers.continuous_act import ContinuousAction
# from mysim.action_parsers.wrappers.clip_action_wrapper import ClipActionWrapper
# 
# action_parser = ClipActionWrapper(ContinuousAction())

import numpy as np

class ClipActionWrapper:
    """
    Wrap any ActionParser. Hard-clips incoming actions to [-1,1], then enforces
    env contract on the parsed output: first 5 in [-1,1], last 3 ∈ {0,1}.
    """
    def __init__(self, inner, low=-1.0, high=1.0):
        self.inner = inner
        self.low = float(low)
        self.high = float(high)

    def get_action_space(self):
        return self.inner.get_action_space()

    def parse_actions(self, actions, state):
        a = np.asarray(actions, dtype=np.float32)
        a = np.clip(a, self.low, self.high)
        out = self.inner.parse_actions(a, state)
        out = np.asarray(out, dtype=np.float32)
        out[..., :5] = np.clip(out[..., :5], -1.0, 1.0)
        out[..., 5:] = (out[..., 5:] > 0).astype(np.float32)
        return out
