# mysim/action_parsers/wrappers/collapse_to_single_tick.py
from typing import Optional, Dict, Any, List
import numpy as np
from ._compat import safe_get_action_space


class CollapseToSingleTick:
    """
    Ensures the action parser returns a 2D array (N_agents, 8) per env.step().
    If the inner parser returns per-agent sequences (T,8), we take the *last* row.
    Then flattens and prepends spectator IDs for RocketSim.
    """

    def __init__(self, inner):
        self.inner = inner

    def get_action_space(self, agent=None):
        return safe_get_action_space(self.inner, agent)

    def reset(self, *args, **kwargs):
        if hasattr(self.inner, "reset"):
            self.inner.reset(*args, **kwargs)

    def parse_actions(
        self, actions, state, shared_info: Optional[Dict[str, Any]] = None, *_, **__
    ) -> np.ndarray:
        seq: List[np.ndarray] = self.inner.parse_actions(actions, state, shared_info)

        # --- Step 1: collapse multi-tick sequences to one tick per agent
        rows = []
        for a in seq:
            arr = np.asarray(a, dtype=np.float32)
            if arr.ndim == 2:  # (T, 8)
                rows.append(arr[-1])  # take last tick of sequence
            elif arr.ndim == 1:  # (8,)
                rows.append(arr)
            else:
                raise ValueError(f"Unsupported action shape from inner parser: {arr.shape}")

        out = np.stack(rows, axis=0)  # shape (N_agents, 8)
        # print(f"[Collapse] out shape={out.shape}") #debug print

        # --- Step 2: flatten + prepend spectator IDs (RocketSim expects 9 per agent)
        flat = []
        # Try to extract spectator IDs; fallback if not found
        ids = getattr(state, "spectator_ids", None)
        if ids is None:
            # Fallback to sequential 1..N_agents
            ids = list(range(1, out.shape[0] + 1))

        for i, spec_id in enumerate(ids[: out.shape[0]]):
            flat.extend([spec_id] + out[i].tolist())

        flat = np.asarray(flat, dtype=np.float32)
        # print(f"[Adapter] input count={len(ids)}, output shape={flat.shape}") # Debug print lines
        # print(f"[Collapse] {out.shape[0]} agents collapsed. Final flat length={len(flat)}")
        return flat
