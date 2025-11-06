import numpy as np
import gym.spaces
from typing import Optional, Dict, Any, Sequence
from mysim.gamestates import GameState
from mysim.action_parsers import ActionParser

class DiscreteAction2(ActionParser):
    """
    Simple discrete action space. All analog actions have `n_bins` bins (odd), mapped to {-1, 0, 1} when n_bins=3.
    Accepts multiple input shapes:
      - scalar/size-1 flat index (mixed-radix decoded to 8 components)
      - flat array with total size % 8 == 0 (reshaped to (-1, 8))
      - arrays whose last dim is already 8
    Output is float32 in [-1,1] for first 5, and {0,1} for last 3.
    """

    def __init__(self, n_bins: int = 3):
        super().__init__()
        assert n_bins % 2 == 1, "n_bins must be an odd number"
        self._n_bins = n_bins
        # Mixed-radix bases for the 8 controls:
        # [throttle, steer, yaw, pitch, roll, jump, boost, handbrake]
        self._bases: Sequence[int] = [self._n_bins] * 5 + [2, 2, 2]

    def get_action_space(self) -> gym.spaces.Space:
        # Keep MultiDiscrete as your declared space; we just decode flexibly in parse_actions.
        return gym.spaces.MultiDiscrete(list(self._bases))

    def _decode_mixed_radix(self, idx: int) -> np.ndarray:
        """Decode a single integer index into 8 components using mixed radix `self._bases`."""
        out = np.zeros(8, dtype=np.int64)
        x = int(idx)
        for j in reversed(range(8)):
            base = self._bases[j]
            out[j] = x % base
            x //= base
        return out

    def _coerce_to_actions_8(self, actions: np.ndarray) -> np.ndarray:
        """Return actions shaped as (-1, 8) of dtype int64 (pre-normalization)."""
        a = np.asarray(actions)
        # Case A: single index (scalar or [1]) → decode
        if a.size == 1 and (a.ndim == 0 or (a.ndim == 1 and a.shape[0] == 1)):
            decoded = self._decode_mixed_radix(int(a.reshape(())))
            return decoded.reshape(1, 8).astype(np.int64)

        # Case B: already has last-dim 8
        if a.ndim >= 1 and a.shape[-1] == 8:
            return a.reshape(-1, 8).astype(np.int64)

        # Case C: flat array with size multiple of 8
        if a.ndim == 1 and (a.size % 8 == 0):
            return a.reshape(-1, 8).astype(np.int64)

        # If none matched, try best-effort: if total size matches players*8 later, we’ll reshape there.
        return a.astype(np.int64)

    def parse_actions(
        self,
        actions: np.ndarray,
        state: GameState,
        shared_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:

        # 1) Coerce to (-1, 8) int grid indices when possible
        a = self._coerce_to_actions_8(actions)

        n_players = len(state.players)

        # Shape reconcile: if still not (-1,8), attempt to infer using n_players
        if a.ndim != 2 or a.shape[-1] != 8:
            total = a.size
            if total == n_players * 8:
                a = a.reshape(n_players, 8)
            elif total % 8 == 0:
                a = a.reshape(-1, 8)
            else:
                raise ValueError(
                    f"[DiscreteAction] Cannot interpret actions with shape {np.shape(actions)} "
                    f"(total size {total}); expected a scalar index, (..., 8), or size % 8 == 0."
                )

        # If batch size doesn’t match players, we’ll broadcast or trim conservatively
        if a.shape[0] != n_players:
            if a.shape[0] == 1 and n_players > 1:
                a = np.repeat(a, n_players, axis=0)
            else:
                # Trim or pad (pad by repeating last row)
                if a.shape[0] > n_players:
                    a = a[:n_players]
                else:
                    pad = np.repeat(a[-1:], n_players - a.shape[0], axis=0)
                    a = np.concatenate([a, pad], axis=0)

        a = a.astype(np.float32)

        # 2) Map analog bins {0..n_bins-1} → [-1..1] evenly
        # Example (n_bins=3): 0→-1, 1→0, 2→+1
        half = self._n_bins // 2
        a[:, :5] = (a[:, :5] / half) - 1.0

        # 3) Ensure binary channels are 0/1
        a[:, 5:] = (a[:, 5:] > 0).astype(np.float32)

        # 4) Clip for safety
        np.clip(a[:, :5], -1.0, 1.0, out=a[:, :5])

        return a
