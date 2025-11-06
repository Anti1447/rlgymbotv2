from typing import Optional, Dict, Any, List, Union
import numpy as np
import gym.spaces
from rlgym.api import ActionParser, AgentID
from mysim.gamestates import GameState

class LookupTableAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, gym.spaces.Space]):
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @property
    def lookup_table(self) -> np.ndarray:
        return self._lookup_table

    def get_action_space(self, agent=None) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([len(self._lookup_table)])

    def reset(self, *args, **kwargs) -> None:
        return None

    def parse_actions(
        self,
        actions: Union[Dict[AgentID, np.ndarray], np.ndarray, List[np.ndarray]],
        state: GameState,
        shared_info: Optional[Dict[str, Any]] = None,
        *_, **__
    ) -> List[np.ndarray]:
        """
        RETURNS: list with one entry per provided action (agent). No expansion to all players.
        """
        table = self._lookup_table

        def decode_idx_to_buttons(idx_arr: np.ndarray) -> np.ndarray:
            a = np.asarray(idx_arr)
            if a.ndim == 2 and a.shape[1] == 1:
                a = a.squeeze(1)               # (T,)
            if a.ndim == 2 and a.shape[1] == 8:
                return a.astype(np.float32)    # already decoded
            a = a.astype(int).ravel()          # (T,)
            a = np.clip(a, 0, len(table) - 1)
            return table[a].astype(np.float32) # (T,8)

        out: List[np.ndarray] = []

        if isinstance(actions, dict):
            # Keep order stable by iterating state.players but ONLY include ones present in dict
            for p in state.players:
                if p.car_id in actions:
                    out.append(decode_idx_to_buttons(actions[p.car_id]))
        elif isinstance(actions, list):
            out = [decode_idx_to_buttons(a) for a in actions]
        else:  # np.ndarray
            a = np.asarray(actions)
            if a.ndim == 1:
                # (N,) where N == number of provided agents
                for i in range(a.shape[0]):
                    out.append(decode_idx_to_buttons(a[i]))
            else:
                # (N, T) or (N, 1) etc.
                for i in range(a.shape[0]):
                    out.append(decode_idx_to_buttons(a[i]))

        # Compact to (8,) when single tick
        out = [v[0] if (v.ndim == 2 and v.shape[0] == 1) else v for v in out]
        return out
