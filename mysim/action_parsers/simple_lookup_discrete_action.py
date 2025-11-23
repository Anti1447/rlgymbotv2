import numpy as np
import gym.spaces
from rlgymbotv2.mysim.gamestates import GameState
from rlgymbotv2.mysim.action_parsers import ActionParser
from rlgymbotv2.mysim.debug_config import dprint
from typing import Optional, Dict, Any


class SimpleLookupDiscreteAction(ActionParser):
    """
    Simplified but powerful lookup-style discrete action parser.
    Produces ~100 meaningful Rocket League control combinations.
    Replaces AdvancedLookupTableAction cleanly.
    """

    def __init__(self, debug: bool = False):
        super().__init__()
        self._lookup_table = self._build_table()
        dprint(f"[SimpleLookupDiscreteAction] Built {len(self._lookup_table)} unique actions")

    @property
    def lookup_table(self):
        return self._lookup_table
    # --------------------------------------------------------
    # BUILD ACTION TABLE
    # --------------------------------------------------------
    def _build_table(self) -> np.ndarray:
        """
        Build the discrete lookup table in RLBot/RLGymSim control order:
        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        """
        actions = []

        # ------------------------------
        # 1️⃣ Ground driving (36)
        # ------------------------------
        for throttle in [-1, 0, 1]:
            for steer in [-1, -0.5, 0, 0.5, 1]:
                for boost in [0, 1]:
                    for handbrake in [0, 1]:
                        # RLBot control order (pitch,yaw swapped vs RocketSim)
                        actions.append([
                            throttle, steer, 0, 0, 0, 0, boost, handbrake
                        ])

        # ------------------------------
        # 2️⃣ Basic jumps / flips (20)
        # ------------------------------
        # Each (pitch, roll) direction pair represents a flip direction.
        # Remember: control order = [thr, steer, pitch, yaw, roll, jump, boost, handbrake]
        jump_dirs = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # cardinal flips
            (-0.7, -0.7), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7),  # diagonals
            (0, 0)  # neutral jump
        ]
        for pitch, roll in jump_dirs:
            for boost in [0, 1]:
                actions.append([
                    1, 0, pitch, 0, roll, 1, boost, 0
                ])

        # Handbrake variants for wavedashes
        for pitch, roll in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            actions.append([1, 0, pitch, 0, roll, 1, 0, 1])

        # ------------------------------
        # 3️⃣ Aerial movement (20)
        # ------------------------------
        for pitch in [-1, -0.5, 0, 0.5, 1]:
            for yaw in [-1, 0, 1]:
                for roll in [-0.5, 0.5]:
                    actions.append([
                        0, 0, pitch, yaw, roll, 0, 1, 0
                    ])

        # prune duplicates / limit to ~80 total
        actions = actions[:80]

        # ------------------------------
        # 4️⃣ Specials (10)
        # ------------------------------
        specials = [
            [0, 0, 0, 0, 0, 0, 0, 0],   # idle
            [0, 0, 0, 0, 0, 0, 1, 0],   # boost idle
            [0, 0, 0, 0, 0, 1, 0, 0],   # jump on spot
            [0, 0, 0, 0, 0, 1, 1, 0],   # boost jump
            [0, 0, 1, -1, 1, 1, 0, 1],  # stall 1
            [0, 0, -1, 1, -1, 1, 0, 1], # stall 2
            [-1, 0, 0, 0, 0, 0, 0, 1],  # reverse handbrake turn
            [1, 0, 0, 0, 0, 0, 0, 1],   # forward powerslide
            [0, 0, 1, 0, 0, 0, 1, 1],   # boost spin
            [0, 0, -1, 0, 0, 0, 1, 1]   # reverse spin
        ]
        actions.extend(specials)

        # ------------------------------
        # Deduplicate and finalize
        # ------------------------------
        actions = np.round(np.array(actions, dtype=np.float32), 3)
        unique_actions = np.unique(actions, axis=0)
        assert unique_actions.shape[1] == 8, "Invalid action vector length"
        return unique_actions

    # --------------------------------------------------------
    # GYM INTERFACE
    # --------------------------------------------------------
    import gym

    def get_action_space(self) -> gym.spaces.Space:
        # Previously: gym.spaces.MultiDiscrete([len(self._lookup_table)])
        return gym.spaces.Discrete(len(self._lookup_table))

    # --------------------------------------------------------
    # PARSING
    # --------------------------------------------------------  
    def parse_actions(self, actions, state, shared_info=None, *_, **__):
        arr = np.asarray(actions)

        # Discrete policy -> 1D array of indices
        if arr.ndim == 0:
            idxs = np.clip(int(arr), 0, len(self._lookup_table)-1)
            out = self._lookup_table[idxs]
        elif arr.ndim == 1:
            idxs = np.clip(arr.astype(int), 0, len(self._lookup_table)-1)
            out = self._lookup_table[idxs]
        elif arr.ndim == 2 and arr.shape[1] == 1:
            idxs = np.clip(arr.squeeze(1).astype(int), 0, len(self._lookup_table)-1)
            out = self._lookup_table[idxs]
        elif arr.ndim == 2 and arr.shape[1] == 8:
            out = arr.astype(np.float32)  # already decoded vector(s)
        else:
            raise ValueError(f"Unexpected action shape: {arr.shape}")

        # Always list-of-arrays for wrapper chain
        if out.ndim == 1:
            return [out]
        return [out[i] for i in range(out.shape[0])]
