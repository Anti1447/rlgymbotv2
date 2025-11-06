import numpy as np
import gym.spaces
from mysim.action_parsers import ActionParser
from mysim.debug_config import debug_actions, global_debug_mode


class SimpleHybridDiscreteAction(ActionParser):
    """
    SimpleHybridDiscreteAction
    --------------------------
    A discrete lookup-table action parser that links steer and yaw (same value)
    while keeping throttle independent and pitch fully controllable.

    This produces 624 unique control combinations covering ground, aerial, and flip actions.
    The layout is compatible with RLGym / RLBot control order:

        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

    Key design choices:
    - Steer and yaw share the same value for smoother turns.
    - Throttle is independent (forward, reverse, idle).
    - Pitch uses 5 discrete bins for fine aerial/tilt control.
    - Small discrete toggles for jump, boost, and handbrake.
    - Includes a few flip and roll variants for variety.
    """

    _cached_table = None
    def __init__(self):
        super().__init__()
        if SimpleHybridDiscreteAction._cached_table is None:
            SimpleHybridDiscreteAction._cached_table = self._build_table()
            if global_debug_mode or debug_actions:
                print(f"[SimpleHybridDiscreteAction] Built {len(SimpleHybridDiscreteAction._cached_table)} unique actions.")
        self._lookup_table = SimpleHybridDiscreteAction._cached_table

    @property
    def lookup_table(self):
        return self._lookup_table

    # --------------------------------------------------------
    # BUILD TABLE
    # --------------------------------------------------------
    def _build_table(self) -> np.ndarray:
        """
        Builds a discrete lookup table (~950 actions) supporting:
        - Independent throttle control
        - Linked steer/yaw
        - 5-bin pitch (-1..1)
        - 3-bin roll (-1, 0, 1)
        - Jump, boost, handbrake toggles
        Allows simultaneous roll + pitch/yaw for aerial recovery and advanced maneuvers.
        """
        # Base discrete bins
        throttle_bins = [-1, 0, 1]                    # Reverse, idle, forward
        steer_yaw_bins = [-1, -0.5, 0, 0.5, 1]        # Linked steer/yaw
        pitch_bins = [-1, -0.5, 0, 0.5, 1]            # Independent pitch bins
        roll_bins = [-1, 0, 1]                        # Simplified roll control
        jumps = [0, 1]
        boosts = [0, 1]
        handbrakes = [0, 1]

        actions = []

        # 0) Always include an explicit idle
        # Explicit neutrals that matter
        actions.append([0,0,0,0,0,0,0,0])  # idle (you already ensure later, but explicit is fine)
        actions.append([0,0,0,0,0,1,0,0])  # jump-only
        actions.append([0,0,0,0,0,1,0,1])  # jump + handbrake (wavedash pop)
        actions.append([0,0,0, 0.5,0,1,0,0])   # jump + small pitch (helps turtle → upright)
        actions.append([0,0,0,-0.5,0,1,0,0])   # jump + small pitch other way
        actions.append([0,0,0,0, 0.5,1,0,0])   # jump + small roll
        actions.append([0,0,0,0,-0.5,1,0,0])   # jump + small roll
        # (Optional) tiny yaw variants if your contract benefits it

        # 1️⃣ Core ground + aerial actions
        for thr in throttle_bins:
            for steer in steer_yaw_bins:
                for pitch in pitch_bins:
                    yaw = steer  # linked steer/yaw
                    for roll in roll_bins:
                        # skip completely neutral (already included elsewhere)
                        if pitch == yaw == roll == 0:
                            continue
                        for jump in jumps:
                            for boost in boosts:
                                for hb in handbrakes:
                                    actions.append([
                                        thr, steer, pitch, yaw, roll, jump, boost, hb
                                    ])

        # 2️⃣ Add pure roll recovery actions (no throttle/steer)
        for roll in roll_bins:
            if roll != 0:
                actions.append([0, 0, 0, 0, roll, 0, 0, 0])  # torque-only roll
                actions.append([0, 0, 0, 0, roll, 0, 1, 0])  # torque + boost

        # 2b) Add pure pitch/yaw torque-only rows (±1), no jump/boost
        for val in (-1, 1):
            actions.append([0, 0, val, 0, 0, 0, 0, 0])  # pitch-only
            actions.append([0, 0, 0, val, 0, 0, 0, 0])  # yaw-only

        # 3️⃣ Add flip and jump-based variants
        for pitch in [-1, 0, 1]:
            for roll in roll_bins:
                for boost in [0, 1]:
                    # Forward & backward flips
                    actions.append([1, 0, pitch, 0, roll, 1, boost, 0])
                    actions.append([-1, 0, pitch, 0, roll, 1, boost, 1])

        # After deduplication, right before returning
        actions = np.round(np.array(actions, dtype=np.float32), 3)
        unique = np.unique(actions, axis=0)

        # ✅ Ensure an all-zero "idle" action exists
        if not ((unique == 0).all(axis=1)).any():
            unique = np.vstack([unique, np.zeros((1, 8), dtype=np.float32)])

        assert unique.shape[1] == 8, "Invalid action vector length"
        if global_debug_mode or debug_actions:
            print(f"[SimpleHybridDiscreteAction] Built {len(unique)} unique actions (roll-integrated).")
        return unique

    # --------------------------------------------------------
    # GYM INTERFACE
    # --------------------------------------------------------
    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self._lookup_table))

    # --------------------------------------------------------
    # PARSING
    # --------------------------------------------------------
    def parse_actions(self, actions, state=None, shared_info=None, *_, **__):
        arr = np.asarray(actions)
        if arr.ndim == 0:
            idx = np.clip(int(arr), 0, len(self._lookup_table) - 1)
            out = self._lookup_table[idx]
        elif arr.ndim == 1:
            idx = np.clip(arr.astype(int), 0, len(self._lookup_table) - 1)
            out = self._lookup_table[idx]
        elif arr.ndim == 2 and arr.shape[1] == 1:
            idx = np.clip(arr.squeeze(1).astype(int), 0, len(self._lookup_table) - 1)
            out = self._lookup_table[idx]
        elif arr.ndim == 2 and arr.shape[1] == 8:
            out = arr.astype(np.float32)
        else:
            raise ValueError(f"Unexpected action shape: {arr.shape}")

        out = np.atleast_2d(out).astype(np.float32)
        return out
