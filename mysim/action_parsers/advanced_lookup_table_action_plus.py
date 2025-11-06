from typing import Any
import numpy as np
import gym.spaces
from mysim.action_parsers.lookup_table_action import LookupTableAction
from mysim.gamestates import PlayerData, GameState


def _parse_bin(b, endpoint=True):
    if isinstance(b, int):
        b = np.linspace(-1, 1, b, endpoint=endpoint)
    else:
        b = np.array(b)
    return b


def _subdivide(lo, hi, depth=0):
    if depth < 0:
        return
    count = 1 + 2 ** (depth // 2)
    bins, delta = np.linspace(lo, hi, count, retstep=True)
    for nx in bins:
        for ny in bins:
            yield nx, ny
    if depth % 2 == 1:
        bins = bins[:-1] + delta / 2
        for nx in bins:
            for ny in bins:
                yield nx, ny


class AdvancedLookupTableActionPlus(LookupTableAction):
    """
    Extended version of the original AdvancedLookupTableAction.
    Builds ~220+ unique Rocket League control combinations
    combining ground, aerial, flip, stall, half-flip, drift and recovery actions.
    """

    def __init__(self, throttle_bins=3, steer_bins=3, torque_subdivisions=2,
                 flip_bins=8, include_stalls=True):
        super().__init__()
        self._lookup_table = self.make_lookup_table(
            throttle_bins, steer_bins, torque_subdivisions, flip_bins, include_stalls
        )

    def get_action_space(self):
        return gym.spaces.MultiDiscrete([len(self._lookup_table)])

    def parse_actions(self, actions, state=None, shared_info=None):
        """
        Converts discrete action indices into 8D control vectors.
        Always returns [np.ndarray(n, 8)] for wrapper compatibility.
        """
        arr = np.asarray(actions)
        idxs = np.clip(arr.astype(int).squeeze(), 0, len(self._lookup_table) - 1)
        controls = self._lookup_table[idxs]

        # Ensure output is 2D (n, 8)
        controls = np.atleast_2d(controls).astype(np.float32)
        return [controls]


    # -----------------------
    # 🔸 MAIN LOOKUP BUILDER
    # -----------------------
    @staticmethod
    def make_lookup_table(throttle_bins=3, steer_bins=3,
                          torque_subdivisions=2, flip_bins=8, include_stalls=True):

        throttle_bins = _parse_bin(throttle_bins)
        steer_bins = _parse_bin(steer_bins)
        flip_bins = (_parse_bin(flip_bins, endpoint=False) + 1) * np.pi

        if isinstance(torque_subdivisions, int):
            torque_face = np.array([
                [x, y] for x, y in _subdivide(-1, 1, torque_subdivisions)
            ])
        else:
            torque_subdivisions = _parse_bin(torque_subdivisions)
            torque_face = np.array([
                [x, y] for x in torque_subdivisions for y in torque_subdivisions
            ])

        actions = []

        # 1️⃣ Ground driving
        pitch = roll = jump = 0
        for throttle in throttle_bins:
            for steer in steer_bins:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        yaw = steer
                        actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

        # 2️⃣ Aerial control
        jump = handbrake = 0
        points = np.array([
            np.insert(p, i, side)
            for i in range(3)
            for side in (-1, 1)
            for p in torque_face
        ])
        points = np.unique(points, axis=0)
        for p in points:
            pitch, yaw, roll = p.tolist()
            for boost in (0, 1):
                throttle = boost
                steer = yaw
                actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

        # 3️⃣ Flips / jumps
        jump = handbrake = 1
        yaw = steer = 0
        angles = [np.nan] + [v for v in flip_bins]
        for angle in angles:
            if np.isnan(angle):
                pitch = roll = 0
            else:
                pitch = np.sin(angle)
                roll = np.cos(angle)
                magnitude = max(abs(pitch), abs(roll))
                pitch /= magnitude
                roll /= magnitude
            for boost in (0, 1):
                throttle = boost
                actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

        # 4️⃣ Optional stalls
        if include_stalls:
            actions.append([0, 0, 0, 1, -1, 1, 0, 1])
            actions.append([0, 0, 0, -1, 1, 1, 0, 1])

        # 5️⃣ NEW: Half-flips & backflips
        for flip_dir in [(1, 0), (-1, 0)]:
            pitch, roll = flip_dir
            actions.append([-1, 0, pitch, 0, roll, 1, 0, 0])  # throttle back
            actions.append([-1, 0, pitch, 0, roll, 1, 0, 1])  # with drift

        # 6️⃣ NEW: Recovery / Aerial alignment
        for pitch in [-1, -0.5, 0, 0.5, 1]:
            for roll in [-1, -0.5, 0, 0.5, 1]:
                actions.append([0, 0, pitch, 0, roll, 0, 0, 0])
                actions.append([0, 0, pitch, 0, roll, 0, 1, 0])

        # 7️⃣ NEW: Drift-steer hybrids
        for steer in [-1, -0.5, 0.5, 1]:
            for throttle in [-0.5, 0.5]:
                actions.append([throttle, steer, 0, steer, 0, 0, 0, 1])

        actions = np.round(np.array(actions), 3)
        mask = ~((actions[:, 0] == 0) & (np.abs(actions[:, 3]) == 1) &
                 (np.sum(np.abs(actions[:, 1:]), axis=1) == 1))
        actions = actions[mask]
        actions = np.unique(actions, axis=0)
        
        # ✅ FIX: reorder pitch/yaw to match RocketSim control order
        # from [thr, steer, pitch, yaw, roll, jump, boost, handbrake]
        # to   [thr, steer, yaw, pitch, roll, jump, boost, handbrake]
        actions = actions[:, [0, 1, 3, 2, 4, 5, 6, 7]]
        
        print(f"[AdvancedLUT+] Generated {len(actions)} unique actions.")
        return actions

