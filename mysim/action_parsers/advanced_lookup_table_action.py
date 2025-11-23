from typing import Any, Dict
import numpy as np
# old (deprecated and now missing)
# from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgymbotv2.mysim.action_parsers.lookup_table_action import LookupTableAction # use local version
import gym.spaces
from rlgymbotv2.mysim.gamestates import PlayerData, GameState
from rlgym.api import ActionParser, AgentID
from rlgymbotv2.mysim.debug_config import global_debug_mode, debug_actions
from rlgymbotv2.mysim.action_parsers.utils import get_lookup_table_size, debug_print_lut_size, find_lookup_table


@property
def lookup_table(self):
    return self._lookup_table

def _parse_bin(b, endpoint=True):
    if isinstance(b, int):
        b = np.linspace(-1, 1, b, endpoint=endpoint)
    else:
        b = np.array(b)
    return b


def _subdivide(lo, hi, depth=0):
    # Add points to a grid of size [lo, hi] x [lo, hi]
    # alternating between square and diamond steps as in the diamond-square algorithm
    # Basically, if we have:
    # •   •   •
    #
    # •   •   •
    #
    # •   •   •
    # then we add points to the grid like this:
    # •   •   •
    #   •   •
    # •   •   •
    #   •   •
    # •   •   •
    # and then like this:
    # • • • • •
    # • • • • •
    # • • • • •
    # • • • • •
    # • • • • •
    # instead of going straight from the first grid to the third grid
    if depth < 0:
        return
    # Square step
    count = 1 + 2 ** (depth // 2)
    bins, delta = np.linspace(lo, hi, count, retstep=True)
    for nx in bins:
        for ny in bins:
            yield nx, ny
    # Diamond step, just shift the square step diagonally and ignore those that exceed hi
    if depth % 2 == 1:
        bins = bins[:-1] + delta / 2
        for nx in bins:
            for ny in bins:
                yield nx, ny

# def debug_lut_size(self):
#     lut = find_lookup_table(action_parser)  # use the recursive finder we discussed
#     throttles = lut[:, 0]
#     steers = lut[:, 1]

#     print("Throttle min/max:", throttles.min(), throttles.max())
#     print("Steer min/max:", steers.min(), steers.max())

#     print("Num actions with throttle=1:", np.sum(throttles == 1))
#     print("Num actions with throttle=-1:", np.sum(throttles == -1))
#     print("Num actions with abs(steer)>0.5:", np.sum(np.abs(steers) > 0.5))


class AdvancedLookupTableAction(LookupTableAction):
    def __init__(self, throttle_bins: Any = 3,
                 steer_bins: Any = 3,
                 torque_subdivisions: Any = 2,
                 flip_bins: Any = 8,
                 include_stalls: bool = False):
        super().__init__()
        self._lookup_table = self.make_lookup_table(throttle_bins, steer_bins, torque_subdivisions, flip_bins,
                                                    include_stalls)

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([len(self._lookup_table)])
    
    def parse_actions(self, actions, state=None, shared_info=None):
        """
        Map integer action indices to their 8D control vectors.
        Always returns a list of np.ndarrays for wrapper compatibility.
        """
        # Handle dict format (common in multi-agent setups)
        if isinstance(actions, dict):
            parsed = {}
            for agent_id, act in actions.items():
                idx = int(np.squeeze(act))
                idx = np.clip(idx, 0, len(self._lookup_table) - 1)
                parsed[agent_id] = self._lookup_table[idx]
            # Return as list for downstream wrappers
            return [v for v in parsed.values()]

        # Handle numpy array or list
        arr = np.asarray(actions)
        if arr.ndim == 1:
            idxs = np.clip(arr.astype(int), 0, len(self._lookup_table) - 1)
            controls = self._lookup_table[idxs]
        elif arr.ndim == 2 and arr.shape[1] == 1:
            idxs = np.clip(arr.squeeze(1).astype(int), 0, len(self._lookup_table) - 1)
            controls = self._lookup_table[idxs]
        elif arr.ndim == 2 and arr.shape[1] == 8:
            # Already a decoded control vector (passed through another wrapper)
            return [arr.astype(np.float32)]
        else:
            raise ValueError(f"Unexpected action shape: {arr.shape}")

        # Always return list for wrapper compatibility
        return [np.asarray(controls, dtype=np.float32)]


    @staticmethod
    def make_lookup_table(throttle_bins: Any = 3,
                          steer_bins: Any = 3,
                          torque_subdivisions: Any = 2,
                          flip_bins: Any = 8,
                          include_stalls: bool = False):
        # Parse bins
        throttle_bins = _parse_bin(throttle_bins)
        steer_bins = _parse_bin(steer_bins)
        flip_bins = (_parse_bin(flip_bins,
                                endpoint=False) + 1) * np.pi  # Split a circle into equal segments in [0, 2pi)
        if isinstance(torque_subdivisions, int):
            torque_face = np.array([
                [x, y]
                for x, y in _subdivide(-1, 1, torque_subdivisions)
            ])
        else:
            if isinstance(torque_subdivisions, np.ndarray) and torque_subdivisions.ndim == 2:
                torque_face = torque_subdivisions
            else:
                torque_subdivisions = _parse_bin(torque_subdivisions)
                torque_face = np.array([
                    [x, y]
                    for x in torque_subdivisions
                    for y in torque_subdivisions
                ])

        actions = []

        # Ground
        pitch = roll = jump = 0
        for throttle in throttle_bins:
            for steer in steer_bins:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        # if boost == 1 and throttle != 1: #commenting out for now... 10/19/25 DEBUG
                        #     continue
                        yaw = steer
                        actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

        # Aerial
        jump = handbrake = 0
        points = np.array([
            np.insert(p, i, side)
            for i in range(3)  # Determines which axis we select faces from
            for side in (-1, 1)  # Determines which side we select
            for p in torque_face  # Selects where we are on the face
        ])
        points = np.unique(points, axis=0)  # Remove duplicates (corners and edges of the cube)
        for p in points:
            pitch, yaw, roll = p.tolist()
            if pitch == roll == 0 and np.isclose(yaw, steer_bins).any():
                continue  # Duplicate with ground
            for boost in (0, 1):
                throttle = boost
                steer = yaw
                actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])

        # Flips and jumps
        jump = handbrake = 1  # Enable handbrake for potential wavedashes
        yaw = steer = 0  # Only need roll for sideflip
        angles = [np.nan] + [v for v in flip_bins]
        for angle in angles:
            if np.isnan(angle):
                pitch = roll = 0  # Empty jump
            else:
                pitch = np.sin(angle)
                roll = np.cos(angle)
                # Project to square of diameter 2 because why not
                magnitude = max(abs(pitch), abs(roll))
                pitch /= magnitude
                roll /= magnitude
            for boost in (0, 1):
                throttle = boost
                actions.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])
        if include_stalls:
            # Add actions for stalling
            actions.append([0, 0, 0, 1, -1, 1, 0, 1])
            actions.append([0, 0, 0, -1, 1, 1, 0, 1])

        # Convert to numpy and remove floating point errors
        actions = np.round(np.array(actions), 3)

        # 🧹 Remove stationary spin / pure yaw actions (throttle=0, only yaw active)
        mask = ~((actions[:, 0] == 0) & (np.abs(actions[:, 3]) == 1) & (np.sum(np.abs(actions[:, 1:]), axis=1) == 1))
        actions = actions[mask]

        assert len(np.unique(actions, axis=0)) == len(actions), "Duplicate actions found"
        return actions
