"""
Utility helpers for working with action parsers and lookup tables.
"""

from gym.spaces import Discrete, MultiDiscrete
import numpy as np

def get_lookup_table_size(action_parser) -> int:
    """
    Returns the number of discrete actions represented by a lookup-table-based parser.
    Works through any wrapper layers by using the parser's exposed action space.
    """
    space = action_parser.get_action_space()  # wrappers should pass this through

    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, MultiDiscrete):
        # Many LUT parsers use MultiDiscrete([len(lookup_table)])
        return int(space.nvec[0])

    raise TypeError(f"Unsupported or non-discrete action space: {space}")

def debug_print_lut_size(action_parser):
    try:
        size = get_lookup_table_size(action_parser)
        print(f"[debug] lookup_table size: {size}")
    except Exception as e:
        print(f"[debug] Failed to read LUT size: {e}")

def find_lookup_table(obj):
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        # direct
        for name in ("lookup_table", "_lookup_table"):
            tab = getattr(obj, name, None)
            if tab is not None:
                return tab
        # descend
        obj = getattr(obj, "inner", None)
    return None
# Example usage:
# lookup_table = find_lookup_table(action_parser)
# a = np.random.randint(0, len(lookup_table), size=(1,))

def find_forward_fallback_idx(lut) -> int:
    """
    Finds the index of a 'safe forward' control in a LookupTableAction.
    Prioritizes throttle=1, near-zero steering, and no jump/boost/handbrake.
    """
    table = lut.lookup_table
    forward_candidates = np.where(
        (table[:, 0] == 1) & 
        (np.abs(table[:, 1]) < 0.1) &
        (table[:, 5] == 0) &
        (table[:, 6] == 0) &
        (table[:, 7] == 0)
    )[0]

    if len(forward_candidates) > 0:
        idx = int(forward_candidates[0])
    else:
        idx = int(np.where((table == 0).all(axis=1))[0][0])

    return idx

def rlbot_to_rocketsim(v):
    t, s, p, y, r, j, b, h = v
    return [t, s, y, p, r, j, b, h]
