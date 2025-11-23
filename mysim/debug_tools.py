import numpy as np

def debug_controls_sample(action_parser, env):
    from rlgymbotv2.mysim.action_parsers.utils import get_lookup_table_size
    N = get_lookup_table_size(action_parser)
    a = np.random.randint(0, N, size=(1,))
    parsed = action_parser.parse_actions(a, env._prev_state)
    print("\n--- ACTION DEBUG SAMPLE ---")
    print("Raw index:", a)
    print("Decoded controls:", parsed)

def list_turning_actions(lut):
    """Quickly print all left/right steering actions for debugging."""
    import numpy as np
    table = getattr(lut, "lookup_table", getattr(lut, "_lookup_table", None))
    if table is None:
        print("[list_turning_actions] No lookup table found.")
        return
    lefts = table[table[:, 1] < 0]
    rights = table[table[:, 1] > 0]
    print(f"\n[Turning Actions] Left-turn entries: {len(lefts)}, Right-turn entries: {len(rights)}")
    print("Sample left:", lefts[:3])
    print("Sample right:", rights[:3])
    print("---------------------------\n")