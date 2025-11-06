"""
Global and per-file debug configuration.
"""

# Master switch – controls everything
global_debug_mode = False

# Optional fine-grained flags
debug_obs = False
debug_actions = False
debug_rewards = False
debug_env = False
debug_learning = False # used for training/ppo_learner.py disabling checkpoint load/save
debug_checkpoints = False  # used for training/ppo_learner.py checkpoint summary print
debug_turtled_start = False  # used for mysim/state_setters/turtled_start.py

def dprint(*args, **kwargs):
    """Print only when debugging is enabled."""
    if global_debug_mode or kwargs.pop("force", False):
        print(*args, **kwargs)
