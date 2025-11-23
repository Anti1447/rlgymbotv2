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
    """Print when ANY debug flag is active, not only global_debug_mode."""
    if (
        global_debug_mode
        or debug_obs
        or debug_actions
        or debug_rewards
        or debug_env
        or debug_learning
        or debug_checkpoints
        or debug_turtled_start
        or kwargs.pop("force", False)
    ):
        print(*args, **kwargs)