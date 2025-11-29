"""
rlgymbotv2/training/ppo_config.py
---------------------------------

Config layout / guidelines

- All values here are module-level constants in UPPER_SNAKE_CASE.
- Group related settings into sections:

  1) Paths / filesystem
     - BASE_DIR, CHECKPOINT_ROOT, CHECKPOINT_MILESTONE_ROOT, etc.

  2) Run mode / control
     - START_FRESH, INTERACTIVE_RESUME, CUSTOM_CKPT_PATH, PHASE, LOG_PHASE.

  3) Environment / self-play
     - TEAM_SIZE, PRIMARY_BLUE_INDEX, PRIMARY_ORANGE_INDEX
     - SELFPLAY_ENABLED, SELFPLAY_SYNC_INTERVAL, SELFPLAY_UPDATE_MODE, etc.

  4) Network / optimization
     - POLICY_LAYERS, CRITIC_LAYERS
     - POLICY_LEARNING_RATE, CRITIC_LEARNING_RATE
     - LR_DECAY_STEPS, MIN_POLICY_LR, MIN_CRITIC_LR, N_PROC_TRAIN, N_PROC_DEBUG.

  5) Mechanics / curriculum
     - MECHANICS_ENABLED, MECHANICS_MIN_PHASE, MECHANICS_WEIGHT.

  6) Evaluation / Elo / league play
     - ENABLE_ELO_SKILL_GAP, ELO_SKILL_GAP_EPISODES, ELO_SKILL_GAP_MAX_STEPS
     - SELFPLAY_OPPONENT_SELECTION, SELFPLAY_ELO_TEMPERATURE, etc.

When adding a new constant:
- Put it in the most relevant section.
- Add a short comment describing how it’s used.
- If other modules need it, explicitly add it to their import lists
  (for example, to ppo_learner.py or selfplay/manager.py).
- Avoid importing training modules back into this file to keep
  dependencies acyclic.
"""

# Revisions:
# 2025-10-20 – Rev 1.0.0 – Initial training config scaffold (paths, PPO hyperparams, mechanics lane). – NJH
# 2025-11-27 – Rev 1.1.0 – Added Strategy C / Elo skill-gap config and league opponent selection knobs. – NJH
# 2025-11-28 – Rev 1.2.0 – Documented config layout and grouped constants by category for future expansion. – NJH


from pathlib import Path

# === PATHS / CHECKPOINTS ===

# Base directory for data/checkpoints (inside the rlgymbotv2 package)
BASE_DIR = Path(__file__).resolve().parent.parent  # .../rlgymbotv2
CHECKPOINT_ROOT = BASE_DIR / "data" / "checkpoints"
CHECKPOINT_MILESTONE_ROOT = CHECKPOINT_ROOT / "milestones"

# NOTE:
# Debug printing of these paths should be done in ppo_learner.py
# so this module stays "pure" and avoids circular imports.


# === TRAINING CONFIG ===

START_FRESH = False

# If True, ask interactively which checkpoint / milestone to resume from.
INTERACTIVE_RESUME = True

# Optional: manually override which checkpoint to resume from.
# Example:
# CUSTOM_CKPT_PATH = (
#     BASE_DIR / "data" / "checkpoints" /
#     "milestones" / "Phase1-BallTouching" / "030M"
# )
CUSTOM_CKPT_PATH = None

# Self-play toggles
SELFPLAY_ENABLED = True          # only turn this on once bot can consistently score
SELFPLAY_SYNC_INTERVAL = 10_000_000  # (unused for now, but kept for future)

# Milestones & checkpoints
# MILESTONE_INTERVAL = 5_000_000  # every 5 million steps (production)
MILESTONE_INTERVAL = 100_000      # just for regenerating / experimentation
SAVE_EVERY = 50_000               # save checkpoint every 50k steps

# Self-play mode: "auto" (gated by metrics) or "manual" (you pick snapshots)
SELFPLAY_UPDATE_MODE = "manual"
MANUAL_FROZEN_DIR = CHECKPOINT_ROOT / "manual_frozen"


# Training phase (1 = ball touching, 2 = shooting & scoring, etc.)
PHASE = 1

# Human-readable phase name for logs / milestone naming
if PHASE == 1:
    LOG_PHASE = "BallTouching"
elif PHASE == 2:
    LOG_PHASE = "BronzeSkills1"
elif PHASE == 3:
    LOG_PHASE = "SilverSkills1"
elif PHASE == 4:
    LOG_PHASE = "SilverSkills2"
else:
    LOG_PHASE = f"Phase{PHASE}"


# === MATCH FORMAT (1v1 / 2v2 / 3v3 scaffolding) ===

# Number of players per team in the sim (1, 2, or 3).
TEAM_SIZE = 1

# Index of the primary learning blue agent in the obs/action ordering.
PRIMARY_BLUE_INDEX = 0

# Index of the primary frozen orange agent (first orange after all blues).
PRIMARY_ORANGE_INDEX = TEAM_SIZE  # first orange after blues

# === ELO / Skill-gap evaluation (Strategy C wiring) ===
ENABLE_ELO_SKILL_GAP = False  # flip to True when ready
ELO_SKILL_GAP_EPISODES = 64
ELO_SKILL_GAP_MAX_STEPS = 3000

# Self-play / league opponent selection mode:
#   "steps"      – newest-by-steps (existing behavior)
#   "elo_best"   – always pick highest-Elo snapshot
#   "elo_sample" – sample snapshots proportional to Elo
SELFPLAY_OPPONENT_SELECTION = "steps"  # bump to "elo_best" or "elo_sample" later
SELFPLAY_ELO_TEMPERATURE = 100.0       # only used when elo_sample
# tuning guidance for SELFPLAY_ELO_TEMPERATURE: lower = greedier, higher = flatter.


# === PPO HYPERPARAMETERS ===

N_PROC_DEBUG = 2
N_PROC_TRAIN = 16        # 64 may be too high for this machine

POLICY_LAYERS = [2048, 2048, 1024, 1024]
CRITIC_LAYERS = [2048, 2048, 1024, 1024]

# === Learning Rates ===
POLICY_LEARNING_RATE = 2e-4
CRITIC_LEARNING_RATE = 2e-4
# Bot that can't score yet:            2e-4
# Bot that can shoot & score:          1e-4
# Learning advanced mechanics, etc.:   0.8e-4 or lower


# === LR schedule (linear decay) ===

# Decay both policy and critic LR from initial -> min over this many env steps.
LR_DECAY_STEPS = 50_000_000
MIN_POLICY_LR = 8e-5
MIN_CRITIC_LR = 8e-5


# === Mechanics lane (mixed training) ===
# Mechanics lane: special envs focused on advanced mechanics (air dribbles, etc.).
# We gate it by phase so it only appears once the bot is more capable.

MECHANICS_MIN_PHASE = 4         # don’t spawn mechanics envs before this phase
MECHANICS_WEIGHT = 0.2          # 20% of envs will be mechanics when enabled
MECHANICS_ENABLED = PHASE >= MECHANICS_MIN_PHASE
