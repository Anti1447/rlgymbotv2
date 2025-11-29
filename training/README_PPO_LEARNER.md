# README — RLGymBot_v2 PPO Training Pipeline

*(Generated from internal code comments — captures all important design notes)*

## Overview

This script (`ppo_learner.py`) is the main entry point for training a Rocket League bot using:

* **RocketSim** physics
* **RLGym-Sim** environment API
* **RLGym-PPO** training stack
* Custom **reward functions**, **action parsers**, **state setters**, and **observation builders**

It handles:

* creating vectorized environments
* launching PPO workers
* checkpoint management
* milestone saving
* self-play (optional)
* mechanics-lane training (optional)
* turtle-recovery metrics
* debug utilities
* rendering via RocketSimVis

---

# 1. Training Configuration Overview

## Core knobs

These affect how PPO trains:

| Setting                | Description                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `N_PROC_TRAIN`         | Number of parallel environments (default 64). Lower if running on lower RAM.        |
| `SAVE_EVERY`           | Save PPO checkpoint every *N* timesteps. Recommended = batch size = iteration size. |
| `POLICY_LAYER_SIZES`   | Hidden layer sizes for the policy.                                                  |
| `CRITIC_LAYER_SIZES`   | Hidden layers for critic.                                                           |
| `POLICY_LEARNING_RATE` | LR for policy. Tune based on bot skill.                                             |
| `CRITIC_LEARNING_RATE` | LR for critic. Usually same as policy LR.                                           |
| `SELFPLAY_ENABLED`     | Enables self-play using a frozen opponent snapshot.                                 |
| `MILESTONE_INTERVAL`   | Save milestone checkpoints (full copies) every X timesteps.                         |

### Learning rate guidance (from comments)

* Early bot that **can’t score yet**: `2e-4`
* Bot that is shooting/scoring: `1e-4`
* Bot learning dribbles, flicks, advanced play: `0.8e-4` or lower

---

# 2. Phases — High-Level Training Curriculum

The training is broken into phases, each with different rewards:

| Phase | Name                        | Goal                                                            |
| ----- | --------------------------- | --------------------------------------------------------------- |
| **1** | BallTouching                | learn movement + ball touches                                   |
| **2** | BronzeSkills1               | learn clean shots, scoring, avoid own goals                     |
| **3** | SilverSkills1               | strong shots, purposeful touches, shooting angles, jump touches |
| **4** | SilverSkills2 / GoldSkills1 | early aerial shaping, boost discipline, shot consistency        |

Phase determines:

* reward weights
* mechanics-lane enablement
* logging labels
* curriculum difficulty

---

# 3. Mechanics Lane (Advanced Mechanics Training)

Disabled until **PHASE ≥ 4**, to prevent overwhelming the bot early.

When enabled:

* a percentage (`MECHANICS_WEIGHT`) of environments use `build_mechanics_env()`
* mechanics envs can later include:

  * wall setups
  * air dribble setups
  * flip reset setups
  * jump challenges

For now, the mechanics env is identical to the main env but structured to diverge later.

---

# 4. Self-Play System

Disabled unless `SELFPLAY_ENABLED = True`.

### How it works:

1. The system maintains an `opponent_pool` that tracks best-performing checkpoints.
2. A **frozen opponent** is loaded (copy of past policy).
3. Each worker env can include that frozen opponent.
4. At every **milestone**, the system:

   * refreshes the pool
   * detects a newer, stronger checkpoint
   * rebuilds the frozen policy **in place**
   * updates all workers via `EnvFactory`

This creates a progression:

* Bot competes against earlier versions
* Frozen opponent only updates at milestones (stable, reliable opponent)

---

# 5. Reward System Notes

Rewards are fully documented per phase:

### Key reward components used:

* **EventReward** (goals, touches, concessions)
* **VelocityBallToGoalReward** (shot alignment)
* **StrongHitReward** (powerful touches > weak taps)
* **FaceBallReward** (good orientation)
* **JumpShotReward** (purposeful aerial touches)
* **BasicShotReward** (hitting toward goal)
* **NegativeVelocityTowardOwnGoalReward** (avoid own-goaling)
* **RecoveryAndLandingReward** (good landings + anti-turtle shaping)
* **LiuDistanceBallToGoalReward** (shot distance shaping)
* **InAirReward** (small bonus for controlled aerial behavior)
* **SaveBoostReward** (boost discipline)
* **CollectBoostReward** (boost economy in later phases)

The reward function is **phase-aware**, gradually introducing learning tasks in a curriculum.

---

# 6. Observation Builder Notes

Using `AdvancedObsPlus` which includes:

* car state
* ball state
* local & global context
* nearest pads to car
* nearest pads to ball
* previous action
* stacking (set to 1 for now; expand later to 3–4)

This provides strong stability and great early learning acceleration.

---

# 7. Action Parser Notes

Default parser:

* **SimpleHybridDiscreteAction** → LUT
* **StateAwareLUTWrapper** → fallback if illegal action
* **FinalRocketSimAdapter** → expands LUT to RocketSim format
* Tick-skip wrappers:

  * `ExpandForRocketSim`
  * `CollapseToSingleTick`
  * `ClipActionWrapper`
  * `StickyButtonsWrapper`
  * `RepeatAction`

### Important debugging notes:

* Confirm LUT contains left + right steering
* Confirm pitch/yaw/roll coverage
* Ensure fallback index is valid (all-zero row)

The code prints:

* unique steer values
* unique pitch values
* unique yaw values
* # of turning actions

to validate the LUT.

---

# 8. State Setters

### Default:

`RandomState(random_ball_speed=True, random_car_speed=True, cars_on_ground=False)`

Produces diverse training data:

* Random ball position / velocity
* Random player poses
* 50% aerial states
* Good for resilience & robust learning

### Optional:

* **TurtledStart** (rich turtle-recovery curriculum)
* **Mechanics state setters** (future expansion)

---

# 9. Terminal Conditions

Used:

* **GoalScoredCondition**
* **NoTouchTimeoutCondition**
* **TimeoutCondition**

NoTouchTimeout:

* ~15 seconds with tick-skip applied
  Gyms reset smoothly & aggressively for data collection.

---

# 10. Turtle Recovery Logging

Custom `TurtleRecoveryLogger`:

* Detects turtle events
* Measures if bot recovers within a sliding window
* Logs success/fail + avg tick count
* Logs to console and optionally to W&B

This is extremely useful for debugging recovery/flip logic.

---

# 11. Milestones System

A “milestone” is a **full snapshot** stored independently of normal PPO checkpoints.

Useful for:

* reproducible experiments
* rolling back curriculum
* self-play reference opponents
* human analysis

### Key points:

* Created every `MILESTONE_INTERVAL` steps
* Full directory copy of the PPO checkpoint
* Meta JSON included:

  * average reward
  * entropy
  * KL divergence
  * timestamp
  * phase state
* Stored under:

  * `data/checkpoints/milestones/PhaseX-Name/###M/`

---

# 12. Checkpointing Rules

### Auto-resume logic tries in priority order:

1. **Manual override** (`CUSTOM_CKPT_PATH`)
2. **Interactive choose milestone**
3. **Auto-select most recent compatible run**
4. **If none matched → new run**

Compatibility includes:

* observation shape
* policy architecture
* critic architecture

---

# 13. Debugging Tools

When any debug flag is active:

* Action parsing prints
* Turning actions printed
* Turtle probe runs on first step
* Reset state analyzed
* Control order shown
* Reward builder traces

Debug flags:

* `global_debug_mode`
* `debug_actions`
* `debug_learning`
* `debug_checkpoints`
* `debug_turtled_start`

---

# 14. Rendering (RocketSimVis)

Environment gets a `.render()` method added at runtime:

```python
env.render()  →  sends state to RocketSimVis
```

This is extremely fast, and great for:

* validating tick-skip
* debugging mechanical motion
* verifying aerial setups

---

# 15. Multiprocessing Notes

Important settings:

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
```

This prevents CPU contention across many workers.

Additionally:

```
multiprocessing.set_start_method("spawn", force=True)
```

Is required for Windows + torch + pickling.

---

# 16. Important Architectural Notes

Taken from inline code comments:

### ✔ Avoid nested functions for env creation

`multiprocessing` cannot pickle nested functions (`make_env`), so env creation must be wrapped in:

```python
class EnvFactory:
    def make_env(...)
```

This ensures correct spawning of PPO workers.

### ✔ Frozen opponent must be retrieved via getter

Because PPO workers run in separate processes, they need:

* a picklable callable
* not a closure capturing a mutable object

Thus `EnvFactory(get_frozen_opponent)`.

### ✔ State-aware LUT fallback

Guarantees:

* LUT never sends invalid actions
* training does not collapse due to illegal torque combinations
* reduces spikes in early learning instability

### ✔ Reset behavior verified at each debug reset

This helps catch:

* bugged initial states
* invalid car poses
* explosive flip setups
* turtled state generation issues

---

# 17. When to Enable Self-Play

Recommended to wait until:

* bot can **score reliably**
* bot can **recover consistently**
* bot performs stable, non-random touches

Enabling too early causes:

* collapse
* inability to get clean hits
* bad learning signal
* opponent always “wins”

---

# 18. When to Enable Mechanics Lane

Enable only after:

* strong shot accuracy (Phase ≥ 4)
* stable landing behavior
* consistent hitting and dribbling

Mechanics lane is a **booster**, not a foundation.

---

# 19. System Requirements

For 64 workers:

* **32–64 GB RAM**
* **Strong CPU** (8–16 cores)
* **CUDA GPU** (training device)

For debugging or light training:

* set `N_PROC_TRAIN = 8`
* render enabled is fine

---

# 20. Launching Training

Typical usage:

```
python -m rlgymbotv2.training.ppo_learner --config configs/training.yaml
```

Make sure:

* RocketSimVis is running (optional)
* GPU is configured
* checkpoint directory is writable

---

# 21. Recommended Future Enhancements

Based on inline TODO comments:

* Add true mechanics lane envs
* Add evaluation script `run_selfplay_eval`
* Enhance milestone visualization
* Add boost economy shaping earlier
* Add adaptive batch sizing based on GPU load
* Add hard reset logic for runaway workers
