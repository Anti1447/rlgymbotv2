"""
ppo_learner.py
--------------
RocketSim + RLGym PPO training entry point.
Handles environment creation, checkpoint management, and milestone saving.

Author: Nathan Hafey (NJH)
Revision: 1.0.0 – 2025-10-10
"""
# Imports
import numpy as np
from mysim.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from pathlib import Path
from mysim.reward_functions.common_rewards import *
from rlgym_ppo import Learner
import json
import re
from datetime import datetime
import torch, os
from mysim.action_parsers.utils import get_lookup_table_size, find_lookup_table
from mysim.debug_config import global_debug_mode, debug_actions, debug_learning, debug_checkpoints, debug_turtled_start
from mysim.training_utils.checkpoint_utils import (
    make_run_dir, latest_checkpoint_folder, read_meta, write_meta, shapes_match, summarize_checkpoints, choose_checkpoint
)
from mysim.training_utils.milestone_utils import make_milestone_dir, promote_to_release, choose_milestone
from mysim.debug_tools import debug_controls_sample
from mysim.common_values import CONTROL_ORDER
from mysim.action_parsers.advanced_lookup_table_action_plus import AdvancedLookupTableActionPlus
from mysim.action_parsers.simple_discrete_hybrid_action import SimpleHybridDiscreteAction



if global_debug_mode or debug_actions or debug_learning or debug_checkpoints or debug_turtled_start:
    print("Debug Mode Active")

# Constants
CHECKPOINT_ROOT = Path("data/checkpoints")
CHECKPOINT_MILESTONE_ROOT = CHECKPOINT_ROOT / "milestones"

# === TRAINING CONFIG ===
START_FRESH = False
INTERACTIVE_RESUME = True  # Set to True to choose from available checkpoints interactively
CUSTOM_CKPT_PATH = None
MILESTONE_INTERVAL = 5_000_000  # every 5 million steps
SAVE_EVERY = 50_000
N_PROC_DEBUG = 1
N_PROC_TRAIN = 32 # 64 is too high for this computer's RAM/GPU
POLICY_LAYERS = [1024, 1024, 512, 512]
CRITIC_LAYERS = [1024, 1024, 512, 512]

# === Learning Rates ===
POLICY_LEARNING_RATE = 2e-4  # policy learning rate
CRITIC_LEARNING_RATE = 2e-4  # critic learning rate
#Bot that can't score yet: 2e-4
#Bot that is actually trying to score on its opponent: 1e-4
#Bot that is learning outplay mechanics (dribbling and flicking, air dribbles, etc.): 0.8e-4 or lower
PHASE = 3  # Training phase (1 = ball touching, 2 = shooting & scoring)
# Optional: manually override which checkpoint to resume from
# CUSTOM_CKPT_PATH = "data/checkpoints/milestones/Phase1-BallTouching/030M"
# Example:
# CUSTOM_CKPT_PATH = "data/checkpoints/rlgym-ppo-run-20251019-013412/checkpoints/ckpt_150000"
CUSTOM_CKPT_PATH = None  # Set to None to disable manual override
LOG_PHASE = ""
if PHASE == 1:
    LOG_PHASE = "BallTouching"
elif PHASE == 2:
    LOG_PHASE = "BronzeSkills1"
elif PHASE == 3:
    LOG_PHASE = "SilverSkills1"
elif PHASE == 4:
    LOG_PHASE = "SilverSkills2"

if global_debug_mode or debug_learning:
    summarize_checkpoints(CHECKPOINT_MILESTONE_ROOT)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)

# --- Turtle recovery metrics logger (drop-in) ---
class TurtleRecoveryLogger(MetricsLogger):
    """
    Aggregates turtle events across workers without changing behavior.
    Success = within `window` ticks after turtle detection, the car is on ground with up_z > 0.6.
    """
    def __init__(self, agent_index: int = 0, window: int = 45):
        super().__init__()
        self.agent_index = int(agent_index)
        self.window = int(window)

    def _collect_metrics(self, game_state):
        # Minimal fields we need per tick; keep it lightweight & picklable.
        p = game_state.players[self.agent_index].car_data
        on_g = bool(getattr(p, "on_ground", False))
        up_z = float(p.up()[2])
        return (on_g, up_z)

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        """
        collected_metrics: list of tuples (on_g, up_z) across timesteps (per worker, per rollout)
        We scan for turtle starts and outcomes within `window` ticks.
        """
        success = 0
        fail = 0
        times = []

        i = 0
        T = len(collected_metrics)
        while i < T:
            on_g, up_z = collected_metrics[i]
            # Turtle start = grounded and up_z below threshold
            if on_g and up_z < -0.2:
                # Look ahead window
                recovered = False
                for dt in range(1, min(self.window + 1, T - i)):
                    on_g2, up_z2 = collected_metrics[i + dt]
                    if on_g2 and up_z2 > 0.6:
                        success += 1
                        times.append(dt)
                        recovered = True
                        break
                if not recovered:
                    fail += 1
                # Move past this window to avoid double-counting
                i += self.window
            else:
                i += 1

        # Summaries
        total = success + fail
        avg_ticks = (sum(times) / len(times)) if times else float('nan')

        # Console for fast feedback
        if total > 0:
            print(f"[turtle] events={total}  success={success}  fail={fail}  "
                  f"avg_ticks={avg_ticks:.1f}  steps={cumulative_timesteps:,}")

        # Optional: log to Weights & Biases if enabled
        if wandb_run is not None and total > 0:
            wandb_run.log({
                "turtle/events": total,
                "turtle/success": success,
                "turtle/fail": fail,
                "turtle/avg_ticks": avg_ticks,
                "turtle/window": self.window,
                "Cumulative Timesteps": cumulative_timesteps
            })

# --- Turtle probe helpers ---
def is_turtled(car_data, up_thresh=-0.2):
    on_g = bool(getattr(car_data, "on_ground", False))
    up_z = float(car_data.up()[2])
    return on_g and (up_z < up_thresh), on_g, up_z


def build_reward_function(phase=PHASE):
    from mysim.reward_functions import CombinedReward
    from mysim.reward_functions.common_rewards import (
        EventReward, SpeedTowardBallReward, FaceBallReward, VelocityReward, InAirReward,
        VelocityBallToGoalReward, StrongHitReward, RecoveryAndLandingReward, BadOrientationPenalty
    )
    if global_debug_mode:
        print(f"[reward] Building reward function for phase {phase}")
    if phase == 1:
        return CombinedReward.from_zipped(
            (EventReward(touch=1), 50.0),
            (SpeedTowardBallReward(), 5.0),
            (FaceBallReward(), 1.0),
            (RecoveryAndLandingReward(), 0.4),
            (VelocityReward(), 0.3),
            (BadOrientationPenalty(), 0.2),
            (InAirReward(), 0.15)
        )
    elif phase == 2: # Bronze Skills 1. Encourage shooting and scoring. Penalize own-goal/conceding goals.
        return CombinedReward.from_zipped(
            (EventReward(goal=1), 20.0),
            (EventReward(concede=1), -10.0), # punish goal conceded
            (EventReward(touch=1), 3.0),
            (VelocityBallToGoalReward(), 5.0),
            (SpeedTowardBallReward(), 2.0),
            (FaceBallReward(), 0.5),
            (VelocityReward(), 0.3),
            (RecoveryAndLandingReward(), 0.2),
            (BadOrientationPenalty(), 0.2),
            (InAirReward(), 0.15)
        )
    elif phase == 3: # Silver Skills 1. More shooting, more scoring, stronger hits. Begin adding jump shots and dribbling. Add boost rewards.
        return CombinedReward.from_zipped(
            (EventReward(goal=1), 20.0),
            (EventReward(concede=1), -10.0), # punish goal conceded
            (VelocityBallToGoalReward(), 8.0),
            (StrongHitReward(), 6.0),
            (JumpShotReward(), 4.0),
            (SpeedTowardBallReward(), 2.0),
            (DribblingReward(), 2.0),
            (FaceBallReward(), 0.5),
            (SaveBoostReward(), 0.5),
            (StealBoostReward(), 0.3),
            (VelocityReward(), 0.3),
            (BadOrientationPenalty(), 0.15),
            (CollectBoostReward(), 0.15),
            (InAirReward(), 0.1),
            (RecoveryAndLandingReward(), 0.1)
        )
    elif phase == 4: # Silver Skills 2. Even more shooting and scoring emphasis. Stronger hits. More dribbling and flicks.
        return CombinedReward.from_zipped(
            (EventReward(goal=1), 25.0),
            (EventReward(concede=1), -12.0), # punish goal conceded
            (VelocityBallToGoalReward(), 5.0),
            (LiuDistanceBallToGoalReward(), 5.0),
            (BasicAerialReward(), 5.0),
            (BasicWallHitReward(), 5.0),
            (StrongHitReward(), 4.0),
            (JumpShotReward(), 3.0),
            (BasicShotReward(), 3.0),
            (SpeedTowardBallReward(), 3.0),
            (DribblingReward(), 2.0),
            (FaceBallReward(), 0.7),
            (SaveBoostReward(), 0.7),
            (StealBoostReward(), 0.5),
            (VelocityReward(), 0.4),
            (BadOrientationPenalty(), 0.1),
            (CollectBoostReward(), 0.1),
            (InAirReward(), 0.1),
            (RecoveryAndLandingReward(), 0.1)
        )


def build_rocketsim_env():
    import rlgym_sim
    from mysim.reward_functions import CombinedReward

    # from rlgym_sim.utils.reward_functions import CombinedReward
    from mysim.reward_functions.common_rewards import (
        VelocityPlayerToBallReward,
        VelocityBallToGoalReward,
        EventReward,
        FaceBallReward,
        InAirReward,
        StrongHitReward,
        SaveBoostReward,
        AlignBallGoal,
        BoostPickupReward,
        SpeedTowardBallReward,
        RewardIfBehindBall,
        BasicShotReward,
        SaveReward,
        JumpShotReward,
        PossessionReward,
        LiuDistanceBallToGoalReward,
        LiuDistancePlayerToBallReward,
        PunishIfInNet,
        StealBoostReward
    )
    from mysim.obs_builders import DefaultObs
    from mysim.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
    from mysim.reward_functions.common_rewards.conditional_rewards import GoalIfTouchedLastConditionalReward
    from mysim import common_values
    from mysim.state_setters import RandomState
    from mysim.action_parsers.discrete_act_2 import DiscreteAction2
    from mysim.action_parsers import AdvancedLookupTableAction
    from mysim.action_parsers.continuous_act import ContinuousAction
    from mysim.action_parsers.wrappers.clip_action_wrapper import ClipActionWrapper
    from mysim.action_parsers.wrappers.sticky_buttons_wrapper import StickyButtonsWrapper
    from mysim.action_parsers.wrappers.state_aware_lut_wrapper import StateAwareLUTWrapper
    from mysim.action_parsers.wrappers.repeat_action import RepeatAction
    from mysim.action_parsers.wrappers.collapse_to_single_tick import CollapseToSingleTick
    from mysim.action_parsers.wrappers.expand_to_tick_skip import ExpandToTickSkip
    from mysim.action_parsers.wrappers.expand_for_rocketsim import ExpandForRocketSim
    from mysim.action_parsers.wrappers.final_rocketsim_adapter import FinalRocketSimAdapter
    from mysim.action_parsers.utils import find_forward_fallback_idx


    spawn_opponents = False
    game_tick_rate = 120
    tick_skip = 8
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 15
    no_touch_timeout_ticks = int(round(no_touch_timeout_seconds * game_tick_rate / tick_skip))
    game_timeout_seconds = 300
    game_timeout_ticks = int(round(game_timeout_seconds * game_tick_rate / tick_skip))

    lut = SimpleHybridDiscreteAction()

    # Use the first all-zeros row as fallback
    idle_idx = int(np.where((lut.lookup_table == 0).all(axis=1))[0][0])

    safe_lut = StateAwareLUTWrapper(lut, fallback_index=idle_idx)
    action_parser = FinalRocketSimAdapter(safe_lut)
    # action_parser = DiscreteAction2()
    # action_parser = FinalRocketSimAdapter(DiscreteAction())
    # if global_debug_mode or debug_actions:
    #     lut = find_lookup_table(action_parser)

    #     if lut is not None and global_debug_mode or debug_actions:
    #         from mysim.debug_tools import list_turning_actions
    #         list_turning_actions(lut)
    #         print("Unique steer values:", np.unique(lut[:, 1]))
    #         print("Unique pitch values:", np.unique(lut[:, 2]))
    #         print("Unique yaw values:",   np.unique(lut[:, 3]))
    #         # steer both directions
    #         assert (lut[:,1]<0).any() and (lut[:,1]>0).any(), "No bidirectional steer!"
    state_setter = None # initialize variable
    state_setter = RandomState(True, True, False)
    # if not debug_turtled_start:
    #     state_setter = RandomState(True, True, False)
    # else:
    #     from mysim.state_setters.turtled_start import TurtledStart
    #     state_setter = TurtledStart(
    #         z=20.0,           # a bit above ground so physics settles cleanly
    #         yaw_random=True,  # learn recovery from multiple orientations
    #         spawn_opponents=False
    #     )
    terminal_conditions = [GoalScoredCondition(), NoTouchTimeoutCondition(no_touch_timeout_ticks), 
                                       TimeoutCondition(game_timeout_ticks)]

    # 10-12-25 Basic starting reward function. Get the bot to touch the ball. Phase 1.
    reward_fn = build_reward_function(phase=PHASE)

    from mysim.obs_builders.advanced_obs_plus import AdvancedObsPlus
    obs_builder = AdvancedObsPlus( #added 10/12/25
        max_allies=2,             # adjust to your team sizes
        max_opponents=3,          # adjust to your scrim sizes
        k_nearest_pads_me=4,
        k_nearest_pads_ball=2,
        stack_size=1,             # keep 1 for now; try 3–4 later
        include_prev_action=True, # matches your old behavior
    )
    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,          # number of controlled blue agents
                         spawn_opponents=spawn_opponents,         # add dummy opponents for realism
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)
    obs, info = env.reset(return_info=True)

    # --- after env.reset()
    if global_debug_mode or debug_actions:
        me0 = env._prev_state.players[0].car_data
        ok, on_g, up_z = is_turtled(me0)
        print("[reset] on_ground:", on_g, " up_z:", f"{up_z:.2f}", " turtled?", ok)
        print("Controlled agents (spectator_ids):", len(env._match._spectator_ids))
        debug_controls_sample(action_parser, env) # This confirms whether your decoded LUT entries actually include throttle/boost patterns that make sense. 10/19/25 NJH
        tab = find_lookup_table(action_parser)
        if tab is None:
            print("[list_turning_actions] No lookup table found.")
        else:
            lefts  = tab[tab[:,1] < 0]
            rights = tab[tab[:,1] > 0]
            print(f"[Turning] Left: {len(lefts)}  Right: {len(rights)}")
        
    if global_debug_mode:
        print("[control order in use]", CONTROL_ORDER)




    # The most recent GameState lives directly on the Gym
    if global_debug_mode or debug_actions:
        state = env._prev_state
        if state is not None:
            print("Players in match:", len(state.players))
        else:
            print("Warning: env._prev_state is None (no snapshot yet)")

    if global_debug_mode or debug_actions or debug_turtled_start:
        print("Action space:", env.action_space)
        a = env.action_space.sample()
        print("Sample action:", a)
        print("\n--- DEBUG ACTION SAMPLE ---")
        print("Action input to env.step:", a)
        parsed = env._match._action_parser.parse_actions(a, env._prev_state)
        if debug_turtled_start or global_debug_mode:
            # --- Torque impulse probe: BEFORE step ---
            agent_idx = 0
            me_before = env._prev_state.players[agent_idx].car_data
            w_before = np.asarray(me_before.angular_velocity, dtype=float)

            # (Optional) print turtle snapshot BEFORE step:
            is_turt, on_g_b, up_z_b = is_turtled(me_before)
            if is_turt:
                print(f"[TURTLE@pre] on_g={on_g_b} up_z={up_z_b:.2f} act={parsed.ravel()} w={w_before}")

            # Step the env
            obs, rew, done, info = env.step(a)

            # --- Torque impulse probe: AFTER step ---
            me_after = env._prev_state.players[agent_idx].car_data
            w_after = np.asarray(me_after.angular_velocity, dtype=float)
            dw = w_after - w_before

            # Only print when we are/were turtled
            is_turt_post, on_g_a, up_z_a = is_turtled(me_after)
            if is_turt or is_turt_post:
                print(f"[TURTLE@post] on_g={on_g_a} up_z={up_z_a:.2f}  Δw={dw}")
        if debug_actions or global_debug_mode:
            print("Parsed action output:", parsed)
            obs, rew, done, info = env.step(a)
            term = done
            trunc = False
            print("Step OK ✅", rew, term, trunc)

    # Add RocketsimVis rendering method
    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    return env

def run_training():
    global START_FRESH
    if global_debug_mode or debug_learning:
        # Torch / GPU info
        # You want cuda available: True and a nonzero device count.
        print("[torch] cuda available:", torch.cuda.is_available())
        print("[torch] device count:", torch.cuda.device_count())
        print("[torch] version:", torch.__version__, "cuda:", torch.version.cuda)
        print("[torch] current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_logger = TurtleRecoveryLogger(agent_index=0, window=45)

    # 64 processes for training
    # 4 for debug
    policy_layers = [1024, 1024, 512, 512]
    critic_layers = [1024, 1024, 512, 512]
    if global_debug_mode or debug_learning or debug_checkpoints or debug_actions or debug_turtled_start:
        n_proc = N_PROC_DEBUG
    else: 
        n_proc = N_PROC_TRAIN


    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    env_for_shape = build_rocketsim_env()
    obs_dim = int(env_for_shape.observation_space.shape[0])

    if global_debug_mode or debug_learning:
        print(f"[init] obs_dim = {obs_dim}")
        print("Action space:", env_for_shape.action_space)
        print("Action space type:", type(env_for_shape.action_space))
        print("Phase:", PHASE)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    existing_runs = sorted(
        [p for p in CHECKPOINT_ROOT.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    # --- Safeguard: no existing checkpoint folders ---
    if not existing_runs:
        print("[warning] No existing checkpoints found. Starting fresh.")
        START_FRESH = True

    resume_run = None
    resume_checkpoint = None
    checkpoint_load_folder = None
    run_dir = None

    if not START_FRESH:
        # === 1. Manual override (highest priority) ===
        if CUSTOM_CKPT_PATH:
            checkpoint_load_folder = CUSTOM_CKPT_PATH
            run_dir = Path(CUSTOM_CKPT_PATH).parents[1]
            print(f"[manual resume] Using custom checkpoint: {checkpoint_load_folder}")

        # === 2. Interactive selection (medium priority) ===
        elif INTERACTIVE_RESUME:
            # Use milestone-only selection
            run_dir, resume_checkpoint = choose_milestone(CHECKPOINT_ROOT)
            if resume_checkpoint:
                checkpoint_load_folder = str(resume_checkpoint)

        # === 3. Auto-resume most recent compatible (fallback) ===
        else:
            for rd in existing_runs:
                meta = read_meta(rd)
                if shapes_match(meta, obs_dim, policy_layers, critic_layers):
                    latest = latest_checkpoint_folder(rd)
                    if latest:
                        resume_run = rd
                        resume_checkpoint = latest
                        run_dir = rd
                        checkpoint_load_folder = str(latest)
                        print(f"[auto-resume] Using {run_dir} → {checkpoint_load_folder}")
                        break

    # === 4. If nothing selected or starting fresh, create new run ===
    if START_FRESH or checkpoint_load_folder is None:
        run_dir = make_run_dir(CHECKPOINT_ROOT)
        checkpoint_load_folder = None
        print(f"[new] Starting from scratch — run_dir = {run_dir}")


    if debug_learning or global_debug_mode or debug_checkpoints or debug_actions:
        n_proc = N_PROC_DEBUG # 2
    else:
        n_proc = N_PROC_TRAIN # 32

    learner = Learner(build_rocketsim_env,
                      # === System ===
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      device=device,  # "cpu" or "cuda"
                      metrics_logger=metrics_logger, # Leave this empty for now.

                      # === PPO Core ===
                      policy_layer_sizes=POLICY_LAYERS,  # policy network
                      critic_layer_sizes=CRITIC_LAYERS,  # critic network
                      ppo_batch_size=50_000,  # batch size - much higher than 300K doesn't seem to help most people. Keep this equal to ts_per_iteration. 
                      # 50k is good for early learning. 100k after bot can hit the ball. Once bot is shooting & scoring, set to 200k or 300k.
                      ts_per_iteration=50_000,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=150_000,  # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=12_500,  # minibatch size - set this as high as your GPU can handle
                      ppo_ent_coef=0.01,  # entropy coefficient - this determines the impact of exploration
                      ppo_epochs=3,   # number of PPO epochs. This is how many times the learning phase is repeated on the same batch of data. I recommend 2 or 3.

                      # === Learning Rates ===
                      policy_lr=POLICY_LEARNING_RATE,  # policy learning rate
                      critic_lr=CRITIC_LEARNING_RATE,  # critic learning rate
                      #Bot that can't score yet: 2e-4
                      #Bot that is actually trying to score on its opponent: 1e-4
                      #Bot that is learning outplay mechanics (dribbling and flicking, air dribbles, etc.): 0.8e-4 or lower

                      # === Checkpointing === 
                      checkpoints_save_folder=str(run_dir),
                      save_every_ts=SAVE_EVERY,  # save every 50k steps
                      checkpoint_load_folder=checkpoint_load_folder, #load from resume_checkpoint (the numbered step subfolder), or None for a fresh run. debug mode added above.
                      
                      # === Miscellaneous ===
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      timestep_limit=10e15,  # Train for 1 quadrillion steps
                      log_to_wandb=True, # Set this to True if you want to use Weights & Biases for logging.
                      render=True,  # Set this to True if you want to render the environment.
                      render_delay=0.02)  # Set this to the delay between frames when rendering.
    import time
    import shutil
    # === Milestone Checkpoint Configuration ===

    MILESTONE_DIR = CHECKPOINT_ROOT / "milestones" / f"Phase{PHASE}-" / LOG_PHASE
    if global_debug_mode or debug_checkpoints:
        print("Milestone directory:", MILESTONE_DIR)
    MILESTONE_DIR.mkdir(parents=True, exist_ok=True)
    learner._next_milestone = MILESTONE_INTERVAL
    learner._milestone_root = MILESTONE_DIR

    # Keep reference to the original internal save() method
    orig_save = getattr(learner, "save", None)

    def wrapped_save(cumulative_timesteps: int):
        """
        Wraps the original Learner.save() so that every time the learner saves
        (normally every `save_every_ts` steps), we also check for milestone triggers.
        """
        if orig_save is not None:
            orig_save(cumulative_timesteps)

        total_steps = cumulative_timesteps
        if total_steps >= learner._next_milestone:
            # === Build phase + milestone folder ===
            phase_name = f"Phase{PHASE}-{LOG_PHASE}"   # e.g., "Phase1-BallTouching"
            milestone_dir = make_milestone_dir(CHECKPOINT_ROOT, phase_name, total_steps)

            print(f"[milestone] ⏱ Auto-saving at {total_steps:,} steps → {milestone_dir}")

            # === Copy checkpoint files into milestone directory ===
            src_folder = os.path.join(learner.checkpoints_save_folder, str(total_steps))
            shutil.copytree(src_folder, milestone_dir, dirs_exist_ok=True)

            # === Collect metrics for meta logging ===
            avg_reward = getattr(learner.agent, "average_reward", None)

            # Fix — if avg_reward is a list, tensor, or cumulative total:
            if isinstance(avg_reward, (list, np.ndarray)) and len(avg_reward) > 0:
                avg_reward = float(np.mean(avg_reward))
            elif isinstance(avg_reward, (int, float)):
                avg_reward = float(avg_reward)
            else:
                avg_reward = None

            ppo = getattr(learner, "ppo_learner", None)
            avg_entropy = getattr(ppo, "last_policy_entropy", None)
            avg_kl = getattr(ppo, "last_kl_divergence", None)

            # === Write enhanced meta info for this milestone ===
            meta_extra = {
                "phase": phase_name,
                "total_timesteps": int(total_steps),
                "avg_reward": avg_reward,
                "policy_entropy": float(avg_entropy) if avg_entropy is not None else None,
                "policy_kl": float(avg_kl) if avg_kl is not None else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            write_meta(
                milestone_dir,
                obs_dim=obs_dim,
                policy_layers=policy_layers,
                critic_layers=critic_layers,
                **meta_extra
            )
            ##################################################### WIP#################
            # === Optional: auto-promote to Releases === IN PROGRESS
            # if avg_reward and avg_reward > 0.9:  # threshold can be tuned per phase
            #     from mysim.training_utils.milestone_utils import promote_to_release
            #     promote_to_release(milestone_dir)
            ##################################################### WIP#################

            # === Schedule next milestone ===
            learner._next_milestone += MILESTONE_INTERVAL


    # Monkey-patch the learner’s save() method
    learner.save = wrapped_save

    # === Start training ===
    start_time = time.time()
    learner.learn()
    assert env_for_shape.action_space.__class__.__name__ == "MultiDiscrete", \
    f"Expected MultiDiscrete action space, got {env_for_shape.action_space}"
    elapsed = (time.time() - start_time) / 3600
    print(f"[training] Completed after {elapsed:.2f} hours.")

if __name__ == "__main__":
    if global_debug_mode or debug_actions:
        parser = SimpleHybridDiscreteAction()
        print("Throttle range:", parser.lookup_table[:, 0].min(), parser.lookup_table[:, 0].max())
        print("Pitch range:", parser.lookup_table[:, 2].min(), parser.lookup_table[:, 2].max())
        print("Yaw=Steer range:", parser.lookup_table[:, 3].min(), parser.lookup_table[:, 3].max())

    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    run_training()
    