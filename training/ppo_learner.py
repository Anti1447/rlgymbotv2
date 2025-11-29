"""
ppo_learner.py
--------------
RocketSim + RLGym PPO training entry point.
Handles environment creation, checkpoint management, and milestone saving.

Author: Nathan Hafey (NJH)
Revision: 1.0.0 – 2025-10-10
"""
# Revisions:
# 2025-10-10 – Rev 1.0.0 – Initial RocketSim + RLGym PPO training entrypoint. – NJH
# 2025-11-28 – Rev 1.1.0 – Added Elo skill-gap auto-eval, self-play gating, and league opponent pool wiring. – NJH
# 2025-11-28 – Rev 1.1.1 – NJH
# - Enabled CPU-based frozen opponent + optional profiling flags for self-play.
# - No change to core PPO config.
# 2025-11-28 – Rev 1.1.2 – NJH
# - Exposed wrapped_save as a module-level function for tests and training hooks.
# - Added global learner/orig_save handles.
# [2025-11-28] Rev 1.1.3 (NJH) - Fix milestone / frozen_policy export cadence to fire
#              only when crossing each 5M-step boundary, even after resume.
# [2025-11-28] Rev 1.1.4 (NJH) - Re-export FrozenPolicyWrapper and expose crossed_milestone
#              helper for tests and milestone logic.

#region imports

#region standard library
import json
import os
import re
import gc
from datetime import datetime
from pathlib import Path
import shutil, time
import multiprocessing
#endregion

#region third-party
import numpy as np
import torch
from rlgym_ppo import Learner
from rlgym_ppo.util import MetricsLogger
#endregion

#region local project imports
from rlgymbotv2.mysim.gamestates import GameState
from rlgymbotv2.mysim.state_setters.debug_state import SymmetricDebugState
from rlgymbotv2.mysim.reward_functions.common_rewards import *
from rlgymbotv2.mysim.action_parsers.utils import (
    get_lookup_table_size,
    find_lookup_table,
)
from rlgymbotv2.mysim.debug_config import (
    global_debug_mode,
    debug_actions,
    debug_learning,
    debug_checkpoints,
    debug_turtled_start,
    debug_selfplay_eval,
)
from rlgymbotv2.mysim.debug_config import dprint, debug_obs
from rlgymbotv2.mysim.training_utils.checkpoint_utils import (
    make_run_dir,
    latest_checkpoint_folder,
    read_meta,
    write_meta,
    shapes_match,
    summarize_checkpoints,
    choose_checkpoint,
)
from rlgymbotv2.mysim.training_utils.milestone_utils import (
    make_milestone_dir,
    promote_to_release,
    choose_milestone,
)
from rlgymbotv2.mysim.debug_tools import debug_controls_sample
from rlgymbotv2.mysim.common_values import CONTROL_ORDER
from rlgymbotv2.mysim.action_parsers.advanced_lookup_table_action_plus import (
    AdvancedLookupTableActionPlus,
)
from rlgymbotv2.mysim.action_parsers.simple_discrete_hybrid_action import (
    SimpleHybridDiscreteAction,
)

from rlgymbotv2.selfplay import (
    FrozenOpponentStore,
    SelfPlayManager,
    SelfPlayEnvWrapper,
    load_frozen_opponent,
    update_elo_from_skill_gap,
    _elo_key_for_milestone,
)

from rlgymbotv2.training.ppo_config import (
    BASE_DIR,
    CHECKPOINT_ROOT,
    CHECKPOINT_MILESTONE_ROOT,
    START_FRESH,
    INTERACTIVE_RESUME,
    CUSTOM_CKPT_PATH,
    SELFPLAY_PROFILE,
    SELFPLAY_ENABLED,
    SELFPLAY_SYNC_INTERVAL,
    MILESTONE_INTERVAL,
    SAVE_EVERY,
    SELFPLAY_UPDATE_MODE,
    MANUAL_FROZEN_DIR,
    PHASE,
    LOG_PHASE,
    TEAM_SIZE,
    PRIMARY_BLUE_INDEX,
    PRIMARY_ORANGE_INDEX,
    N_PROC_DEBUG,
    N_PROC_TRAIN,
    POLICY_LAYERS,
    CRITIC_LAYERS,
    POLICY_LEARNING_RATE,
    CRITIC_LEARNING_RATE,
    LR_DECAY_STEPS,
    MIN_POLICY_LR,
    MIN_CRITIC_LR,
    MECHANICS_MIN_PHASE,
    MECHANICS_WEIGHT,
    MECHANICS_ENABLED,
    ENABLE_ELO_SKILL_GAP,
    ELO_SKILL_GAP_EPISODES,
    ELO_SKILL_GAP_MAX_STEPS,
)
try:
    # When inside the rlgymbotv2.training package
    from .frozen_policy_wrapper import FrozenPolicyWrapper
except ImportError:
    # Fallback for relative-import issues (e.g. direct script runs)
    from rlgymbotv2.training.frozen_policy_wrapper import FrozenPolicyWrapper  # type: ignore
from rlgymbotv2.training.ppo_eval_helpers import run_skill_gap_eval, crossed_milestone
#endregion  # local project imports

#endregion  # imports

learner = None        # will be set in run_training()
orig_save = None      # original learner.save implementation
# These are populated inside run_training(), used by wrapped_save()
selfplay_manager = None # type: ignore
phase_name: str = "PhaseUnknown"
obs_dim: int = 0
device: str = "cpu"



if global_debug_mode or debug_actions or debug_learning or debug_checkpoints or debug_turtled_start or debug_selfplay_eval:
    print("Debug Mode Active")

if global_debug_mode or debug_learning:
    summarize_checkpoints(CHECKPOINT_MILESTONE_ROOT)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

class EnvFactory:
    def make_env(self):
        frozen = FrozenOpponentStore.value

        # Use config flag for profiling (no env var needed)
        profile_flag = SELFPLAY_PROFILE

        # --- WORKER-SIDE LAZY LOAD FOR MANUAL SELF-PLAY ---
        # Main process sets FrozenOpponentStore.value, but workers start with their own
        # copy of the module, so .value will be None there. Fix that here.
        if SELFPLAY_ENABLED and SELFPLAY_UPDATE_MODE == "manual" and frozen is None:
            manual_dir = MANUAL_FROZEN_DIR
            fp = manual_dir / "frozen_policy.pt"
            dprint(
                f"[EnvFactory] (worker) FrozenOpponentStore.value is None, "
                f"trying lazy load from {fp} (exists={fp.exists()})"
            )
            if fp.exists():
                # Use CPU for the opponent in workers; cheap enough and avoids CUDA issues.
                FrozenOpponentStore.value = load_frozen_opponent(fp, device="cpu")
                frozen = FrozenOpponentStore.value
                dprint(f"[EnvFactory] (worker) lazy-loaded frozen opponent: {type(frozen)}")
            else:
                print(f"[EnvFactory] (worker) WARNING: frozen_policy.pt not found at {fp}")

        dprint(
            f"[EnvFactory] make_env: SELFPLAY_ENABLED={SELFPLAY_ENABLED}, "
            f"frozen is {type(frozen)} (None? {frozen is None})"
        )

        # Always spawn opponents when self-play is toggled on
        spawn_opponents = SELFPLAY_ENABLED
        dprint(f"[EnvFactory] spawn_opponents={spawn_opponents}")

        if MECHANICS_ENABLED and np.random.rand() < MECHANICS_WEIGHT:
            base_env = build_mechanics_env(spawn_opponents=spawn_opponents)
        else:
            base_env = build_rocketsim_env(spawn_opponents=spawn_opponents)

        if SELFPLAY_ENABLED and frozen is not None:
            dprint("[EnvFactory] Wrapping in SelfPlayEnvWrapper")
            return SelfPlayEnvWrapper(
                base_env,
                opponent_policy=frozen,
                profile=profile_flag,
            )

        dprint("[EnvFactory] Returning base_env (no self-play wrapper)")
        return base_env


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
        report = {
            "x_vel": avg_linvel[0],
            "y_vel": avg_linvel[1],
            "z_vel": avg_linvel[2],
            "Cumulative Timesteps": cumulative_timesteps
        }
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
    from rlgymbotv2.mysim.reward_functions import CombinedReward
    from rlgymbotv2.mysim.reward_functions.common_rewards import (
        EventReward, SpeedTowardBallReward, FaceBallReward, VelocityReward, InAirReward,
        VelocityBallToGoalReward, StrongHitReward, RecoveryAndLandingReward, BadOrientationPenalty,
        NegativeVelocityTowardOwnGoalReward, BasicShotReward, ClearReward, LiuDistanceBallToGoalReward,
        SaveReward, SaveBoostReward, CollectBoostReward, BasicAerialReward, JumpShotReward, ShotSetupReward
    )
    if global_debug_mode:
        print(f"[reward] Building reward function for phase {phase}")

    if phase == 1: # BallTouching
        return CombinedReward.from_zipped( # aiming to get rewards under 100 points total
            (EventReward(touch=1), 1.0), # Big Reward for touching the ball
            (SpeedTowardBallReward(), 0.05), # small reward for moving toward the ball
            (FaceBallReward(), 0.01), # tiny reward for facing the ball
            (InAirReward(), 0.0015) # don't forget how to jump
        )
    elif phase == 2:  # BronzeSkills1
        return CombinedReward.from_zipped( # aiming to get rewards under 100 points total
            # === Event Rewards ===
            (EventReward(goal=1), 20),
            (EventReward(concede=1), -10), # reward for scoring, penalty for conceding. 1/1 ratio for balance
            (StrongHitReward(), 0.2), # Smaller reward for hitting ball
            # === Continuous Rewards ===
            (VelocityBallToGoalReward(), 0.2),
            (SpeedTowardBallReward(), 0.05),
            (FaceBallReward(), 0.015),
            (InAirReward(), 0.0015)
        )
    elif phase == 3:  # SilverSkills1 — clean shooting fundamentals
        return CombinedReward.from_zipped(
            # Scoring & aggression
            (EventReward(goal=1), 25.0),
            (EventReward(concede=1), -8.0),

            # Shooting trajectory reward (primary driver)
            (VelocityBallToGoalReward(), 8.0),
            (LiuDistanceBallToGoalReward(), 3.0),
            (NegativeVelocityTowardOwnGoalReward(), -4.0),
            (LiuDistancePlayerToBallReward(), 1.0),

            # Strong hits > weak taps
            (StrongHitReward(), 5.0),

            # Encourage purposeful jump touches (but don’t force aerials yet)
            (JumpShotReward(), 2.5),

            # This helps cutting/positioning a LOT in Silver
            (FaceBallReward(), 1.0),

            # Mild ball-chasing shaping
            (SpeedTowardBallReward(), 1.0),

            # Boost discipline (light touch)
            (SaveBoostReward(), 0.4),

            # Clean movement
            (RecoveryAndLandingReward(), 0.1),
            (BadOrientationPenalty(), 0.1),
            (InAirReward(), 0.075),
        )

    elif phase == 4:  # GoldSkills1 — shooting consistency + basic aerials
        return CombinedReward.from_zipped(
            # Scoring + aggression
            (EventReward(goal=1), 20.0),
            (EventReward(concede=1), -10.0),

            # Better shooting (main reward)
            (VelocityBallToGoalReward(), 10.0),

            # Strong shots + stable touches
            (StrongHitReward(), 5.0),
            (BasicShotReward(), 4.0),

            # Encourage simple aerial attempts (low-weight)
            (BasicAerialReward(), 4.0),

            # Boost economy: now important
            (SaveBoostReward(), 0.8),
            (CollectBoostReward(), 0.3),

            # Movement shaping
            (FaceBallReward(), 1.0),
            (SpeedTowardBallReward(), 1.0),
            (RecoveryAndLandingReward(), 0.3),
            (BadOrientationPenalty(), 0.1),
        )


def build_rocketsim_env(spawn_opponents=None):
    import rlgym_sim
    from rlgymbotv2.mysim.reward_functions import CombinedReward
    from rlgymbotv2.mysim.reward_functions.common_rewards import (
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
    from rlgymbotv2.mysim.obs_builders import DefaultObs
    from rlgymbotv2.mysim.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
    from rlgymbotv2.mysim.reward_functions.common_rewards.conditional_rewards import GoalIfTouchedLastConditionalReward
    from rlgymbotv2.mysim import common_values
    from rlgymbotv2.mysim.state_setters import RandomState
    from rlgymbotv2.mysim.action_parsers.discrete_act_2 import DiscreteAction2
    from rlgymbotv2.mysim.action_parsers import AdvancedLookupTableAction
    from rlgymbotv2.mysim.action_parsers.continuous_act import ContinuousAction
    from rlgymbotv2.mysim.action_parsers.wrappers.clip_action_wrapper import ClipActionWrapper
    from rlgymbotv2.mysim.action_parsers.wrappers.sticky_buttons_wrapper import StickyButtonsWrapper
    from rlgymbotv2.mysim.action_parsers.wrappers.state_aware_lut_wrapper import StateAwareLUTWrapper
    from rlgymbotv2.mysim.action_parsers.wrappers.repeat_action import RepeatAction
    from rlgymbotv2.mysim.action_parsers.wrappers.collapse_to_single_tick import CollapseToSingleTick
    from rlgymbotv2.mysim.action_parsers.wrappers.expand_to_tick_skip import ExpandToTickSkip
    from rlgymbotv2.mysim.action_parsers.wrappers.expand_for_rocketsim import ExpandForRocketSim
    from rlgymbotv2.mysim.action_parsers.wrappers.final_rocketsim_adapter import FinalRocketSimAdapter
    from rlgymbotv2.mysim.action_parsers.utils import find_forward_fallback_idx

    # If self-play is on, force a 1v1. Otherwise keep it 1v0.
    # If not specified, follow global SELFPLAY_ENABLED
    if spawn_opponents is None:
        spawn_opponents = SELFPLAY_ENABLED
    game_tick_rate = 120
    tick_skip = 8
    
    # Use global TEAM_SIZE for future 2v2 / 3v3
    team_size = TEAM_SIZE
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 15
    no_touch_timeout_ticks = int(round(no_touch_timeout_seconds * game_tick_rate / tick_skip))
    game_timeout_seconds = 300
    game_timeout_ticks = int(round(game_timeout_seconds * game_tick_rate / tick_skip))

    if global_debug_mode or debug_selfplay_eval or debug_learning:
        print(f"[build_env] spawn_opponents={spawn_opponents}, "
              f"blue_team_size={blue_team_size}, orange_team_size={orange_team_size}")


    lut = SimpleHybridDiscreteAction()

    # Use the first all-zeros row as fallback
    idle_idx = int(np.where((lut.lookup_table == 0).all(axis=1))[0][0])

    safe_lut = StateAwareLUTWrapper(lut, fallback_index=idle_idx)
    action_parser = FinalRocketSimAdapter(safe_lut)

    if debug_actions or debug_obs: 
        state_setter = SymmetricDebugState()
    else:
        state_setter = RandomState(True, True, False)

    terminal_conditions = [
        GoalScoredCondition(),
        NoTouchTimeoutCondition(no_touch_timeout_ticks),
        TimeoutCondition(game_timeout_ticks)
    ]

    reward_fn = build_reward_function(phase=PHASE)

    from rlgymbotv2.mysim.obs_builders.advanced_obs_plus import AdvancedObsPlus
    obs_builder = AdvancedObsPlus(
        max_allies=2,
        max_opponents=3,
        k_nearest_pads_me=4,
        k_nearest_pads_ball=2,
        stack_size=1,
        include_prev_action=True,
    )

    env = rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=state_setter
    )
    obs, info = env.reset(return_info=True)

        # --- after env.reset()
    if global_debug_mode or debug_selfplay_eval or debug_learning:
        state = info.get("state", None)
        num_players = len(getattr(state, "players", [])) if state is not None else "?"
        # players are ordered blue(s) then orange(s) in rlgym_sim
        print(f"[build_env] players in state: {num_players}")
        if state is not None:
            blue = [p for p in state.players if p.team_num == 0]
            orange = [p for p in state.players if p.team_num == 1]
            print(f"[build_env] blue players={len(blue)}, orange players={len(orange)}")


    # --- after env.reset()
    if global_debug_mode or debug_actions:
        me0 = env._prev_state.players[0].car_data
        ok, on_g, up_z = is_turtled(me0)
        print("[reset] on_ground:", on_g, " up_z:", f"{up_z:.2f}", " turtled?", ok)
        print("Controlled agents (spectator_ids):", len(env._match._spectator_ids))
        debug_controls_sample(action_parser, env)
        tab = find_lookup_table(action_parser)
        if tab is None:
            print("[list_turning_actions] No lookup table found.")
        else:
            lefts = tab[tab[:, 1] < 0]
            rights = tab[tab[:, 1] > 0]
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

    if global_debug_mode or debug_actions:
        print("\n--- DEBUG ACTION SAMPLE ---")

        # Number of controlled agents
        num_players = len(env._match._spectator_ids)

        # Sample correct number of actions
        if hasattr(env.action_space, "sample"):
            a = [env.action_space.sample() for _ in range(num_players)]
        else:
            a = [env.action_space.sample()]

        print("Action input to env.step:", a)

        parsed = env._match._action_parser.parse_actions(a, env._prev_state)
        print("Parsed action output:", parsed)

        obs, rew, done, info = env.step(a)
        print("Step OK", rew, done, info)

    # Add RocketsimVis rendering method
    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    return env


def build_mechanics_env(spawn_opponents=None):
    """
    Mechanics lane environment.

    For now, this reuses the standard RocketSim env.
    Later we’ll swap in mechanics-focused state setters / rewards.

    spawn_opponents:
      - True  => 1v1
      - False => 1v0
      - None  => defaults to SELFPLAY_ENABLED
    """
    return build_rocketsim_env(spawn_opponents=spawn_opponents)


def debug_constant_actions():
    env = build_rocketsim_env()
    print("[debug] action_space:", env.action_space)

    # full throttle forward, no steer/jump/etc
    const = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    obs, info = env.reset(return_info=True)
    for t in range(120):
        actions = np.repeat(const, repeats=len(env._match._spectator_ids), axis=0)
        obs, rew, done, info = env.step(actions)
        state = env._prev_state
        blue = state.players[0].car_data
        if len(state.players) > 1:
            orange = state.players[1].car_data
        else:
            orange = None

        print(f"t={t:03d} | blue={blue.position}")
        if orange is not None:
            print(f"         orange={orange.position}")

def debug_dump_obs():
    """
    Build a one-off environment, reset it, run AdvancedObsPlus.pre_step(state),
    and dump Blue vs Orange observations in labeled sections.

    Sections (for stack_size == 1):
      [0:6)    = ball (rel pos 3, rel vel 3)
      [6:16)   = self (vel_cf 3, angvel_cf 3, boost 1, has_flip 1, on_ground 1, up_z 1)
      [16:19)  = context (kickoff, time_left_norm, score_diff_squash)
      [19:29)  = action awareness (prev_action 8, can_boost, can_flip)
      [29:29+ally_dim)  = allies (max_allies * 11)
      [29+ally_dim:29+ally_dim+opp_dim) = opponents (max_opponents * 11)
      [..pad_range..)   = pads summary (nearest to me and to ball)
      [last 6]          = goal vectors (their_goal_from_ball 3, my_goal_from_ball 3)
    """
    print("\n[debug] Running obs-dump...\n")

    from rlgymbotv2.mysim.obs_builders.advanced_obs_plus import AdvancedObsPlus

    # Build vanilla env (no self-play injection)
    env = build_rocketsim_env(
        spawn_opponents=SELFPLAY_ENABLED
    )

    # Get state from info dict
    obs, info = env.reset(return_info=True)
    state = info["state"]

    # Use the SAME builder instance the env uses
    builder = env._match._obs_builder
    assert isinstance(builder, AdvancedObsPlus), f"Unexpected obs builder: {type(builder)}"

    # Make sure pre_step ran
    builder.pre_step(state)
    cache = builder._cached

    blue = state.players[0]
    orange = state.players[1]

    prev_action = np.zeros(8, dtype=np.float32)

    blue_obs_full = builder.build_obs(blue, state, prev_action)
    orange_obs_full = builder.build_obs(orange, state, prev_action)

    # If we ever enable stacking, focus on the *last* frame for sectioning
    if builder.stack_size > 1:
        frame_size = blue_obs_full.size // builder.stack_size
        blue_obs = blue_obs_full[-frame_size:]
        orange_obs = orange_obs_full[-frame_size:]
        print(f"[debug] stack_size={builder.stack_size}, per-frame size={frame_size}")
    else:
        blue_obs = blue_obs_full
        orange_obs = orange_obs_full
        frame_size = blue_obs.size

    # --- basic sanity ---
    assert np.all(np.isfinite(blue_obs)), "Blue obs has NaN or INF!"
    assert np.all(np.isfinite(orange_obs)), "Orange obs has NaN or INF!"
    assert blue_obs.size == orange_obs.size, "Obs size mismatch!"
    print(f"[debug] Obs lengths match = {blue_obs.size}")

    # --- raw physics and rotation sanity ---
    print("\n[debug] Raw physics:")
    print(f"  Blue pos={blue.car_data.position}, forward={blue.car_data.forward()}, up={blue.car_data.up()}")
    print(f"  Orange pos={orange.car_data.position}, forward={orange.car_data.forward()}, up={orange.car_data.up()}")

    # Validate car-frame rotation: forward->~[1,0,0], up->~[0,0,1] in car frame
    R_b = builder._rot_mat(np.asarray(blue.car_data.forward(), dtype=np.float32),
                           np.asarray(blue.car_data.up(), dtype=np.float32))
    R_o = builder._rot_mat(np.asarray(orange.car_data.forward(), dtype=np.float32),
                           np.asarray(orange.car_data.up(), dtype=np.float32))

    f_b_cf = builder._to_car_frame(R_b, np.asarray(blue.car_data.forward(), dtype=np.float32))
    u_b_cf = builder._to_car_frame(R_b, np.asarray(blue.car_data.up(), dtype=np.float32))
    f_o_cf = builder._to_car_frame(R_o, np.asarray(orange.car_data.forward(), dtype=np.float32))
    u_o_cf = builder._to_car_frame(R_o, np.asarray(orange.car_data.up(), dtype=np.float32))

    print("\n[debug] Car-frame sanity:")
    print(f"  Blue forward in car frame ~ {f_b_cf}")
    print(f"  Blue up      in car frame ~ {u_b_cf}")
    print(f"  Orange fwd   in car frame ~ {f_o_cf}")
    print(f"  Orange up    in car frame ~ {u_o_cf}")

    # --- compute section boundaries from builder config ---
    ball_dim = 3 + 3
    me_dim = 3 + 3 + 1 + 1 + 1 + 1       # vel_cf, angvel_cf, boost, has_flip, on_ground, up_z
    ctx_dim = 3                          # kickoff, time_left_norm, score_diff_squash
    act_dim = (8 if builder.include_prev_action else 0) + 2   # prev_action(8), [can_boost, can_flip]
    other_per = 3 + 3 + 3 + 2           # rel_p, rel_v, rel_w, [boost, alive] = 11
    ally_dim = builder.max_allies * other_per
    opp_dim = builder.max_opponents * other_per
    pad_dim = 6 * (builder.k_nearest_pads_me + builder.k_nearest_pads_ball)
    goal_dim = 6

    # Check that the sum matches the per-frame size
    expected = ball_dim + me_dim + ctx_dim + act_dim + ally_dim + opp_dim + pad_dim + goal_dim
    if expected != frame_size:
        print(f"[warn] Expected per-frame size {expected}, got {frame_size}")

    idx = 0
    def take(n):
        nonlocal idx
        s = idx
        e = idx + n
        idx = e
        return s, e

    ball_s, ball_e = take(ball_dim)
    me_s, me_e = take(me_dim)
    ctx_s, ctx_e = take(ctx_dim)
    act_s, act_e = take(act_dim)
    ally_s, ally_e = take(ally_dim)
    opp_s, opp_e = take(opp_dim)
    pad_s, pad_e = take(pad_dim)
    goals_s, goals_e = take(goal_dim)

    # Helper: print section with max |blue - orange|
    def dump_section(name, s, e):
        b = blue_obs[s:e]
        o = orange_obs[s:e]
        diff = np.max(np.abs(b - o)) if b.size > 0 else 0.0
        print(f"\n[section: {name}] idx[{s}:{e}] len={e-s} max|Δ|={diff:.5f}")
        print(f"  blue  = {np.array2string(b, precision=4, floatmode='fixed')}")
        print(f"  orange= {np.array2string(o, precision=4, floatmode='fixed')}")

    # --- dump labeled sections ---
    dump_section("BALL (rel pos+vel in car frame)", ball_s, ball_e)
    dump_section("SELF (vel_cf, angvel_cf, boost, has_flip, on_ground, up_z)", me_s, me_e)
    dump_section("CONTEXT (kickoff, time_left_norm, score_diff)", ctx_s, ctx_e)
    dump_section("ACTION AWARENESS (prev_action, can_boost, can_flip)", act_s, act_e)
    dump_section(f"ALLIES x{builder.max_allies} (each: rel_p, rel_v, rel_w, boost, alive)", ally_s, ally_e)
    dump_section(f"OPPONENTS x{builder.max_opponents}", opp_s, opp_e)
    dump_section(
        f"PADS (nearest me x{builder.k_nearest_pads_me}, nearest ball x{builder.k_nearest_pads_ball})",
        pad_s, pad_e,
    )
    dump_section("GOAL VECTORS (their_goal_from_ball, my_goal_from_ball)", goals_s, goals_e)

    # --- nearest pad distances (world frame) ---
    pads = cache.get("pads", [])
    if pads:
        my_car_b = blue.car_data
        ball_pos = cache["ball_pos"]

        bp = np.asarray([p[0] for p in pads], dtype=np.float32)
        d_me = np.linalg.norm(bp - my_car_b.position, axis=1)
        d_ball = np.linalg.norm(bp - ball_pos, axis=1)

        idx_me = np.argsort(d_me)[: builder.k_nearest_pads_me]
        idx_ball = np.argsort(d_ball)[: builder.k_nearest_pads_ball]

        print("\n[debug] Nearest pads to BLUE car:")
        for i in idx_me:
            pos, is_big, active, timer = pads[i]
            print(f"  pad[{i}]: dist={d_me[i]:7.1f}, pos={pos}, big={is_big}, active={active}, timer={timer:.2f}")

        print("\n[debug] Nearest pads to BALL:")
        for i in idx_ball:
            pos, is_big, active, timer = pads[i]
            print(f"  pad[{i}]: dist={d_ball[i]:7.1f}, pos={pos}, big={is_big}, active={active}, timer={timer:.2f}")
    else:
        print("\n[debug] No pads found in state.boost_pads")

    print("\n[debug] Done.\n")

def debug_dump_pads():
    """
    Debug just the boost-pad encoding from AdvancedObsPlus.

    - Uses a NORMAL env reset (whatever state_setter you normally use)
    - Prints how many boost pads exist in the GameState
    - Decodes the pad slice from the obs (k_me + k_ball pads, 6 features each)
    - Prints cached pad world positions + flags from the builder cache
    """
    print("\n[debug] Running pad-dump...\n")

    from rlgymbotv2.mysim.obs_builders.advanced_obs_plus import AdvancedObsPlus

    # NOTE: use your usual env here – NOT the symmetric debug state setter.
    env = build_rocketsim_env(
        spawn_opponents=SELFPLAY_ENABLED
    )
    obs, info = env.reset(return_info=True)
    state = info["state"]

    builder = env._match._obs_builder
    assert isinstance(builder, AdvancedObsPlus), f"Unexpected obs builder: {type(builder)}"

    builder.pre_step(state)
    cache = builder._cached

    blue = state.players[0]
    prev_action = np.zeros(8, dtype=np.float32)

    blue_obs_full = builder.build_obs(blue, state, prev_action)

    # If stacking is ever enabled, look only at the last frame
    if builder.stack_size > 1:
        frame_size = blue_obs_full.size // builder.stack_size
        blue_obs = blue_obs_full[-frame_size:]
    else:
        frame_size = blue_obs_full.size
        blue_obs = blue_obs_full

    # --- REBUILD section boundaries (must match debug_dump_obs) ---
    other_per = 3 + 3 + 3 + 2          # rel_p, rel_v, rel_w, [boost, alive] = 11
    ball_dim = 3 + 3
    me_dim = 3 + 3 + 1 + 1 + 1 + 1
    ctx_dim = 3
    act_dim = (8 if builder.include_prev_action else 0) + 2
    ally_dim = builder.max_allies * other_per
    opp_dim = builder.max_opponents * other_per
    pad_dim = 6 * (builder.k_nearest_pads_me + builder.k_nearest_pads_ball)
    goal_dim = 6

    expected = ball_dim + me_dim + ctx_dim + act_dim + ally_dim + opp_dim + pad_dim + goal_dim
    print(f"[debug] frame_size={frame_size}, expected={expected}")

    idx = 0
    def skip(n):
        nonlocal idx
        s = idx
        idx += n
        return s, idx

    skip(ball_dim)
    skip(me_dim)
    skip(ctx_dim)
    skip(act_dim)
    skip(ally_dim)
    skip(opp_dim)
    pad_s, pad_e = skip(pad_dim)
    # goals_s, goals_e = skip(goal_dim)  # not needed here

    print(f"[debug] pad slice idx[{pad_s}:{pad_e}], len={pad_e - pad_s}")

    pad_slice = blue_obs[pad_s:pad_e]

    k_total = builder.k_nearest_pads_me + builder.k_nearest_pads_ball
    if pad_slice.size != 6 * k_total:
        print(f"[warn] pad_slice size {pad_slice.size} != 6 * (k_me+k_ball)={6*k_total}")

    pad_feats = pad_slice.reshape((k_total, 6))

    print(f"[debug] state.boost_pads len = {len(getattr(state, 'boost_pads', []))}")
    print("[debug] Encoded pad features per slot (rel_x, rel_y, rel_z, is_big, active, timer_norm):")
    for i, row in enumerate(pad_feats):
        print(f"  slot {i}: {np.array2string(row, precision=4, floatmode='fixed')}")

    # If AdvancedObsPlus.pre_step cached pads, inspect them
    pads = cache.get("pads", [])
    print(f"\n[debug] cache['pads'] len = {len(pads)}")

    for i, p in enumerate(pads[:10]):  # show first 10 to avoid spam
        # Adjust these tuple fields if your cache format differs
        pos, is_big, is_active, timer = p
        print(f"  pad[{i}]: pos={pos}, big={is_big}, active={is_active}, timer={timer:.2f}")

    print("\n[debug] pad-dump done.\n")


def run_training():
    global START_FRESH
    global learner, orig_save, selfplay_manager, phase_name, obs_dim, device
    # --- GPU Info ---
    if global_debug_mode or debug_learning:
        print("[torch] cuda available:", torch.cuda.is_available())
        print("[torch] device count:", torch.cuda.device_count())
        print("[torch] version:", torch.__version__, "cuda:", torch.version.cuda)
        print("[torch] current device:",
              torch.cuda.current_device() if torch.cuda.is_available() else None)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"[ppo_learner] Using device: {device}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu" # Force CPU for now due to Rocketsim issues
    metrics_logger = TurtleRecoveryLogger(agent_index=0, window=45)

    if debug_checkpoints or global_debug_mode:
        dprint("[DEBUG] CHECKPOINT_ROOT:", CHECKPOINT_ROOT.resolve())
        dprint("[DEBUG] CHECKPOINT_MILESTONE_ROOT:", CHECKPOINT_MILESTONE_ROOT.resolve())

    # --- Process Count ---
    if global_debug_mode or debug_learning or debug_checkpoints or debug_actions or debug_turtled_start or debug_selfplay_eval:
        n_proc = N_PROC_DEBUG
    else:
        n_proc = N_PROC_TRAIN

    min_inference_size = max(1, int(round(n_proc * 0.9)))

    # --- Observation Shape ---
    env_for_shape = build_rocketsim_env()
    obs_dim = int(env_for_shape.observation_space.shape[0])

    if global_debug_mode or debug_learning:
        print(f"[init] obs_dim = {obs_dim}")
        print("Action space:", env_for_shape.action_space)
        print("Action space type:", type(env_for_shape.action_space))
        print("Phase:", PHASE)

    # --- Checkpoint Setup ---
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted(
        [p for p in CHECKPOINT_ROOT.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not existing_runs:
        print("[warning] No existing checkpoints found. Starting fresh.")
        START_FRESH = True

    resume_checkpoint = None
    checkpoint_load_folder = None
    run_dir = None

    # === Resume Logic ===
    if not START_FRESH:
        # 1. Manual override
        if CUSTOM_CKPT_PATH:
            checkpoint_load_folder = CUSTOM_CKPT_PATH
            run_dir = Path(CUSTOM_CKPT_PATH).parents[1]
            print(f"[manual resume] Using custom checkpoint: {checkpoint_load_folder}")

        # 2. Interactive milestone
        elif INTERACTIVE_RESUME:
            run_dir, resume_checkpoint = choose_milestone(CHECKPOINT_ROOT)
            if resume_checkpoint:
                checkpoint_load_folder = str(resume_checkpoint)

        # 3. Auto-resume most recent compatible
        else:
            for rd in existing_runs:
                meta = read_meta(rd)
                if shapes_match(meta, obs_dim, POLICY_LAYERS, CRITIC_LAYERS):
                    latest = latest_checkpoint_folder(rd)
                    if latest:
                        resume_checkpoint = latest
                        run_dir = rd
                        checkpoint_load_folder = str(latest)
                        print(f"[auto-resume] Using {run_dir} → {checkpoint_load_folder}")
                        break

    # === Create New Run If Needed ===
    if START_FRESH or checkpoint_load_folder is None:
        run_dir = make_run_dir(CHECKPOINT_ROOT)
        checkpoint_load_folder = None
        print(f"[new] Starting from scratch — run_dir = {run_dir}")

    # Ensure n_proc after resume logic
    if debug_learning or global_debug_mode or debug_checkpoints or debug_actions:
        n_proc = N_PROC_DEBUG
    else:
        n_proc = N_PROC_TRAIN

    # --- Initialize frozen opponent BEFORE creating envs / learner ---
    if global_debug_mode or debug_learning or debug_selfplay_eval:
        print(f"[selfplay] config: SELFPLAY_ENABLED={SELFPLAY_ENABLED}, "
          f"SELFPLAY_UPDATE_MODE={SELFPLAY_UPDATE_MODE}")

    if SELFPLAY_ENABLED and SELFPLAY_UPDATE_MODE == "manual":
        manual_dir = MANUAL_FROZEN_DIR
        fp = manual_dir / "frozen_policy.pt"
        if global_debug_mode or debug_learning or debug_selfplay_eval:
            print(f"[selfplay] manual_dir={manual_dir}")
            print(f"[selfplay] looking for frozen_policy at: {fp} (exists={fp.exists()})")

        if fp.exists():
            FrozenOpponentStore.value = load_frozen_opponent(fp, device="cpu")
            if global_debug_mode or debug_learning or debug_selfplay_eval:
                print(f"[selfplay] Manual frozen opponent loaded from {fp}")
                print(f"[selfplay] FrozenOpponentStore.value type="
                      f"{type(FrozenOpponentStore.value)}")
        else:
            FrozenOpponentStore.value = None
            if global_debug_mode or debug_learning or debug_selfplay_eval:
                print(f"[selfplay] WARNING: manual frozen_policy.pt not found at {fp}; "
                      f"self-play will behave like 1v0 until you sync.")
    else:
        FrozenOpponentStore.value = None
        if global_debug_mode or debug_learning or debug_selfplay_eval:
            print("[selfplay] Manual mode not active or self-play disabled; "
                  "FrozenOpponentStore set to None.")


    env_factory = EnvFactory()

    # Self-play manager (used only in AUTO mode)
    selfplay_manager = SelfPlayManager(
        checkpoint_root=CHECKPOINT_ROOT,
        device="cpu", # Frozen opponents always on CPU for inference
        min_pool_steps=5_000_000,
        reward_gate=0.5,
    )

    # --- Expose learner + save hook for tests and wrapped_save() ---
    global learner, orig_save

    # --- Construct PPO Learner ---
    learner = Learner(
        env_factory.make_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        device=device,
        metrics_logger=metrics_logger,
        policy_layer_sizes=POLICY_LAYERS,
        critic_layer_sizes=CRITIC_LAYERS,
        ppo_batch_size=SAVE_EVERY,
        ts_per_iteration=SAVE_EVERY,
        exp_buffer_size=SAVE_EVERY * 3,
        ppo_minibatch_size=25_000,
        ppo_ent_coef=0.01,
        ppo_epochs=3,
        policy_lr=POLICY_LEARNING_RATE,
        critic_lr=CRITIC_LEARNING_RATE,
        gae_gamma = 0.995,
        gae_lambda = 0.95,
        ppo_clip_range = 0.2,
        checkpoints_save_folder=str(run_dir),
        save_every_ts=SAVE_EVERY,
        checkpoint_load_folder=checkpoint_load_folder,
        standardize_returns=True,
        standardize_obs=False,
        timestep_limit=10e15,
        log_to_wandb=True,
        render=False,
        render_delay=0.0
    )
    # --- Milestone Setup ---
    phase_name = f"Phase{PHASE}-{LOG_PHASE}".replace(" ", "")
    MILESTONE_DIR = CHECKPOINT_ROOT / "milestones" / phase_name
    MILESTONE_DIR.mkdir(parents=True, exist_ok=True)

    learner._next_milestone = MILESTONE_INTERVAL
    learner._milestone_root = MILESTONE_DIR

    orig_save = getattr(learner, "save", None)
    policy_module = getattr(learner.agent, "policy", None)
    # if policy_module is not None:
    #     policy_module.to(device)
    #     print("[ppo_learner] Policy first param device:",
    #           next(policy_module.parameters()).device)
    # else:
    #     print("[ppo_learner] WARNING: learner.agent has no 'policy' attribute")
    # --- Wrapped Save Hook ---
    learner = learner  # bind the module-level name to this instance
    orig_save = getattr(learner, "save", None)
    learner.save = wrapped_save

    # --- Training ---
    start_time = time.time()



    learner.learn()
    # training_loop_skeleton(learner, selfplay_manager, max_timesteps=None)

    if global_debug_mode or debug_actions or debug_learning:
        print(
            "[train] Finished learner.learn(). "
            f"Action space at build time was: {env_for_shape.action_space} "
            f"({env_for_shape.action_space.__class__.__name__})"
        )

    elapsed = (time.time() - start_time) / 3600
    print(f"[training] Completed after {elapsed:.2f} hours.")

    learner._next_milestone = MILESTONE_INTERVAL
    learner._milestone_root = MILESTONE_DIR

    orig_save = getattr(learner, "save", None)

# --- Wrapped Save Hook ---
def wrapped_save(cumulative_timesteps: int):
    if orig_save is not None:
        orig_save(cumulative_timesteps)
    # rlgym-ppo should give us an int, but make sure:
    total_steps = int(cumulative_timesteps)
    # --- LR schedule + self-play gating ---
    avg_reward = getattr(learner.agent, "average_reward", None)
    if isinstance(avg_reward, (list, np.ndarray)):
        avg_reward = float(np.mean(avg_reward)) if len(avg_reward) > 0 else None
    # AUTO self-play manager updates on every save (if enabled)
    if SELFPLAY_ENABLED and SELFPLAY_UPDATE_MODE == "auto":
        selfplay_manager.maybe_update(
            total_steps,
            avg_reward,
            SELFPLAY_ENABLED,
        )
    # --- Milestone trigger: ONLY when we cross a 5M boundary ---
    #
    # Example with SAVE_EVERY = 100_000, MILESTONE_INTERVAL = 5_000_000:
    #   4_900_000 -> 5_000_000  => milestone fires
    #   5_000_000 -> 5_100_000  => no new milestone
    #
    prev_steps = max(0, total_steps - SAVE_EVERY)
    if not crossed_milestone(prev_steps, total_steps):
        return  # normal 100k save only, no milestone

    # We just crossed N * 5M for some N; use total_steps as the label
    milestone_dir = make_milestone_dir(CHECKPOINT_ROOT, phase_name, total_steps)
    print(f"[milestone] ⏱ Auto-saving at {total_steps:,} steps → {milestone_dir}")
    src_folder = os.path.join(learner.checkpoints_save_folder, str(total_steps))
    shutil.copytree(src_folder, milestone_dir, dirs_exist_ok=True)
    # Always export a frozen policy from milestones
    try:
        from rlgymbotv2.training.frozen_policy_wrapper import FrozenPolicyWrapper
        policy = learner.ppo_learner.policy
        frozen = FrozenPolicyWrapper(policy)
        out_path = os.path.join(milestone_dir, "frozen_policy.pt")
        torch.save(frozen, out_path)
        print(f"[milestone] Exported frozen opponent policy → {out_path}")
    except Exception as e:
        print(f"[milestone] Failed to export frozen opponent: {e}")
    ppo = getattr(learner, "ppo_learner", None)
    avg_entropy = getattr(ppo, "last_policy_entropy", None)
    avg_kl = getattr(ppo, "last_kl_divergence", None)
    write_meta(
        milestone_dir,
        obs_dim=obs_dim,
        policy_layers=POLICY_LAYERS,
        critic_layers=CRITIC_LAYERS,
        phase=phase_name,
        total_timesteps=int(total_steps),
        avg_reward=avg_reward,
        policy_entropy=float(avg_entropy) if avg_entropy is not None else None,
        policy_kl=float(avg_kl) if avg_kl is not None else None,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )
    # Track the *next* future boundary (mainly for debugging / introspection)
    k = total_steps // MILESTONE_INTERVAL
    learner._next_milestone = (k + 1) * MILESTONE_INTERVAL



def debug_introspect_pad():
    print("\n[debug] Introspecting RocketSim pad object...\n")

    env = build_rocketsim_env(
        spawn_opponents=SELFPLAY_ENABLED
    )
    obs, info = env.reset(return_info=True)
    state = info["state"]

    pads = getattr(state, "boost_pads", None)
    if pads is None or len(pads) == 0:
        print("[debug] No pads found!")
        return

    bp = pads[0]
    print("[debug] Pad object type:", type(bp))

    print("\n[debug] dir(bp):")
    for name in dir(bp):
        if not name.startswith("_"):
            print(" ", name)

    print("\n[debug] getattr dump:")
    for name in dir(bp):
        if not name.startswith("_"):
            try:
                val = getattr(bp, name)
                print(f" {name}: {val}")
            except Exception as e:
                print(f" {name}: <error: {e}>")

    print("\n[debug] Done.")

if __name__ == "__main__":
    # if global_debug_mode or debug_actions:
    #     parser = SimpleHybridDiscreteAction()
    #     print("Throttle range:", parser.lookup_table[:, 0].min(), parser.lookup_table[:, 0].max())
    #     print("Pitch range:", parser.lookup_table[:, 2].min(), parser.lookup_table[:, 2].max())
    #     print("Yaw=Steer range:", parser.lookup_table[:, 3].min(), parser.lookup_table[:, 3].max())

    multiprocessing.set_start_method("spawn", force=True)
    # a123 = 0
    # if a123 == 0:
    #     debug_introspect_pad()
    if debug_actions and not debug_obs:
        debug_constant_actions()
    elif debug_obs:
        debug_dump_obs()
        # debug_dump_pads()
    else:
        run_training()
