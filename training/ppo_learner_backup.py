################################
# Date: 10/10/2025
# Revision: 1.0.0
# Description: Initial Release
# Title: ppo_learner.py
# Author: NJH
################################
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
from mysim.action_parsers.utils import get_lookup_table_size
from mysim.debug_config import global_debug_mode, debug_actions, debug_learning
from mysim.training_utils.checkpoint_utils import (
    make_run_dir, latest_checkpoint_folder, read_meta, write_meta, shapes_match, summarize_checkpoints
)
from mysim.debug_tools import debug_controls_sample


if global_debug_mode or debug_actions or debug_learning:
    print("Debug Mode Active")

# Constants
CHECKPOINT_ROOT = Path("data/checkpoints")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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

# # Move these functions to the top of example.py
# def get_latest_directory(directory):
#     """Returns the most recently modified directory inside the given directory."""
#     list_of_dirs = [d for d in os.scandir(directory) if d.is_dir()]
#     if not list_of_dirs:
#         return None
#     latest_dir = max(list_of_dirs, key=lambda d: d.stat().st_mtime).path
#     return latest_dir

# def get_latest_checkpoint(run_directory):
#     """Returns the most recently modified 'checkpoint' directory inside the latest run directory."""
#     latest_run_dir = get_latest_directory(run_directory)
#     if not latest_run_dir:
#         return None

#     latest_checkpoint_dir = get_latest_directory(latest_run_dir)  # Find latest subdirectory inside
#     return latest_checkpoint_dir

def debug_controls_sample(action_parser, env):
    import numpy as np
    N = get_lookup_table_size(action_parser)
    a = np.random.randint(0, N, size=(1,))
    parsed = action_parser.parse_actions(a, env._prev_state)
    print("\n--- ACTION DEBUG SAMPLE ---")
    print("Raw index:", a)
    print("Decoded controls:", parsed)
    print("Decoded control vector (throttle, steer, yaw, pitch, roll, jump, boost, handbrake):")
    for i, c in enumerate(parsed[0]):
        print(f"{i}: {c:.2f}")

def build_reward_function(phase=1):
    from mysim.reward_functions import CombinedReward
    from mysim.reward_functions.common_rewards import (
        EventReward, SpeedTowardBallReward, FaceBallReward, VelocityReward, InAirReward,
        VelocityBallToGoalReward, StrongHitReward
    )
    if phase == 1:
        return CombinedReward.from_zipped(
            (EventReward(touch=1), 50.0),
            (SpeedTowardBallReward(), 5.0),
            (FaceBallReward(), 1.0),
            (VelocityReward(), 0.3),
            (InAirReward(), 0.15)
        )
    elif phase == 2:
        return CombinedReward.from_zipped(
            (VelocityBallToGoalReward(), 3.0),
            (SpeedTowardBallReward(), 2.0),
            (FaceBallReward(), 0.5),
            (VelocityReward(), 0.3),
            (InAirReward(), 0.15)
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
    # from rlgym_sim.utils.reward_functions.common_rewards import SpeedTowardBallReward, \
    #     EventReward, FaceBallReward, InAirReward, VelocityBallToGoalReward, StrongHitReward, \
    #     VelocityReward, SaveBoostReward, AlignBallGoal, BoostPickupReward, RewardIfBehindBall, \
    #     SaveReward, JumpShotReward, PossessionReward, LiuDistanceBallToGoalReward, LiuDistancePlayerToBallReward, \
    #     PunishIfInNet, StealBoostReward
    # from rlgym_sim.utils.reward_functions.common_rewards import (
    #     VelocityPlayerToBallReward,
    #     VelocityBallToGoalReward,
    #     EventReward,
    #     FaceBallReward,
    #     InAirReward,
    #     StrongHitReward,
    #     SaveBoostReward,
    #     AlignBallGoal,
    #     BoostPickupReward,
    #     RewardIfBehindBall,
    #     SaveReward,
    #     JumpShotReward,
    #     PossessionReward,
    #     LiuDistanceBallToGoalReward,
    #     LiuDistancePlayerToBallReward,
    #     PunishIfInNet,
    #     StealBoostReward
    # )
    from mysim.obs_builders import DefaultObs
    from mysim.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
    from mysim.reward_functions.common_rewards.conditional_rewards import GoalIfTouchedLastConditionalReward
    from mysim import common_values
    from mysim.state_setters import RandomState
    from mysim.action_parsers import DiscreteAction
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
    from mysim.action_parsers.simple_lookup_discrete_action import SimpleLookupDiscreteAction


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

    # action_parser = DiscreteAction()
    # action_parser = ClipActionWrapper(ContinuousAction()) 

    # action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    # 1. Base discrete parser (90-action table)
    # ---------------- Debug --------------------------
    if global_debug_mode or debug_actions: 
        test = AdvancedLookupTableAction()
        print(test.parse_actions([0]))
        print(test.parse_actions([45]))
    # ---------------- Debug --------------------------
    lut = AdvancedLookupTableAction(
        throttle_bins=3,
        steer_bins=3,
        torque_subdivisions=2,   # good aerial coverage
        flip_bins=8,
        include_stalls=False
    )

    # # 2. Add a state-aware safety guard
    # #    (drops illegal actions like flips without flip, boost with 0 boost, etc.)
    # idle_idx = int(np.where((lut.lookup_table == 0).all(axis=1))[0][0])  # first all-zero row
    
    # # State-aware guard
    # forward_idx = find_forward_fallback_idx(lut)
    # safe_lut = StateAwareLUTWrapper(lut, fallback_index=forward_idx)

    # # Sticky short holds across steps
    # sticky = StickyButtonsWrapper(safe_lut, hold_ticks=3)

    # # collapse to a single (N_agents, 8) array per env.step()
    # collapse = CollapseToSingleTick(sticky)

    # # OUTERMOST: expand per tick for RocketSim
    # # action_parser = ExpandForRocketSim(collapse, tick_skip=tick_skip) # uncomment later
    # # 🔽 Outermost adapter ensures correct count and shape
    # action_parser = FinalRocketSimAdapter(collapse)

    # # #####################################################
    # # Debug: 10/19/25, debug StateAwareLUTWrapper inner parsers
    # if global_debug_mode or debug_actions:
    #     print("[debug] action_parser =", type(action_parser))
    #     if hasattr(action_parser, "inner"):
    #         print("[debug] inner =", type(action_parser.inner))
    #         if hasattr(action_parser.inner, "inner"):
    #             print("[debug] inner.inner =", type(action_parser.inner.inner))
    #             if hasattr(action_parser.inner.inner, "inner"):
    #                 print("[debug] inner.inner.inner =", type(action_parser.inner.inner.inner))
        
    # # #####################################################

    action_parser = SimpleLookupDiscreteAction()

    # 4. (Optional) small tick repeat if you really want to emulate frame-skip
    #    but keep it tiny — 1 or 2 only
    # action_parser = RepeatAction(action_parser, repeats=1)

    state_setter = RandomState(True, True, False)
    terminal_conditions = [GoalScoredCondition(), NoTouchTimeoutCondition(no_touch_timeout_ticks), 
                                       TimeoutCondition(game_timeout_ticks)]

    # 10-12-25 Basic starting reward function. Get the bot to touch the ball. Phase 1.
    reward_fn = build_reward_function(phase=1)

    # 10-19-25 Second reward function. Begin gradual transitioning of bot to push the ball into the opposing goal. Phase 1.5. from 0.5-1M steps.
    # reward_fn = CombinedReward.from_zipped(
    #     # 1️⃣ Continuous shaping
    #     (VelocityBallToGoalReward(), 3.0),   # move ball toward goal
    #     (SpeedTowardBallReward(), 2.0),      # keep car moving toward ball
    #     (FaceBallReward(), 0.5),             # orientation sanity
    #     (VelocityReward(), 0.3),             # general driving stability
    #     (InAirReward(), 0.15),               # preserve aerial capability

    #     # 2️⃣ Sparse / event-based
    #     (EventReward(
    #         goal=15.0,                       # strong signal for finishing
    #         concede=-8.0,                    # discourage own goals
    #         touch=5                          # touch still matters but less so
    #     ), 1.0)
    # )

    # # 10-19-25 Third reward function. Make bot to push the ball into the opposing goal. Phase 2.
    # reward_fn = CombinedReward.from_zipped(
    #     # 1️⃣ Continuous shaping
    #     (VelocityBallToGoalReward(), 8.0),   # move ball toward goal
    #     (SpeedTowardBallReward(), 2.0),      # keep car moving toward ball
    #     (FaceBallReward(), 0.5),             # orientation sanity
    #     (VelocityReward(), 0.3),             # general driving stability
    #     (InAirReward(), 0.15),               # preserve aerial capability

    #     # 2️⃣ Sparse / event-based
    #     (EventReward(
    #         goal=15.0,                       # strong signal for finishing
    #         concede=-8.0,                    # discourage own goals
    #         touch=0.05                       # touch still matters but tiny
    #     ), 1.0)
    # )

    # INITIAL REWARD FUNCTION. 3/1/25. GET BOT TO TOUCH THE BALL CONSISTENTLY
    # reward_fn = CombinedReward.from_zipped(
    #                                         #(EventReward(team_goal=1, concede=-1), 50.0), # Big reward for scoring goal
    #                                         (SaveReward(), 5.0), # Big reward for saving ball from going in net
    #                                         (PossessionReward(max_reward=1.0), 1.0), # Reward for having possession of the ball
    #                                         (BasicShotReward(), 5.0),
    #                                         (AirTouchReward(), 10.0),
    #                                         (DribblingReward(), 3.0),
    #                                         (JumpShotReward(), 1.0), # add some reward for jumping at the ball
    #                                         (StrongHitReward(), 3.0), # add some reward for hitting the ball harder                                            
    #                                         (VelocityBallToGoalReward(), 5.0),
    #                                         (LiuDistanceBallToGoalReward(), 2.0),
    #                                         (AlignBallGoal(), 1.0),
    #                                         (LiuDistancePlayerToBallReward(), 0.5),
    #                                         (VelocityPlayerToBallReward(), 0.75), # Move towards the ball!
    #                                         (BoostPickupReward(big_pad_reward=1.0, small_pad_reward=0.5), 0.75),
    #                                         (SaveBoostReward(), 0.25),
    #                                         (StealBoostReward(), 1.5),
    #                                         (FaceBallReward(), 0.05), # Make sure we don't start driving backward at the ball
    #                                         (InAirReward(), 0.004), # Make sure we don't forget how to jump
    #                                         (PunishIfInNet(), -1.0) # Punish the bot for being in the net
    # )
    # REWARD FUNCTION 2. 3/1/25. GET BOT TO BEGIN LEARNING TO SCORE THE BALL
    # reward_fn = CombinedReward.from_zipped(
    #     (StrongHitReward(max_reward=1.0), 50.0), # bigger reward for hitting the ball harder
    #     (AlignBallGoal(), 15),
    #     (BoostPickupReward(big_pad_reward=1.0, small_pad_reward=0.5), 7.5),
    #     (SaveBoostReward(), 5.0),
    #     (SpeedTowardBallReward(), 2.5),
    #     (FaceBallReward(), 1),
    #     (InAirReward(), 0.02)
    # )
    # REWARD FUNCTION 2. 3/1/25. GET BOT TO BEGIN LEARNING TO SCORE THE BALL
    # reward_fn = CombinedReward.from_zipped(
    #                         (GoalIfTouchedLastConditionalReward(), 54),
    #                         (VelocityBallToGoalReward(), 15),
    #                         (StrongHitReward(max_reward=1.0), 10.0), # bigger reward for hitting the ball harder
    #                         (EventReward(touch=1), 4), # Moderate reward for actually hitting the ball
    #                         (SpeedTowardBallReward(), 4), # Move towards the ball!
    #                         (FaceBallReward(), 0.6), # Make sure we don't start driving backward at the ball
    #                         (VelocityReward(), 0.07), # Make sure model is training
    #                         (InAirReward(), 0.015) # Make sure we don't forget how to jump
    # )
    # Default observation builder to replace with custom. Commented out 10/12/25
    # obs_builder = DefaultObs(
    #                        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
    #                                           1 / common_values.BACK_NET_Y, 
    #                                           1 / common_values.CEILING_Z]),
    #                        ang_coef=1 / np.pi,
    #                        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
    #                        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    from mysim.obs_builders.advanced_obs_plus import AdvancedObsPlus
    obs_builder = AdvancedObsPlus( #added 10/12/25
        max_allies=2,             # adjust to your team sizes
        max_opponents=3,          # adjust to your scrim sizes
        k_nearest_pads_me=4,
        k_nearest_pads_ball=2,
        stack_size=1,             # keep 1 for now; try 3–4 later
        include_prev_action=True, # matches your old behavior
        action_lookup=lut.lookup_table,)   # numpy array (N, 8)
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
        print("Controlled agents (spectator_ids):", len(env._match._spectator_ids))
        debug_controls_sample(action_parser, env) # This confirms whether your decoded LUT entries actually include throttle/boost patterns that make sense. 10/19/25 NJH

    # The most recent GameState lives directly on the Gym
    if global_debug_mode or debug_actions:
        state = env._prev_state
        if state is not None:
            print("Players in match:", len(state.players))
        else:
            print("Warning: env._prev_state is None (no snapshot yet)")

    if global_debug_mode or debug_actions:
        print("Action space:", env.action_space)
        a = env.action_space.sample()
        print("Sample action:", a)
        print("\n--- DEBUG ACTION SAMPLE ---")
        print("Action input to env.step:", a)
        parsed = env._match._action_parser.parse_actions(a, env._prev_state)
        print("Parsed action output:", parsed)
        obs, rew, done, info = env.step(a)
        term = done
        trunc = False
        print("Step OK ✅", rew, term, trunc)

    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    return env

shots = [
    # Shot 1: Stationary ball in the air, car positioned for an aerial
    {
        "ball": {
            "position": [0, 0, 500],  # Ball in the air
            "linear_velocity": [0, 0, 0],  # Ball is stationary
            "angular_velocity": [0, 0, 0]  # No spin
        },
        "cars": [
            {
                "position": [0, -2000, 17],  # Car positioned on the ground
                "rotation": [0, np.pi / 2, 0],  # Facing y+
                "linear_velocity": [0, 0, 0],  # Stationary
                "angular_velocity": [0, 0, 0],  # No spin
                "boost": 1.0  # Full boost
            }
        ]
    },
    # Shot 2: Ball stationary near the wall, car positioned for a wall shot
    {
        "ball": {
            "position": [-4000, 0, 500],  # Ball near the wall
            "linear_velocity": [0, 0, 0],  # Stationary
            "angular_velocity": [0, 0, 0]  # No spin
        },
        "cars": [
            {
                "position": [-3500, -400, 17],  # Car positioned on the same side
                "rotation": [0, 3 * np.pi / 4, 0], # Facing 45° between +y and -x
                "linear_velocity": [0, 0, 0],  # Stationary
                "angular_velocity": [0, 0, 0],  # No spin
                "boost": 1.0  # Full boost
            }
        ]
    },
    # Shot 3: Ball stationary near the wall, car positioned for a wall shot
    {
        "ball": {
            "position": [4000, 0, 500],  # Ball near the wall
            "linear_velocity": [0, 0, 0],  # Stationary
            "angular_velocity": [0, 0, 0]  # No spin
        },
        "cars": [
            {
                "position": [3500, -400, 17],  # Car positioned on the same side
                "rotation": [0, np.pi / 4, 0], # Facing 45° between +y and +x
                "linear_velocity": [0, 0, 0],  # Stationary
                "angular_velocity": [0, 0, 0],  # No spin
                "boost": 1.0  # Full boost
            }
        ]
    },
    # Shot 4: Ball higher in the air, car positioned for an aerial
    {
        "ball": {
            "position": [0, 0, 1000],  # Ball higher in the air
            "linear_velocity": [0, 0, 0],  # Stationary
            "angular_velocity": [0, 0, 0]  # No spin
        },
        "cars": [
            {
                "position": [0, -2500, 17],  # Car positioned on the ground
                "rotation": [0, np.pi, 0],  # Facing forward, y+
                "linear_velocity": [0, 0, 0],  # Stationary
                "angular_velocity": [0, 0, 0],  # No spin
                "boost": 1.0  # Full boost
            }
        ]
    }

]

def build_rocketsim_env_training():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import SpeedTowardBallReward, \
        EventReward, FaceBallReward, InAirReward, VelocityBallToGoalReward, StrongHitReward, \
        VelocityReward, SaveBoostReward, AlignBallGoal, BoostPickupReward, RewardIfBehindBall, \
        SaveReward, JumpShotReward, PossessionReward, LiuDistanceBallToGoalReward, LiuDistancePlayerToBallReward, \
        PunishIfInNet, StealBoostReward, BasicAerialReward, BasicWallHitReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    # from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition # old (deprecated and now missing)
    from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils.reward_functions.common_rewards.conditional_rewards import GoalIfTouchedLastConditionalReward
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.state_setters.training_pack_state import TrainingPackStateSetter
    from rlgym_sim.utils.state_setters import RandomState
    #from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction # old (deprecated and now missing)
    from rlgym_sim.utils.action_parsers import DiscreteAction

    spawn_opponents = False #
    game_tick_rate = 120
    tick_skip = 8
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 10
    no_touch_timeout_ticks = int(round(no_touch_timeout_seconds * game_tick_rate / tick_skip))
    game_timeout_seconds = 300
    game_timeout_ticks = int(round(game_timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()

    # state_setter = RandomState(True, True, False)
    # With variation enabled
    # state_setter = TrainingPackStateSetter(shots=shots, variation=True)

    # Without variation
    state_setter = TrainingPackStateSetter(shots=shots, variation=False)
    terminal_conditions = [GoalScoredCondition(), NoTouchTimeoutCondition(no_touch_timeout_ticks), 
                                       TimeoutCondition(game_timeout_ticks)]


    # INITIAL REWARD FUNCTION. 3/1/25. GET BOT TO TOUCH THE BALL CONSISTENTLY
    reward_fn = CombinedReward.from_zipped(
                                            (EventReward(team_goal=1, concede=-1), 50.0), # Big reward for scoring goal
                                            (SaveReward(), 5.0), # Big reward for saving ball from going in net
                                            (PossessionReward(max_reward=1.0), 1.0), # Reward for having possession of the ball
                                            (BasicShotReward(), 5.0),
                                            (BasicWallHitReward(), 10.0),
                                            (BasicAerialReward(), 15.0),
                                            (DribblingReward(), 3.0),
                                            (JumpShotReward(), 1.0), # add some reward for jumping at the ball
                                            (StrongHitReward(), 3.0), # add some reward for hitting the ball harder                                            
                                            (VelocityBallToGoalReward(), 2.5),
                                            (LiuDistanceBallToGoalReward(), 0.5),
                                            #(AlignBallGoal(), 1.0),
                                            (LiuDistancePlayerToBallReward(), 1.0),
                                            (SpeedTowardBallReward(), 0.75), # Move towards the ball!
                                            # (BoostPickupReward(big_pad_reward=1.0, small_pad_reward=0.5), 0.75),
                                            # (SaveBoostReward(), 0.25),
                                            # (StealBoostReward(), 1.5),
                                            (FaceBallReward(), 0.2), # Make sure we don't start driving backward at the ball
                                            (InAirReward(), 0.005), # Make sure we don't forget how to jump
                                            (PunishIfInNet(), -1.0) # Punish the bot for being in the net
    )
    # REWARD FUNCTION 2. 3/1/25. GET BOT TO BEGIN LEARNING TO SCORE THE BALL
    # reward_fn = CombinedReward.from_zipped(
    #     (StrongHitReward(max_reward=1.0), 50.0), # bigger reward for hitting the ball harder
    #     (AlignBallGoal(), 15),
    #     (BoostPickupReward(big_pad_reward=1.0, small_pad_reward=0.5), 7.5),
    #     (SaveBoostReward(), 5.0),
    #     (SpeedTowardBallReward(), 2.5),
    #     (FaceBallReward(), 1),
    #     (InAirReward(), 0.02)
    # )
    # rewards_to_combine = (
    #                         (EventReward(team_goal=1, concede=-1)),
    #                         EventReward(touch=1), # Giant reward for actually hitting the ball
    #                         SpeedTowardBallReward(), # Move towards the ball!
    #                         FaceBallReward(), # Make sure we don't start driving backward at the ball
    #                         InAirReward() # Make sure we don't forget how to jump
    # )
    # reward_weights = (50, 5, 1, 0.01)
    # REWARD FUNCTION 2. 3/1/25. GET BOT TO BEGIN LEARNING TO SCORE THE BALL
    # reward_fn = CombinedReward.from_zipped(
    #                         (GoalIfTouchedLastConditionalReward(), 54),
    #                         (VelocityBallToGoalReward(), 15),
    #                         (StrongHitReward(max_reward=1.0), 10.0), # bigger reward for hitting the ball harder
    #                         (EventReward(touch=1), 4), # Moderate reward for actually hitting the ball
    #                         (SpeedTowardBallReward(), 4), # Move towards the ball!
    #                         (FaceBallReward(), 0.6), # Make sure we don't start driving backward at the ball
    #                         (VelocityReward(), 0.07), # Make sure model is training
    #                         (InAirReward(), 0.015) # Make sure we don't forget how to jump
    # )

    obs_builder = DefaultObs(
                           pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
                                              1 / common_values.BACK_NET_Y, 
                                              1 / common_values.CEILING_Z]),
                           ang_coef=1 / np.pi,
                           lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                           ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)
    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    return env

def choose_checkpoint(root):
    runs = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    print("\nAvailable checkpoints:")
    for i, run in enumerate(runs):
        ckpt = latest_checkpoint_folder(run)
        if ckpt:
            print(f"{i}: {run.name} -> {ckpt.name}")
    sel = input("Select checkpoint index to resume (or press Enter for latest): ").strip()
    if sel.isdigit():
        run = runs[int(sel)]
        return run, latest_checkpoint_folder(run)
    return runs[0], latest_checkpoint_folder(runs[0])

def run_training():
    if global_debug_mode or debug_learning:
        # Torch / GPU info
        # You want cuda available: True and a nonzero device count.
        print("[torch] cuda available:", torch.cuda.is_available())
        print("[torch] device count:", torch.cuda.device_count())
        print("[torch] version:", torch.__version__, "cuda:", torch.version.cuda)
        print("[torch] current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_logger = ExampleLogger()

    # 64 processes for training
    # 4 for debug
    policy_layers = [1024, 1024, 512, 512]
    critic_layers = [1024, 1024, 512, 512]
    if global_debug_mode or debug_learning:
        n_proc = 2
    else: 
        n_proc = 32
    save_every = 50_000   # or whatever you like


    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    # Configurable checkpoint root via env var, portable. Used for loading latest training checkpoint.
    # DEFAULT_CKPT_ROOT = Path("training/checkpoints")  # repo-relative
    # CKPT_ROOT = Path(os.environ.get("RLGYM_PPO_CKPT_ROOT", DEFAULT_CKPT_ROOT))
    # latest_checkpoint_dir = get_latest_checkpoint(CKPT_ROOT)
    # print(f"Latest checkpoint directory: {latest_checkpoint_dir}")
    # Here is a great starting point for a bot from scratch.
    # Format: (reward, weight)
    # rewards = (
    # 	(EventReward(touch=1), 50), # Giant reward for actually hitting the ball
    # 	(SpeedTowardBallReward(), 5), # Move towards the ball!
    # 	(FaceBallReward(), 1), # Make sure we don't start driving backward at the ball
    # 	(AirReward(), 0.15) # Make sure we don't forget how to jump
    # )
    # # NOTE: SpeedTowardBallReward and AirReward can be found in the rewards section of this guide
    # Always build env once to determine obs_dim
    env_for_shape = build_rocketsim_env()
    obs_dim = int(env_for_shape.observation_space.shape[0])

    if global_debug_mode or debug_learning:
        print(f"[init] obs_dim = {obs_dim}")

    START_FRESH = False  # <-- toggle this to True when you want to start from scratch

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        [p for p in CHECKPOINT_ROOT.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    resume_run = None
    resume_ckpt = None

    if not START_FRESH:
        for rd in run_dirs:
            meta = read_meta(rd)
            if shapes_match(meta, obs_dim, policy_layers, critic_layers):
                latest = latest_checkpoint_folder(rd)
                if latest:
                    resume_run = rd
                    resume_ckpt = latest
                    break

    if resume_run and not START_FRESH:
        run_dir = resume_run
        checkpoint_load_folder = str(resume_ckpt)
        print(f"[resume] Using run_dir = {run_dir}")
        print(f"[resume] Loading checkpoint = {checkpoint_load_folder}")
    else:
        run_dir = make_run_dir()
        checkpoint_load_folder = None
        print(f"[new] Starting from scratch — run_dir = {run_dir}")

    # Optional: manually override which checkpoint to resume from
    CUSTOM_CKPT_PATH = None
    # Example:
    # CUSTOM_CKPT_PATH = "data/checkpoints/rlgym-ppo-run-20251019-013412/checkpoints/ckpt_150000"

    if CUSTOM_CKPT_PATH and not START_FRESH:
        checkpoint_load_folder = CUSTOM_CKPT_PATH
        run_dir = Path(CUSTOM_CKPT_PATH).parents[1]  # one up from /checkpoints/ckpt_xxxxxx
        print(f"[manual resume] Using custom checkpoint: {checkpoint_load_folder}")
    
    INTERACTIVE_RESUME = False  # Set to True to choose from available checkpoints interactively
    if not START_FRESH and INTERACTIVE_RESUME:
        run_dir, resume_ckpt = choose_checkpoint(CHECKPOINT_ROOT)

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger, # Leave this empty for now.
                      ppo_batch_size=50_000,  # batch size - much higher than 300K doesn't seem to help most people. Keep this equal to ts_per_iteration. 
                      # 50k is good for early learning. 100k after bot can hit the ball. Once bot is shooting & scoring, set to 200k or 300k.
                      policy_layer_sizes=policy_layers,  # policy network
                      critic_layer_sizes=critic_layers,  # critic network
                      device=device,  # "cpu" or "cuda"
                      ts_per_iteration=50_000,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=150_000,  # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=12_500,  # minibatch size - set this as high as your GPU can handle
                      ppo_ent_coef=0.01,  # entropy coefficient - this determines the impact of exploration
                      policy_lr=2e-4,  # policy learning rate
                      critic_lr=2e-4,  # critic learning rate
                      #Bot that can't score yet: 2e-4
                      #Bot that is actually trying to score on its opponent: 1e-4
                      #Bot that is learning outplay mechanics (dribbling and flicking, air dribbles, etc.): 0.8e-4 or lower
                      ppo_epochs=3,   # number of PPO epochs. This is how many times the learning phase is repeated on the same batch of data. I recommend 2 or 3.
                      # saving
                      checkpoints_save_folder=str(run_dir),
                      save_every_ts=save_every,  # save every 100k steps
                      # loading (resume if compatible)
                      checkpoint_load_folder=checkpoint_load_folder, #load from resume_ckpt (the numbered step subfolder), or None for a fresh run. debug mode added above.
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      timestep_limit=10e15,  # Train for 1 quadrillion steps
                      log_to_wandb=True, # Set this to True if you want to use Weights & Biases for logging.
                      render=True,  # Set this to True if you want to render the environment.
                      render_delay=0.02)  # Set this to the delay between frames when rendering.
    import time
    import shutil
    # === Milestone Checkpoint Configuration ===
    MILESTONE_INTERVAL = 6_000_000   # Save milestone every 6M timesteps
    MILESTONE_DIR = CHECKPOINT_ROOT / "milestones"
    MILESTONE_DIR.mkdir(parents=True, exist_ok=True)
    learner._next_milestone = MILESTONE_INTERVAL
    learner._milestone_root = MILESTONE_DIR

    # Keep reference to the original internal save() method
    orig_save = getattr(learner, "save", None)

    def wrapped_save(cumulative_timesteps: int):
        """
        Wrap the original Learner.save() so that every time the learner saves
        (normally every `save_every_ts` steps), we also check for milestone triggers.
        """
        if orig_save is not None:
            orig_save(cumulative_timesteps)

        total_steps = cumulative_timesteps
        if total_steps >= learner._next_milestone:
            milestone_dir = learner._milestone_root / f"{int(total_steps // 1_000_000):03d}M"
            milestone_dir.mkdir(parents=True, exist_ok=True)
            print(f"[milestone] ⏱ Auto-saving at {total_steps:,} steps → {milestone_dir}")

            # Duplicate the just-saved checkpoint into milestones
            src_folder = os.path.join(learner.checkpoints_save_folder, str(total_steps))
            shutil.copytree(src_folder, milestone_dir, dirs_exist_ok=True)

            # === Collect metrics for meta logging ===
            avg_reward = getattr(learner.agent, "average_reward", None)
            ppo = getattr(learner, "ppo_learner", None)
            avg_entropy = getattr(ppo, "last_policy_entropy", None)
            avg_kl = getattr(ppo, "last_kl_divergence", None)

            # Write enhanced meta info for this milestone
            meta_extra = {
                "phase": "touch_training",
                "total_timesteps": int(total_steps),
                "avg_reward": float(avg_reward) if avg_reward is not None else None,
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

            learner._next_milestone += MILESTONE_INTERVAL

    # Monkey-patch the learner’s save() method
    learner.save = wrapped_save

    # === Start training ===
    learner.learn()


    # env = build_rocketsim_env_training()
    # print(f"Observation space size: {env.observation_space.shape[0]}")
    # env = build_rocketsim_env()
    # print(f"Observation space size: {env.observation_space.shape[0]}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    run_training()
    