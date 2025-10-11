################################
# Date: 10/10/2025
# Revision: 1.0.0
# Description: Initial Release
# Title: ppo_learner.py
# Author: NJH
################################
import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from pathlib import Path
import os
from rlgym_sim.utils.reward_functions.common_rewards import *
from rlgym_ppo import Learner

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

# Move these functions to the top of example.py
def get_latest_directory(directory):
    """Returns the most recently modified directory inside the given directory."""
    list_of_dirs = [d for d in os.scandir(directory) if d.is_dir()]
    if not list_of_dirs:
        return None
    latest_dir = max(list_of_dirs, key=lambda d: d.stat().st_mtime).path
    return latest_dir

def get_latest_checkpoint(run_directory):
    """Returns the most recently modified 'checkpoint' directory inside the latest run directory."""
    latest_run_dir = get_latest_directory(run_directory)
    if not latest_run_dir:
        return None

    latest_checkpoint_dir = get_latest_directory(latest_run_dir)  # Find latest subdirectory inside
    return latest_checkpoint_dir

def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import SpeedTowardBallReward, \
        EventReward, FaceBallReward, InAirReward, VelocityBallToGoalReward, StrongHitReward, \
        VelocityReward, SaveBoostReward, AlignBallGoal, BoostPickupReward, RewardIfBehindBall, \
        SaveReward, JumpShotReward, PossessionReward, LiuDistanceBallToGoalReward, LiuDistancePlayerToBallReward, \
        PunishIfInNet, StealBoostReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    # from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition
    from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils.reward_functions.common_rewards.conditional_rewards import GoalIfTouchedLastConditionalReward
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.state_setters import RandomState
    #from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym_sim.utils.action_parsers import DiscreteAction

    spawn_opponents = True
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

    action_parser = DiscreteAction()

    # action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    state_setter = RandomState(True, True, False)
    terminal_conditions = [GoalScoredCondition(), NoTouchTimeoutCondition(no_touch_timeout_ticks), 
                                       TimeoutCondition(game_timeout_ticks)]


    # INITIAL REWARD FUNCTION. 3/1/25. GET BOT TO TOUCH THE BALL CONSISTENTLY
    reward_fn = CombinedReward.from_zipped(
                                            (EventReward(team_goal=1, concede=-1), 50.0), # Big reward for scoring goal
                                            (SaveReward(), 5.0), # Big reward for saving ball from going in net
                                            (PossessionReward(max_reward=1.0), 1.0), # Reward for having possession of the ball
                                            (BasicShotReward(), 5.0),
                                            (AirTouchReward(), 10.0),
                                            (DribblingReward(), 3.0),
                                            (JumpShotReward(), 1.0), # add some reward for jumping at the ball
                                            (StrongHitReward(), 3.0), # add some reward for hitting the ball harder                                            
                                            (VelocityBallToGoalReward(), 5.0),
                                            (LiuDistanceBallToGoalReward(), 2.0),
                                            (AlignBallGoal(), 1.0),
                                            (LiuDistancePlayerToBallReward(), 0.5),
                                            (SpeedTowardBallReward(), 0.75), # Move towards the ball!
                                            (BoostPickupReward(big_pad_reward=1.0, small_pad_reward=0.5), 0.75),
                                            (SaveBoostReward(), 0.25),
                                            (StealBoostReward(), 1.5),
                                            (FaceBallReward(), 0.05), # Make sure we don't start driving backward at the ball
                                            (InAirReward(), 0.004), # Make sure we don't forget how to jump
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
    # from rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition
    from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils.reward_functions.common_rewards.conditional_rewards import GoalIfTouchedLastConditionalReward
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.state_setters.training_pack_state import TrainingPackStateSetter
    from rlgym_sim.utils.state_setters import RandomState
    #from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym_sim.utils.action_parsers import DiscreteAction

    spawn_opponents = True
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

if __name__ == "__main__":
    metrics_logger = ExampleLogger()

    # 64 processes for training
    # 4 for debug
    n_proc = 64

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    # Configurable checkpoint root via env var, portable. Used for loading latest training checkpoint.
    DEFAULT_CKPT_ROOT = Path("training/checkpoints")  # repo-relative
    CKPT_ROOT = Path(os.environ.get("RLGYM_PPO_CKPT_ROOT", DEFAULT_CKPT_ROOT))
    latest_checkpoint_dir = get_latest_checkpoint(CKPT_ROOT)
    print(f"Latest checkpoint directory: {latest_checkpoint_dir}")

    # learner = Learner(build_rocketsim_env,
    #                   n_proc=n_proc,
    #                   min_inference_size=min_inference_size,
    #                   metrics_logger=metrics_logger,
    #                   ppo_batch_size=150000,
    #                   ts_per_iteration=50000,
    #                   exp_buffer_size=150000,
    #                   ppo_minibatch_size=50000,
    #                   ppo_ent_coef=0.01,
    #                   ppo_epochs=1,
    #                   add_unix_timestamp=False,
    #                   #checkpoint_load_folder=latest_checkpoint,
    #                   policy_layer_sizes=[512,512,256,256],
    #                   critic_layer_sizes=[512,512,256,256],
    #                   policy_lr=2e-4,
    #                   critic_lr=2e-4,
    #                   standardize_returns=True,
    #                   standardize_obs=False,
    #                   save_every_ts=100_000,
    #                   timestep_limit=1.2e8, # limits cumulative timesteps to stop before we begin pushing the ball constantly
    #                   log_to_wandb=True,
    #                   render=True,
    #                   render_delay=0.02)
    # learner.learn()

    learner = Learner(build_rocketsim_env_training,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger, # Leave this empty for now.
                      ppo_batch_size=100_000,  # batch size - much higher than 300K doesn't seem to help most people
                      policy_layer_sizes=[512, 512, 256, 256],  # policy network
                      critic_layer_sizes=[512, 512, 256, 256],  # critic network
                      ts_per_iteration=100_000,  # timesteps per training iteration - set this equal to the batch size
                      exp_buffer_size=300_000,  # size of experience buffer - keep this 2 - 3x the batch size
                      ppo_minibatch_size=50_000,  # minibatch size - set this as high as your GPU can handle
                      ppo_ent_coef=0.01,  # entropy coefficient - this determines the impact of exploration
                      policy_lr=5e-5,  # policy learning rate
                      critic_lr=5e-5,  # critic learning rate
                      ppo_epochs=2,   # number of PPO epochs
                      checkpoint_load_folder=latest_checkpoint_dir,  # Load the latest checkpoint. comment this out for debug
                      standardize_returns=True, # Don't touch these.
                      standardize_obs=False, # Don't touch these.
                      save_every_ts=100_000,  # save every 100k steps
                      timestep_limit=100_000_000_000,  # Train for 100B steps
                      log_to_wandb=True, # Set this to True if you want to use Weights & Biases for logging.
                      render=True,  # Set this to True if you want to render the environment.
                      render_delay=0.02)  # Set this to the delay between frames when rendering.
    learner.learn()
    # env = build_rocketsim_env_training()
    # print(f"Observation space size: {env.observation_space.shape[0]}")
    # env = build_rocketsim_env()
    # print(f"Observation space size: {env.observation_space.shape[0]}")