import numpy as np

from rlgymbotv2.mysim import RewardFunction, math
from rlgymbotv2.mysim.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, \
    BLUE_GOAL_BACK, BALL_MAX_SPEED, BACK_WALL_Y, BALL_RADIUS, BACK_NET_Y
from rlgymbotv2.mysim.gamestates import GameState, PlayerData

class BallYCoordinateReward(RewardFunction):
    def __init__(self, exponent=1):
        # Exponent should be odd so that negative y -> negative reward
        self.exponent = exponent

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            return (state.ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent
        else:
            return (state.inverted_ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent
        
class BasicShotReward(RewardFunction):
    def __init__(self, max_reward=1.0):
        super().__init__()
        self.max_reward = max_reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position

        # Calculate the velocity of the ball towards the opponent's goal
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / BALL_MAX_SPEED
        velocity_towards_goal = float(np.dot(norm_pos_diff, norm_vel))

        # Calculate the reward based on the velocity towards the goal and the position of the ball
        reward = velocity_towards_goal * self.max_reward
        return reward
    
class ClearReward(RewardFunction):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player, state, previous_action):
        if player.team_num == BLUE_TEAM:
            own_goal = np.array(BLUE_GOAL_BACK)
        else:
            own_goal = np.array(ORANGE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = state.ball.position - own_goal

        norm_pos = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / BALL_MAX_SPEED

        # >0 means ball is moving AWAY from own goal
        away_speed = float(np.dot(norm_vel, norm_pos))
        return max(0.0, away_speed) * self.scale

class LiuDistanceBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(state.ball.position - objective) - (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        return np.exp(-0.5 * dist / BALL_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196
        
class NegativeVelocityTowardOwnGoalReward(RewardFunction):
    """
    Penalizes ball velocity toward the player's own goal.
    Encourages clearing, shadowing, and challenge timing.
    """
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        # Identify own goal center
        if player.team_num == BLUE_TEAM:
            own_goal = np.array(BLUE_GOAL_BACK)
        else:
            own_goal = np.array(ORANGE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = own_goal - state.ball.position

        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / BALL_MAX_SPEED

        # Positive when ball moves *toward own goal*
        toward_own_goal = float(np.dot(norm_pos_diff, norm_vel))

        # Only punish the bad part (don’t reward clearing here)
        return -max(0.0, toward_own_goal) * self.scale

        
class SaveReward(RewardFunction):
    def __init__(self, save_radius=1000, max_reward=1.0):
        """
        Reward function for detecting and rewarding saves.
        :param save_radius: Circle's radius around the goal in the x-y plane to consider for a save.
        :param max_reward: Maximum reward for a successful save.
        """
        super().__init__()
        self.save_radius = save_radius
        self.max_reward = max_reward
        self.last_ball_velocity = None  # To track the ball's velocity in the previous frame
    def reset(self, initial_state: GameState):
        """
        Reset the reward function at the start of an episode.
        :param initial_state: The initial game state.
        """
        self.last_ball_velocity = initial_state.ball.linear_velocity
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Calculate the reward for a save.
        :param player: The player data.
        :param state: The current game state.
        :param previous_action: The player's previous action.
        :return: The reward for the save.
        """
        # Determine the goal position based on the player's team
        if player.team_num == BLUE_TEAM:
            goal_center = np.array(BLUE_GOAL_BACK)
            goal_direction = np.array([0, 1, 0])  # Positive Y direction
        else:
            goal_center = np.array(ORANGE_GOAL_BACK)
            goal_direction = np.array([0, -1, 0])  # Negative Y direction

        # Check if the ball is within the save radius
        ball_position = state.ball.position
        distance_to_goal = np.linalg.norm(ball_position[:2] - goal_center[:2])  # Only consider x-y plane
        if distance_to_goal > self.save_radius:
            return 0.0  # No reward if the ball is outside the save radius

        # Assign ball velocity at the top to avoid referencing before assignment
        ball_velocity = state.ball.linear_velocity

        # Check if the player has redirected the ball away from the goal
        if player.ball_touched:
            last_velocity_toward_goal = np.dot(self.last_ball_velocity, goal_direction)
            current_velocity_toward_goal = np.dot(ball_velocity, goal_direction)
            if last_velocity_toward_goal > 0 and current_velocity_toward_goal < 0:
                # The ball was moving toward the goal and is now moving away
                return self.max_reward

        # Check if the ball is moving toward the player's goal
        velocity_toward_goal = np.dot(ball_velocity, goal_direction)
        if velocity_toward_goal <= 0:
            return 0.0  # No reward if the ball is not moving toward the goal

        # Update the last ball velocity for the next frame
        self.last_ball_velocity = ball_velocity
        return 0.0

class ShotSetupReward(RewardFunction):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def reset(self, initial_state): pass

    def get_reward(self, player, state, previous_action):
        if player.team_num == BLUE_TEAM:
            target_goal = np.array(ORANGE_GOAL_BACK)
        else:
            target_goal = np.array(BLUE_GOAL_BACK)

        car_pos = player.car_data.position
        ball_pos = state.ball.position

        # Behind ball?
        ball_forward_vec = ball_pos - target_goal
        car_forward_vec = car_pos - ball_pos

        behind_factor = np.dot(
            ball_forward_vec / np.linalg.norm(ball_forward_vec),
            car_forward_vec / np.linalg.norm(car_forward_vec)
        )

        if behind_factor > 0.6:
            return behind_factor * self.scale

        return 0.0
    
class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, norm_vel))