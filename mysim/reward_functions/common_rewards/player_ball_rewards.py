import numpy as np

from mysim import RewardFunction, math
from mysim.common_values import BALL_RADIUS, CAR_MAX_SPEED
from mysim.gamestates import GameState, PlayerData
from mysim.common_values import CEILING_Z, BLUE_TEAM, ORANGE_GOAL_BACK, BLUE_GOAL_BACK

class AirTouchReward(RewardFunction):
    """
    Reward an aerial touch: if the player touches the ball while airborne,
    scale by ball height (higher touches worth more).
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if not player.ball_touched:
            return 0.0
        # Consider it an "air" touch if player isn't on ground at the moment of touch
        is_air = not player.on_ground
        if not is_air:
            return 0.0
        height_frac = float(np.clip(state.ball.position[2] / CEILING_Z, 0.0, 1.0))
        return height_frac

class BasicAerialReward(RewardFunction):
    """
    Encourage simple aerial behaviors near the ball:
      - being airborne (proxy air-time),
      - nose pitched up,
      - boosting while moving upward,
      - velocity aligned toward the ball.
    Uses internal per-player trackers; no custom PlayerData fields required.
    """
    def __init__(self, max_reward: float = 1.0, max_air_steps: int = 90):
        """
        :param max_reward: cap for total reward from this function
        :param max_air_steps: steps to consider "full" airborne time (≈6s if 15Hz)
        """
        self.max_reward = float(max_reward)
        self.max_air_steps = int(max_air_steps)
        self._air_steps = {}        # car_id -> int
        self._last_boost = {}       # car_id -> float

    def reset(self, initial_state: GameState):
        self._air_steps.clear()
        self._last_boost.clear()

    def _is_boosting(self, player: PlayerData) -> bool:
        cid = player.car_id
        prev = self._last_boost.get(cid, player.boost_amount)
        # decreasing boost amount => using boost (simple, robust heuristic)
        boosting = player.boost_amount < prev
        self._last_boost[cid] = player.boost_amount
        return boosting

    def _air_step_frac(self, player: PlayerData) -> float:
        cid = player.car_id
        steps = self._air_steps.get(cid, 0)
        if player.on_ground:
            steps = 0
        else:
            steps += 1
        self._air_steps[cid] = steps
        return min(1.0, steps / max(1, self.max_air_steps))

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Only evaluate when close to the ball (avoid rewarding random aerials far away)
        dist = float(np.linalg.norm(player.car_data.position - state.ball.position))
        if dist > 500.0:
            # still advance trackers so state is consistent next tick
            _ = self._is_boosting(player)
            _ = self._air_step_frac(player)
            return 0.0

        reward = 0.0

        # Airborne component
        air_frac = self._air_step_frac(player)
        reward += 0.2 * air_frac * self.max_reward

        # Nose up (encourage upward pitch)
        forward = player.car_data.forward()               # unit vector
        upward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        nose_up_alignment = float(np.dot(forward, upward))  # [-1, 1]
        if nose_up_alignment > 0.5:
            reward += 0.2 * self.max_reward

        # Boosting while moving upward
        lv = player.car_data.linear_velocity
        moving_up = lv[2] > 0.0
        if self._is_boosting(player) and moving_up:
            reward += 0.3 * self.max_reward

        # Velocity aligned toward the ball
        dir_to_ball = state.ball.position - player.car_data.position
        norm_v = np.linalg.norm(lv)
        norm_d = np.linalg.norm(dir_to_ball)
        if norm_v > 1e-6 and norm_d > 1e-6:
            direction_to_ball = dir_to_ball / norm_d
            alignment = float(np.dot(lv / norm_v, direction_to_ball))  # [-1, 1]
            if alignment > 0.8:
                reward += 0.3 * self.max_reward

        # Clamp to max_reward
        return float(min(self.max_reward, max(0.0, reward)))
    
class BasicWallHitReward(RewardFunction):
    def __init__(self, max_reward=1.0):
        """
        Rewards the bot for performing basic wall hit mechanics.
        :param max_reward: Maximum reward for successful wall hit behavior.
        """
        super().__init__()
        self.max_reward = max_reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0

        # Check if the bot is near the wall
        car_position = player.car_data.position
        if abs(car_position[0]) > 3500:  # Near the side walls
            reward += 0.2 * self.max_reward

        # Check if the bot is within 500 units of the ball
        distance_to_ball = np.linalg.norm(player.car_data.position - state.ball.position)
        if distance_to_ball > 500:
            return reward

        # Check if the bot is aligned to face the ball
        ball_position = state.ball.position
        direction_to_ball = ball_position - car_position
        direction_to_ball /= np.linalg.norm(direction_to_ball)  # Normalize
        forward_vector = player.car_data.forward()
        alignment = np.dot(forward_vector, direction_to_ball)
        if alignment > 0.8:  # Facing the ball
            reward += 0.3 * self.max_reward

        # Check if the bot has jumped off the wall
        if not player.on_ground and abs(car_position[0]) > 3500:
            reward += 0.2 * self.max_reward

        # Reward for hitting the ball
        if player.ball_touched:
            reward += 0.3 * self.max_reward

        return reward

class ControlReward(RewardFunction):
    def __init__(self, max_reward=1.0):
        """
        Rewards the bot for controlling the ball strategically while considering the opponent's position.
        :param max_reward: Maximum reward for successful control.
        """
        super().__init__()
        self.max_reward = max_reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Determine the goal position based on the player's team
        if player.team_num == BLUE_TEAM:
            goal_position = np.array(BLUE_GOAL_BACK)
        else:
            goal_position = np.array(ORANGE_GOAL_BACK)

        # Find the opponent
        opponent = next(p for p in state.players if p.team_num != player.team_num)

        # Check if the ball is being rolled toward the corner
        ball_position = state.ball.position
        ball_velocity = state.ball.linear_velocity
        goal_to_ball = ball_position[:2] - goal_position[:2]
        goal_to_ball_direction = goal_to_ball / np.linalg.norm(goal_to_ball)

        # Reward if the ball is moving away from the goal and toward the corner
        if np.dot(ball_velocity[:2], goal_to_ball_direction) < 0:
            return self.max_reward / 2

        # Reward if the bot cuts into the ball to change its direction away from the opponent
        opponent_position = opponent.car_data.position
        ball_to_opponent = opponent_position[:2] - ball_position[:2]
        ball_to_opponent_direction = ball_to_opponent / np.linalg.norm(ball_to_opponent)

        if player.ball_touched and np.dot(ball_velocity[:2], ball_to_opponent_direction) < 0:
            return self.max_reward
        return 0.0

class DribblingReward(RewardFunction):
    def __init__(self, max_reward=1.0):
        """
        Rewards the bot for dribbling the ball.
        :param max_reward: Maximum reward for successful dribbling.
        """
        super().__init__()
        self.max_reward = max_reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Check if the ball is on top of the car
        ball_position = state.ball.position
        car_position = player.car_data.position
        ball_height = ball_position[2] - car_position[2]

        # Ball should be close to the car in the x-y plane and slightly above it in z
        if np.linalg.norm(ball_position[:2] - car_position[:2]) < BALL_RADIUS and 0.5 < ball_height < 2.0:
            # Check if the ball is moving with the car
            ball_velocity = state.ball.linear_velocity
            car_velocity = player.car_data.linear_velocity
            velocity_diff = np.linalg.norm(ball_velocity - car_velocity)

            # Reward if the ball is moving with the car
            if velocity_diff < 500:  # Adjust threshold as needed
                return self.max_reward
        return 0.0

class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))

class JumpShotReward(RewardFunction):
    """
    Reward when the player touches the ball shortly after a jump and while airborne.
    Tracks 'recent jump' per player internally; no custom PlayerData fields required.

    Assumes DiscreteAction layout: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    so the jump button is index 5 in previous_action.
    """
    def __init__(self, max_steps_since_jump: int = 90, jump_button_index: int = 5):
        """
        :param max_steps_since_jump: how many env steps after a jump still count as 'recent'
                                     (~90 steps ≈ ~6s if your step rate is ~15 Hz)
        :param jump_button_index: index of the jump button in the action array
        """
        self.max_steps = int(max_steps_since_jump)
        self.jump_idx = int(jump_button_index)
        self._since_jump = {}  # car_id -> steps since last jump (airborne)

    def reset(self, initial_state: GameState):
        self._since_jump.clear()

    def _update_since_jump(self, player: PlayerData, previous_action) -> int:
        cid = player.car_id
        steps = self._since_jump.get(cid, self.max_steps + 1)

        # Grounded resets the counter (we only care about airborne jump windows)
        if player.on_ground:
            steps = self.max_steps + 1
        else:
            # Detect a jump press in the last action
            jumped = False
            if previous_action is not None:
                try:
                    val = previous_action[self.jump_idx]
                    jumped = bool(val >= 1 or val > 0.5)  # works for discrete/binary
                except Exception:
                    jumped = False

            steps = 0 if jumped else min(self.max_steps + 1, steps + 1)

        self._since_jump[cid] = steps
        return steps

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        steps = self._update_since_jump(player, previous_action)

        if player.ball_touched and (not player.on_ground) and steps <= self.max_steps:
            # Scale by height to prefer higher jump touches
            height_frac = float(np.clip(player.car_data.position[2] / CEILING_Z, 0.0, 1.0))
            return height_frac

        return 0.0
    
class LiuDistancePlayerToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        return np.exp(-0.5 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

class OffsidePunishment(RewardFunction):
    def __init__(self, penalty=-1.0):
        """
        Penalizes the bot for being farther from its own goal than the ball.
        :param penalty: The negative reward to apply when offside.
        """
        super().__init__()
        self.penalty = penalty

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Determine the goal position based on the player's team
        if player.team_num == BLUE_TEAM:
            goal_position = np.array(BLUE_GOAL_BACK)
        else:
            goal_position = np.array(ORANGE_GOAL_BACK)

        # Calculate distances to the goal
        player_distance = np.linalg.norm(player.car_data.position[:2] - goal_position[:2])
        ball_distance = np.linalg.norm(state.ball.position[:2] - goal_position[:2])

        # Penalize if the player is farther from the goal than the ball
        if player_distance > ball_distance:
            return self.penalty
        return 0.0
    
class PossessionReward(RewardFunction):
    def __init__(self, max_reward=1.0):
        """
        Reward function for rewarding consecutive ball touches by the agent.
        :param max_reward: Maximum reward for possession (1.0 by default).
        """
        super().__init__()
        self.max_reward = max_reward
        self.touch_count = 0  # Tracks consecutive touches by the agent
        self.last_touch_player = None  # Tracks the last player who touched the ball

    def reset(self, initial_state: GameState):
        """
        Reset the reward function at the start of an episode.
        :param initial_state: The initial game state.
        """
        self.touch_count = 0
        self.last_touch_player = None

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        """
        Calculate the reward for ball possession.
        :param player: The player data.
        :param state: The current game state.
        :param previous_action: The player's previous action.
        :return: The reward for ball possession.
        """
        # Check if the player touched the ball
        if state.last_touch == player.car_id:
            # If the same player continues touching the ball, increment the touch count
            if self.last_touch_player == player.car_id:
                self.touch_count += 1
            else:
                # Reset the touch count if this is the first touch by this player
                self.touch_count = 1
                self.last_touch_player = player.car_id

            # Calculate the reward based on the touch count
            reward = min(self.touch_count, 5) * (self.max_reward / 5)
            return reward

        # Reset the touch count if an opponent touches the ball
        elif state.last_touch != player.car_id and state.last_touch is not None:
            self.touch_count = 0
            self.last_touch_player = state.last_touch

        return 0.0
    
# Import CAR_MAX_SPEED from common game values
from mysim.common_values import CAR_MAX_SPEED

class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity
        
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)
        
        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)
        if dist_to_ball > 6000:
            return 0.0

        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0
        
class StrongHitReward(RewardFunction):
    def __init__(self, max_reward=1.0):
        super().__init__()
        self.max_reward = max_reward
        self.last_ball_velocity = None

    def reset(self, initial_state: GameState):
        self.last_ball_velocity = initial_state.ball.linear_velocity
    def get_reward(self, player, state: GameState, previous_action):
        reward = 0.0
        ball_velocity = state.ball.linear_velocity
        if state.last_touch == player.car_id:
            
            hit_strength = np.linalg.norm(ball_velocity - self.last_ball_velocity)
            normalized_hit_strength = np.clip(hit_strength / 100, 0, 1)  # Normalize hit strength to [0, 1]

            # Apply square root scaling
            reward = np.sqrt(normalized_hit_strength) * self.max_reward
        self.last_ball_velocity = ball_velocity
            
        return reward
    
class TouchBallReward(RewardFunction):
    def __init__(self, aerial_weight=0.):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            # Default just rewards 1, set aerial weight to reward more depending on ball height
            return ((state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)) ** self.aerial_weight
        return 0
    
class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self, use_scalar_projection: bool = False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity           # (3,)
        pos_diff = state.ball.position - player.car_data.position  # (3,)

        # Distance to ball
        dist = float(np.linalg.norm(pos_diff))
        if dist > 6000.0:           # far away → negligible shaping (your choice)
            return 0.0
        if dist < 1e-6:             # avoid divide-by-zero
            return 0.0

        if self.use_scalar_projection:
            # scalar projection of velocity onto the ball direction: |v| * cos(theta)
            # equivalently dot(vel, pos_diff_hat)
            dir_to_ball = pos_diff / dist
            return float(np.dot(vel, dir_to_ball)) / CAR_MAX_SPEED
        else:
            # component velocity toward ball, normalized by car max speed
            dir_to_ball = pos_diff / dist
            return float(np.dot(dir_to_ball, vel)) / CAR_MAX_SPEED
