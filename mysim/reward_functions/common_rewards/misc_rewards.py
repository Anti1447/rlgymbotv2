import numpy as np

from mysim import math
from mysim.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, CEILING_Z
from mysim.gamestates import GameState, PlayerData
from mysim.reward_functions import RewardFunction



class EventReward(RewardFunction):
    def __init__(self, goal=0., team_goal=0., concede=-0., touch=0., shot=0., save=0., demo=0., boost_pickup=0.):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param boost_pickup: reward for picking up boost. big pad = +1.0 boost, small pad = +0.12 boost.
        """
        super().__init__()
        self.weights = np.array([goal, team_goal, concede, touch, shot, save, demo, boost_pickup])

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array([player.match_goals, team, opponent, player.ball_touched, player.match_shots,
                         player.match_saves, player.match_demolishes, player.boost_amount])

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(player, initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward


class AlignBallGoal(RewardFunction):
    def __init__(self, defense=1., offense=1.):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * math.cosine_similarity(ball - pos, pos - protecc)

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * math.cosine_similarity(ball - pos, attacc - pos)

        return defensive_reward + offensive_reward
    
class BadOrientationPenalty(RewardFunction):
    """
    Applies a small punishment when the car is on the ground but not upright.

    Conditions:
      - Player is on_ground (1)
      - car.up().z < upright_threshold (e.g., < 0.75)
      - Additional penalty if completely upside down (car.up().z < 0)
    """

    def __init__(self, side_penalty=0.1, upside_down_penalty=0.3, upright_threshold=0.75):
        super().__init__()
        self.side_penalty = side_penalty
        self.upside_down_penalty = upside_down_penalty
        self.upright_threshold = upright_threshold

    def reset(self, initial_state: GameState):
        pass  # stateless

    def get_reward(self, player: PlayerData, state: GameState, previous_action):
        # Skip airborne states
        if not player.on_ground:
            return 0.0

        car_up_z = float(player.car_data.up()[2])

        # Fully upright → no penalty
        if car_up_z >= self.upright_threshold:
            return 0.0

        # On side or upside down
        if car_up_z < 0:
            return -self.upside_down_penalty
        else:
            return -self.side_penalty

class BoostPickupReward(RewardFunction):
    def __init__(self, big_pad_reward=1.0, small_pad_reward=0.5):
        super().__init__()
        self.big_pad_reward = big_pad_reward
        self.small_pad_reward = small_pad_reward
        self.last_boost_amount = {}

    def reset(self, initial_state: GameState):
        self.last_boost_amount = {player.car_id: player.boost_amount for player in initial_state.players}

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0.0
        boost_diff = player.boost_amount - self.last_boost_amount[player.car_id]

        if boost_diff > 0:
            if boost_diff > 0.12:  # Big pad
                reward = self.big_pad_reward
            else:  # Small pad
                reward = self.small_pad_reward

        self.last_boost_amount[player.car_id] = player.boost_amount
        return reward

class ConstantReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 1

class CollectBoostReward(RewardFunction):
    """
    Small shaping reward for collecting boost.
    Rewards the agent when its boost amount increases since the last tick.
    Favors small pads slightly more than big 100 pads to encourage better map flow.
    """

    def __init__(self, small_pad_bonus=1.0, big_pad_bonus=0.3):
        super().__init__()
        self.small_pad_bonus = small_pad_bonus
        self.big_pad_bonus = big_pad_bonus
        self.last_boost_amount = {}

    def reset(self, initial_state: GameState):
        # Track last boost amounts for all players
        self.last_boost_amount = {
            player.car_id: player.boost_amount for player in initial_state.players
        }

    def get_reward(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray
    ) -> float:
        # Get last known boost amount
        last = self.last_boost_amount.get(player.car_id, player.boost_amount)
        current = player.boost_amount
        diff = current - last

        # Update tracker
        self.last_boost_amount[player.car_id] = current

        # No boost gained → no reward
        if diff <= 0:
            return 0.0

        # Big pad usually adds ~1.0 (100%), small pad ~0.12 (12%)
        if diff > 0.5:
            # Big 100 pad collected → small reward
            return self.big_pad_bonus
        else:
            # Small pad collected → slightly higher reward
            # Scale smoothly with amount, capping near small_pad_bonus
            return self.small_pad_bonus * np.clip(diff / 0.12, 0, 1)


class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0
        
class RecoveryAndLandingReward(RewardFunction):
    """
    Combined reward encouraging:
      1️⃣ Smooth midair recovery (staying or becoming upright)
      2️⃣ Clean landings (on all 4 wheels)
      3️⃣ Proper surface alignment (using surface normal if available)

    Features:
      - Rewards aligning car.up() with surface normal or +Z
      - Detects 0→1 transitions of on_ground for landing events
      - Punishes bad landings (angled contact)
      - Optional bonus for landing on the ball
    """

    def __init__(
        self,
        upright_scale: float = 0.3,
        recovery_scale: float = 0.2,
        good_landing_scale: float = 1.0,
        bad_landing_penalty: float = 0.5,
        uprightness_weight: float = 0.2,
    ):
        super().__init__()
        self.upright_scale = upright_scale
        self.recovery_scale = recovery_scale
        self.good_landing_scale = good_landing_scale
        self.bad_landing_penalty = bad_landing_penalty
        self.uprightness_weight = uprightness_weight

        self._last_on_ground = {}
        self._last_up_align = {}

    # --------------------------------------------------------------
    def reset(self, initial_state: GameState):
        self._last_on_ground.clear()
        self._last_up_align.clear()

    # --------------------------------------------------------------
    def get_reward(self, player: PlayerData, state: GameState, previous_action):
        car_id = player.car_id
        car_data = player.car_data
        car_up = car_data.up()

        # --- Step 1: Determine alignment metric ---
        # Try to use surface normal if available
        surface_normal = getattr(player, "surface_normal", None)
        if surface_normal is not None:
            align = float(np.clip(np.dot(car_up, surface_normal), -1.0, 1.0))
        else:
            # Fall back to global +Z axis
            align = float(np.clip(car_up[2], -1.0, 1.0))

        # --- Step 2: Midair uprightness and recovery reward ---
        prev_align = self._last_up_align.get(car_id, align)
        recovery_delta = max(0.0, align - prev_align)

        upright_reward = align * self.upright_scale
        recovery_reward = recovery_delta * self.recovery_scale
        total_reward = upright_reward + recovery_reward

        # --- Step 3: Landing quality evaluation ---
        prev_ground = self._last_on_ground.get(car_id, player.on_ground)
        curr_ground = player.on_ground

        if prev_ground == 0 and curr_ground == 1:
            # Detect transition air→contact
            surf_obj = getattr(player, "last_hit_object", None)
            hit_ball = surf_obj == "ball" if surf_obj is not None else False

            if align > 0.85:
                # Good landing (aligned within ~30°)
                base = self.good_landing_scale
                if hit_ball:
                    base *= 1.25  # bonus for soft ball landing
                total_reward += base + align * self.uprightness_weight
            else:
                # Bad landing
                total_reward -= self.bad_landing_penalty * (1.0 - align)

        # --- Step 4: Update trackers ---
        self._last_on_ground[car_id] = curr_ground
        self._last_up_align[car_id] = align

        return float(total_reward)

class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # 1 reward for each frame with 100 boost, sqrt because 0->20 makes bigger difference than 80->100
        return np.sqrt(player.boost_amount)

class StealBoostReward(RewardFunction):
    def __init__(self, reward=1.0):
        """
        Rewards the bot for stealing big corner boosts on the opponent's half of the field,
        but only if the bot is closer to its own net than the ball.
        :param reward: Reward value for successfully stealing a big corner boost.
        """
        super().__init__()
        self.reward = reward
        self.last_boost_amount = {}

    def reset(self, initial_state: GameState):
        # Track the last boost amount for each player
        self.last_boost_amount = {player.car_id: player.boost_amount for player in initial_state.players}

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Define the big corner boost locations
        big_corner_boosts = [
            np.array([-3072, -4096, 73]),  # Blue half, left corner
            np.array([3072, -4096, 73]),   # Blue half, right corner
            np.array([-3072, 4096, 73]),   # Orange half, left corner
            np.array([3072, 4096, 73])     # Orange half, right corner
        ]

        # Determine the bot's team and goal position
        own_goal = np.array(BLUE_GOAL_BACK) if player.team_num == BLUE_TEAM else np.array(ORANGE_GOAL_BACK)
        opponent_goal = np.array(ORANGE_GOAL_BACK) if player.team_num == BLUE_TEAM else np.array(BLUE_GOAL_BACK)

        # Check if the bot is closer to its own net than the ball
        bot_position = player.car_data.position
        ball_position = state.ball.position
        distance_bot_to_own_goal = np.linalg.norm(bot_position - own_goal)
        distance_ball_to_own_goal = np.linalg.norm(ball_position - own_goal)

        if distance_bot_to_own_goal > distance_ball_to_own_goal:
            # Bot is not closer to its own net than the ball, quarter reward
            return 0.25

        # Check if the bot picked up a big corner boost on the opponent's half
        boost_diff = player.boost_amount - self.last_boost_amount[player.car_id]
        if boost_diff > 0.12:  # Big boost pad detected
            for boost_location in big_corner_boosts:
                # Check if the boost is on the opponent's half
                if (player.team_num == BLUE_TEAM and boost_location[1] > 0) or \
                   (player.team_num == ORANGE_TEAM and boost_location[1] < 0):
                    # Check if the bot is near the boost location
                    if np.linalg.norm(bot_position - boost_location) < 500:  # Within 500 units of the boost
                        self.last_boost_amount[player.car_id] = player.boost_amount
                        return self.reward

        # Update the last boost amount
        self.last_boost_amount[player.car_id] = player.boost_amount
        return 0.0
    
class VelocityReward(RewardFunction):
    # Simple reward function to ensure the model is training.
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.linalg.norm(player.car_data.linear_velocity) / CAR_MAX_SPEED * (1 - 2 * self.negative)