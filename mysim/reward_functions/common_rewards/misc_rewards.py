import numpy as np

from rlgym_sim.utils import math
from rlgym_sim.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, CEILING_Z
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions import RewardFunction



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