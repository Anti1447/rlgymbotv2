from abc import abstractmethod

import numpy as np

from mysim import RewardFunction
from mysim.common_values import BLUE_TEAM, ORANGE_TEAM, BACK_NET_Y, BACK_WALL_Y
from mysim.gamestates import PlayerData, GameState


class ConditionalRewardFunction(RewardFunction):
    def __init__(self, reward_func: RewardFunction):
        super().__init__()
        self.reward_func = reward_func

    @abstractmethod
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        raise NotImplementedError

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_reward(player, state, previous_action)
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_final_reward(player, state, previous_action)
        return 0


class RewardIfClosestToBall(ConditionalRewardFunction):
    def __init__(self, reward_func: RewardFunction, team_only=True):
        super().__init__(reward_func)
        self.team_only = team_only

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        dist = np.linalg.norm(player.car_data.position - state.ball.position)
        for player2 in state.players:
            if not self.team_only or player2.team_num == player.team_num:
                dist2 = np.linalg.norm(player2.car_data.position - state.ball.position)
                if dist2 < dist:
                    return False
        return True


class RewardIfTouchedLast(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return state.last_touch == player.car_id

class GoalIfTouchedLastReward(RewardFunction):
    def __init__(self, goal_reward=1.0, concede_reward=-1.0):
        super().__init__()
        self.goal_reward = goal_reward
        self.concede_reward = concede_reward
        self.last_touch = None

    def reset(self, initial_state: GameState):
        self.last_touch = None

    def get_reward(self, player, state: GameState, previous_action):
        reward = 0.0
        if state.last_touch == player.car_id:
            self.last_touch = player.car_id

        if state.blue_score > state.orange_score:
            if player.team_num == 0 and self.last_touch == player.car_id:
                reward += self.goal_reward
            elif player.team_num == 1:
                reward += self.concede_reward
        elif state.orange_score > state.blue_score:
            if player.team_num == 1 and self.last_touch == player.car_id:
                reward += self.goal_reward
            elif player.team_num == 0:
                reward += self.concede_reward

        return reward
    
class GoalIfTouchedLastConditionalReward(ConditionalRewardFunction):
    def __init__(self, goal_reward=1.0, concede_reward=-1.0):
        super().__init__(GoalIfTouchedLastReward(goal_reward, concede_reward))

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return state.last_touch == player.car_id

class FiftyFiftyReward(RewardFunction):
    def __init__(self, max_reward=1.0):
        """
        Rewards the bot for winning 50-50 challenges against the opponent.
        :param max_reward: Maximum reward for a successful 50-50.
        """
        super().__init__()
        self.max_reward = max_reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Find the opponent
        opponent = next(p for p in state.players if p.team_num != player.team_num)

        # Check if both bots are near the ball in 3D space
        ball_position = state.ball.position
        player_distance = np.linalg.norm(player.car_data.position - ball_position)  # Full 3D distance
        opponent_distance = np.linalg.norm(opponent.car_data.position - ball_position)  # Full 3D distance


        if player_distance < 500 and opponent_distance < 500:  # Both bots are near the ball
            # Check the ball's velocity after the touch
            ball_velocity = state.ball.linear_velocity
            goal_direction = np.array([0, 1, 0]) if player.team_num == BLUE_TEAM else np.array([0, -1, 0])
            velocity_toward_goal = np.dot(ball_velocity, goal_direction)

            # Reward if the ball moves toward the opponent's goal
            if velocity_toward_goal > 0 and state.last_touch == player.car_id:
                return self.max_reward
        return 0.0

class RewardIfBehindBall(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return player.team_num == BLUE_TEAM and player.car_data.position[1] < state.ball.position[1] \
               or player.team_num == ORANGE_TEAM and player.car_data.position[1] > state.ball.position[1]
    
class PunishIfInNet(RewardFunction):
    def __init__(self, own_net_penalty=0.5, opponent_net_penalty=1.0):
        """
        Punishes the bot for being inside its own net or the opponent's net.
        :param own_net_penalty: Penalty for being in the bot's own net.
        :param opponent_net_penalty: Penalty for being in the opponent's net.
        These parameters are set to positive values, and should be set to negative in the reward function.
        """
        super().__init__()
        self.own_net_penalty = own_net_penalty
        self.opponent_net_penalty = opponent_net_penalty

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Get the bot's y-coordinate
        bot_y = player.car_data.position[1]

        # Determine the y-range for the nets
        if BACK_WALL_Y < abs(bot_y) <= BACK_NET_Y:  # Bot is inside a net
            if player.team_num == BLUE_TEAM and bot_y < 0:  # Blue team's own net
                return self.own_net_penalty
            elif player.team_num == ORANGE_TEAM and bot_y > 0:  # Orange team's own net
                return self.own_net_penalty
            else:  # Opponent's net
                return self.opponent_net_penalty

        # No penalty if the bot is not in any net
        return 0.0