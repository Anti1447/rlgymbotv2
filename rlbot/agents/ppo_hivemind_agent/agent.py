################################
# Date: 10/10/2025
# Revision: 1.0.0
# Description: Initial Release. Loads exported PPO policy for live play
# Title: agent.py
# Author: NJH
################################
from __future__ import annotations
from .discrete_action import DiscreteAction
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import numpy as np

from pathlib import Path
from .policy_loader import load_latest_artifact


class PPOHivemindAgent(BaseAgent):
    def initialize_agent(self):
        # Load exported policy once at startup
        self._discrete = DiscreteAction()
        try:
            self.policy = load_latest_artifact(Path("artifacts"))
            self.logger.info("Loaded policy artifact for PPOHivemindAgent")
        except Exception as e:
            self.logger.warn(f"Falling back to heuristic controls: {e}")
            self.policy = None

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Build observation
        obs = build_observation(packet)

        # If no policy, do a simple fallback (drive to ball)
        if self.policy is None:
            return heuristic_drive_to_ball(packet)

        # Model inference → action → controller
        try:
            action = self.policy.predict(obs)
            return map_action_to_controls(action)
        except Exception as e:
            self.logger.warn(f"Policy inference failed, using heuristic: {e}")
            return heuristic_drive_to_ball(packet)


# -------------------------
# Observation & action glue
# -------------------------

def build_observation(packet: GameTickPacket) -> np.ndarray:
    """Minimal placeholder observation: car(x,y,vx,vy), ball(x,y,vx,vy).
    Replace with your RLGym-style features to match the trained policy.
    """
    car = packet.game_cars[self.index]
    ball = packet.game_ball
    def v2(p):
        return np.array([p.x, p.y], dtype=np.float32)

    obs = np.concatenate([
        v2(car.physics.location),
        v2(car.physics.velocity),
        v2(ball.physics.location),
        v2(ball.physics.velocity),
    ])
    return obs


def map_action_to_controls(self, action: np.ndarray) -> SimpleControllerState:
    """
    Convert DiscreteAction bins into controller inputs.
    Handles both integer bins (already discretized) or logits.
    """
    ctl = SimpleControllerState()

    # 1. If model outputs logits, argmax across each dimension
    if action.ndim > 1:
        action = np.argmax(action, axis=1)

    # 2. Ensure shape (8,)
    action = np.array(action).flatten()
    if action.size != 8:
        self.logger.warn(f"Unexpected action shape {action.shape}, defaulting to zeros")
        action = np.zeros(8, dtype=np.int32)

    # 3. Parse bins → [-1,1] continuous controls
    parsed = self._discrete.parse_actions(action.reshape(1, -1), state=None)[0]

    ctl.throttle = float(parsed[0])
    ctl.steer = float(parsed[1])
    ctl.pitch = float(parsed[2])
    ctl.yaw = float(parsed[3])
    ctl.roll = float(parsed[4])
    ctl.jump = bool(parsed[5])
    ctl.boost = bool(parsed[6])
    ctl.handbrake = bool(parsed[7])
    return ctl


def heuristic_drive_to_ball(packet: GameTickPacket) -> SimpleControllerState:
    car = packet.game_cars[0]
    ball = packet.game_ball
    dx = ball.physics.location.x - car.physics.location.x
    dy = ball.physics.location.y - car.physics.location.y
    steer = np.clip(dx / 3000.0, -1, 1)
    throttle = 1.0
    ctl = SimpleControllerState()
    ctl.throttle = throttle
    ctl.steer = float(steer)
    ctl.boost = True
    return ctl