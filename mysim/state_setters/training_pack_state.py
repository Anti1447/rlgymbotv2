from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.math import rand_vec3
import numpy as np
from numpy import random as rand

X_MAX = 7000
Y_MAX = 9000
Z_MAX_BALL = 1850
Z_MAX_CAR = 1900
PITCH_MAX = np.pi / 2
YAW_MAX = np.pi
ROLL_MAX = np.pi
GRAVITY = np.array([0, 0, -650])  # Standard Rocket League gravity


class TrainingPackStateSetter(StateSetter):
    def __init__(self, shots=None, variation=False):
        """
        TrainingPackStateSetter constructor.

        :param shots: A list of predefined shots. Each shot is a dictionary with car and ball states.
        :param variation: If True, apply variation to the ball's position and velocity.
        """
        super().__init__()
        self.shots = shots if shots is not None else []
        self.variation = variation
        self.current_shot_index = 0

    def reset(self, state_wrapper: StateWrapper):
        """
        Resets the game state to a predefined state with optional variation.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        if self.shots:
            self._set_predefined_shot(state_wrapper)
        else:
            raise ValueError("No shots defined. Cannot set state.")
        
        # Place opponents outside the playable arena
        for i, car in enumerate(state_wrapper.cars):
            if i > 0:  # Assuming the first car is the controlled bot
                car.set_pos(99999, 99999, 99999)  # Place the car far outside the arena
                car.set_lin_vel(0, 0, 0)  # No movement
                car.set_ang_vel(0, 0, 0)  # No rotation
                car.boost = 0  # No boost

    def _apply_variation(self, base_value, variance_range):
        """
        Applies variation to a base value within a specified range.

        :param base_value: The base value to vary.
        :param variance_range: The range of variation (e.g., ±500).
        :return: The varied value.
        """
        return base_value + rand.uniform(-variance_range, variance_range)

    def _set_predefined_shot(self, state_wrapper: StateWrapper):
        """
        Sets the state to a predefined shot with optional variation.

        :param state_wrapper: StateWrapper object to be modified.
        """
        shot = self.shots[self.current_shot_index]

        # Set ball state with optional variation
        ball_position = shot["ball"]["position"]
        ball_velocity = shot["ball"]["linear_velocity"]
        ball_angular_velocity = shot["ball"]["angular_velocity"]

        if self.variation:
            ball_position = [
                self._apply_variation(ball_position[0], 500),  # X variation
                self._apply_variation(ball_position[1], 500),  # Y variation
                self._apply_variation(ball_position[2], 100)   # Z variation
            ]
            ball_velocity = [
                self._apply_variation(ball_velocity[0], 500),  # X velocity variation
                self._apply_variation(ball_velocity[1], 500),  # Y velocity variation
                self._apply_variation(ball_velocity[2], 200)   # Z velocity variation
            ]

        state_wrapper.ball.set_pos(*ball_position)
        state_wrapper.ball.set_lin_vel(*ball_velocity)
        state_wrapper.ball.set_ang_vel(*ball_angular_velocity)

        # Suspend gravity if the ball's net velocity is 0
        if np.linalg.norm(ball_velocity) == 0:
            state_wrapper.ball.gravity = np.array([0, 0, -0.001])  # Disable gravity/set very low gravity
        else:
            state_wrapper.ball.gravity = GRAVITY  # Enable standard gravity

        # Set car states
        for i, car in enumerate(state_wrapper.cars):
            if i < len(shot["cars"]):  # Only set state for cars defined in the shot
                car_data = shot["cars"][i]
                car.set_pos(*car_data["position"])
                car.set_rot(*car_data["rotation"])
                car.set_lin_vel(*car_data["linear_velocity"])
                car.set_ang_vel(*car_data["angular_velocity"])
                car.boost = car_data["boost"]
            else:  # Place extra cars (e.g., opponents) outside the playable arena
                car.set_pos(99999, 99999, 99999)  # Place the car far outside the arena
                car.set_lin_vel(0, 0, 0)  # No movement
                car.set_ang_vel(0, 0, 0)  # No rotation
                car.boost = 0  # No boost

        # Move to the next shot in the sequence
        self.current_shot_index = (self.current_shot_index + 1) % len(self.shots)