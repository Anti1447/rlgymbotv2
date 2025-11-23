# mysim/state_setters/turtled_start.py
import math, numpy as np
from rlgymbotv2.mysim.state_setters import StateSetter
from rlgymbotv2.mysim.state_setters.wrappers.state_wrapper import StateWrapper  # same import you use elsewhere

class TurtledStart(StateSetter):
    """
    Spawns all cars turtled (hood down), stationary, on flat ground near center.
    No hardcoded action sequences — just initial conditions for recovery learning.
    """

    def __init__(self, z=20.0, yaw_random=True, spawn_opponents=False):
        super().__init__()
        self.z = float(z)
        self.yaw_random = bool(yaw_random)
        self.spawn_opponents = bool(spawn_opponents)

    def reset(self, state: StateWrapper) -> None:
        # Ball
        state.ball.set_pos(0.0, 0.0, 92.75)
        state.ball.set_lin_vel(0.0, 0.0, 0.0)
        state.ball.set_ang_vel(0.0, 0.0, 0.0)
        if hasattr(state, "set_kickoff_pause"):
            state.set_kickoff_pause(False)

        def place_turtled(car, x, y):
            # Put center a hair above floor so the ROOF is almost touching
            car.set_pos(x, y, 14.0)                 # lower than before; closer to contact
            car.set_lin_vel(0.0, 0.0, -1500.0)      # strong downward so we hit floor immediately
            # Keep it awake and biased to tip
            car.set_ang_vel(0.15, 0.05, 0.10)       # small spin helps torques take effect

            # Upside down with a bigger roll so a WHEEL edge is likely to touch
            # pitch=π → upside down; roll≈0.55 rad (~31.5°) biases wheel toward ground
            car.set_rot(pitch=np.pi, yaw=0.0 if not self.yaw_random else np.random.uniform(-np.pi, np.pi), roll=0.55)

            if hasattr(car, "boost_amount"):
                car.boost_amount = 33.0
            if hasattr(state, "set_kickoff_pause"):
                state.set_kickoff_pause(False)


        spacing = 600.0
        for i, car in enumerate(state.blue_cars()):
            x = (i % 2) * spacing - spacing/2
            y = -(i // 2) * spacing
            place_turtled(car, x, y)

        if self.spawn_opponents:
            for i, car in enumerate(state.orange_cars()):
                x = (i % 2) * spacing - spacing/2
                y = +(i // 2) * spacing
                place_turtled(car, x, y)
