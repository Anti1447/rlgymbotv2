import numpy as np
import gym
from gym import spaces

from rlgymbotv2.selfplay.wrappers import SelfPlayEnvWrapper


class DummyOpponent:
    def __init__(self):
        self.calls = 0

    def act(self, obs_np):
        self.calls += 1
        # obs_np shape should be 1D
        assert obs_np.ndim == 1
        return 0  # dummy discrete action


class DummyMultiAgentEnv(gym.Env):
    def __init__(self, obs_dim=6):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._step_count = 0

    def reset(self, **kwargs):
        # Return [blue, orange] obs
        obs = [
            np.zeros(self.observation_space.shape, dtype=np.float32),
            np.ones(self.observation_space.shape, dtype=np.float32),
        ]
        self._step_count = 0
        return obs

    def step(self, actions):
        # Expect [a_blue, a_orange]
        if isinstance(actions, (list, tuple)):
            assert len(actions) == 2
        self._step_count += 1

        obs = [
            np.full(self.observation_space.shape, self._step_count, dtype=np.float32),
            np.full(self.observation_space.shape, -self._step_count, dtype=np.float32),
        ]
        reward = [1.0, 0.0]
        done = self._step_count >= 3
        info = {}
        return obs, reward, done, info


def test_selfplay_wrapper_basic_flow():
    base_env = DummyMultiAgentEnv(obs_dim=4)
    opp = DummyOpponent()
    env = SelfPlayEnvWrapper(base_env, opponent_policy=opp, profile=True)

    obs = env.reset()
    assert obs.shape == (4,)

    done = False
    steps = 0
    while not done:
        a_blue = 1
        obs, rew, done, info = env.step(a_blue)
        assert isinstance(rew, float)
        assert obs.shape == (4,)
        steps += 1

    assert steps == 3
    # Opponent should have been called once per step
    assert opp.calls == steps
