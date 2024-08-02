# from pdb import set_trace as T

import functools
import numpy as np
import gymnasium

import pufferlib
import pufferlib.emulation
import pufferlib.postprocess

from myosuite.utils import gym


def env_creator(name="myoElbowPose1D6MRandom-v0"):
    return functools.partial(make_env, name)


def make_env(name):
    """Create an environment by name"""
    env = gym.make(name)

    # TODO: find ways to override DEFAULT_RWD_KEYS_AND_WEIGHTS
    # reward weights: env.unwrapped.rwd_keys_wt

    env = MyoWrapper(env)
    env = pufferlib.postprocess.ClipAction(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)


class MyoWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # override observation space
        self._observation_space = gym.spaces.Box(
            low=env.observation_space.low[0],
            high=env.observation_space.high[0],
            shape=env.observation_space.shape,
            dtype=np.float64,  # change from float32, which causes type error
        )

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info
