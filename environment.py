# from pdb import set_trace as T

import functools
import numpy as np
import gymnasium

import pufferlib
import pufferlib.emulation
import pufferlib.postprocess

from myosuite.utils import gym


def env_creator(name="myoElbowPose1D6MFixed-v0"):
    return functools.partial(make_env, name)


def make_env(name, discretize=True):
    """Create an environment by name"""
    env = gym.make(name)

    # TODO: find ways to override DEFAULT_RWD_KEYS_AND_WEIGHTS
    # reward weights: env.unwrapped.rwd_keys_wt

    env = MyoWrapper(env, discretize)

    if discretize is False:
        env = pufferlib.postprocess.ClipAction(env)

    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env)


class MyoWrapper(gymnasium.Wrapper):
    def __init__(self, env, discretize):
        super().__init__(env)

        # override observation space
        self._observation_space = gym.spaces.Box(
            low=env.observation_space.low[0],
            high=env.observation_space.high[0],
            shape=env.observation_space.shape,
            dtype=np.float64,  # change from float32, which causes type error
        )

        # Discretize action space: 3 possible actions (-1, 0, 1) per joint
        self.discretize = discretize
        if self.discretize:
            self._action_space = gym.spaces.MultiDiscrete(env.action_space.shape[0] * [3])

        self.last_pose_err = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_pose_err = self._get_pose_err(obs)
        return obs, info

    def _get_pose_err(self, obs):
        return obs[2]  # obs['pose_err'], idx 2

    def step(self, action):
        if self.discretize:
            # Discretize action space: 3 possible actions per joint
            # Mapped from (0, 1, 2) to (-1, 0, 1), 1 being neutral
            action = action - 1

        obs, reward, done, truncated, info = self.env.step(action)

        # CHECK ME: is done flag correct?
        done = info["done"] or info["solved"]

        if done or truncated:
            info["episode_solved"] = info["solved"]

        # Reward if pose error gets smaller
        curr_pose_err = self._get_pose_err(obs)
        pose_err_bonus = self.last_pose_err - curr_pose_err
        self.last_pose_err = curr_pose_err

        solve_bonus = 1 if info["solved"] else 0

        reward = solve_bonus + 10 * pose_err_bonus

        return obs, reward, done, truncated, info
