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

        self.last_pose_err = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_pose_err = self._get_pose_err(obs)
        return obs, info

    def _get_pose_err(self, obs):
        return obs[2]  # obs['pose_err'], idx 2

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # CHECK ME: is done flag correct?
        done = info["done"] or info["solved"]

        if done or truncated:
            info["episode_solved"] = info["solved"]

        # Reward if pose error gets smaller
        curr_pose_err = self._get_pose_err(obs)
        pose_err_bonus = self.last_pose_err - curr_pose_err
        self.last_pose_err = curr_pose_err

        large_action_penalty = info["rwd_dict"]["act_reg"] * 0.01
        solve_bonus = 0.2 if info["solved"] else 0

        reward = -1 if info["done"] else solve_bonus + pose_err_bonus + large_action_penalty

        return obs, reward, done, truncated, info
