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


def make_env(name):
    """Create an environment by name"""
    env = gym.make(name)

    # TODO: find ways to override DEFAULT_RWD_KEYS_AND_WEIGHTS
    # reward weights: env.unwrapped.rwd_keys_wt

    env = MyoWrapper(env)
    env = pufferlib.postprocess.ClipAction(env)
    env = EpisodeStats(env)
    # env = gym.wrappers.RecordEpisodeStatistics(env)

    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

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

        # # CHECK ME: is done flag correct?
        # done = info["done"] or info["solved"]

        # if done or truncated:
        #     info["episode_solved"] = info["solved"]

        # # Reward if pose error gets smaller
        # curr_pose_err = self._get_pose_err(obs)
        # pose_err_bonus = self.last_pose_err - curr_pose_err
        # self.last_pose_err = curr_pose_err

        # solve_bonus = 1 if info["solved"] else 0

        # reward = solve_bonus + 10 * pose_err_bonus

        # reward numbers seem too large, so we scale them down
        # reward = reward / 1000

        # TODO: obs needs to be float32
        return obs, reward, done, truncated, info


### Put the wrappers here, for now


class EpisodeStats(gymnasium.Wrapper):
    """Wrapper for Gymnasium environments that stores
    episodic returns and lengths in infos"""

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reset()

    def reset(self, seed=None, options=None):
        self.info = dict(episode_return=[], episode_length=0)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        self.info["episode_return"] += reward
        self.info["episode_length"] += 1

        if terminated or truncated:
            for k, v in self.info.items():
                info[k] = v

        return observation, reward, terminated, truncated, info
