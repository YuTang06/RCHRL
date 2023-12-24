import numpy as np
import argparse
from collections import deque
from gym import spaces


import envs.create_maze_env


def get_goal_sample_fn(env_name, evaluate):
    if env_name == 'AntMaze':
        if evaluate:
            return lambda: np.array([0., 16.])
        else:
            return lambda: np.random.uniform((-4, -4), (20, 20))
    elif env_name == 'AntMazeSparse':
        return lambda: np.array([2., 9.])
    else:
        assert False, 'Unknown env'


def get_reward_fn(env_name):
    if env_name in ['AntMaze']:
        return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
    elif env_name == 'AntMazeSparse':
        return lambda obs, goal: float(np.sum(np.square(obs[:2] - goal)) ** 0.5 < 1)
    # elif env_name == 'AntFall':
    elif env_name in ["drawer-open-v2", "door-open-v2", "door-close-v2", "reach-v2"]:
        return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
    else:
        assert False, 'Unknown env'

def get_success_fn(env_name):
    if env_name in ['AntMaze', "drawer-open-v2", "door-open-v2", "door-close-v2", "reach-v2"]:
        return lambda reward: reward > -5.0
    elif env_name == 'AntMazeSparse':
        return lambda reward: reward > 1e-6
    else:
        assert False, 'Unknown env'


class EnvWithGoal(object):

    def __init__(self, base_env, env_name):
        self.base_env = base_env
        self.env_name = env_name
        self.evaluate = False
        self.reward_fn = get_reward_fn(env_name)
        self.success_fn = get_success_fn(env_name)
        self.goal = None
        self.distance_threshold = 5 if env_name in ['AntMaze'] else 1
        self.count = 0
        self.early_stop = False if env_name in ['AntMaze'] else True
        self.early_stop_flag = False

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):
        # self.viewer_setup()
        self.early_stop_flag = False
        self.goal_sample_fn = get_goal_sample_fn(self.env_name, self.evaluate)
        obs = self.base_env.reset()
        self.count = 0
        self.goal = self.goal_sample_fn()
        self.desired_goal = self.goal if self.env_name in ['AntMaze'] else None
        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }

    def step(self, a):
        obs, _, done, info = self.base_env.step(a)
        reward = self.reward_fn(obs, self.goal)
        if self.early_stop and self.success_fn(reward):
            self.early_stop_flag = True
        self.count += 1
        done = self.early_stop_flag and self.count % 10 == 0
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:2],
            'desired_goal': self.desired_goal,
        }
        return next_obs, reward, done or self.count >= 500, info

    def render(self):
        self.base_env.render()

    @property
    def action_space(self):
        return self.base_env.action_space

    def observation_space(self):
        return self.base_env.observation_space

import metaworld
import random
class EnvWithSawyerArm(object):

    def __init__(self, base_env, env_name, seed):
        self.base_env = base_env
        self.seed_num = seed
        self.env_name = env_name
        self.goal = None
        self.reward_fn = get_reward_fn(env_name)
        self.success_fn = get_success_fn(env_name)
        self.early_stop = False
        self.count = 0

    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self):

        self.early_stop_flag = False
        task = metaworld.ML1(self.env_name).train_tasks[self.seed_num]
        self.base_env.set_task(task)  # Set task

        self.count = 0
        obs = self.base_env.reset()
        self.goal = self.base_env._target_pos
        self.desired_goal = self.goal

        return {
            'observation': obs.copy(),
            'achieved_goal': obs[:len(self.goal)],
            'desired_goal': self.desired_goal,
        }

    def step(self, a, evaluate = False):
        obs, reward, done, info = self.base_env.step(a)
        self.count += 1
        next_obs = {
            'observation': obs.copy(),
            'achieved_goal': obs[:len(self.goal)],
            'desired_goal': self.desired_goal,
        }

        return next_obs, reward, done or self.count >= 500, info

    def render(self):
        self.base_env.render()

    @property
    def action_space(self):
        return self.base_env.action_space
