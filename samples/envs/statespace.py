import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from control import StateSpace


class StateSpaceEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Observation: Defined by config
    Actions: Defined by config
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg.dt

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])
        self.seed(seed=cfg.random_seed)

        a = np.mat(cfg.sys.params.A)
        b = np.mat(cfg.sys.params.B)
        c = np.mat(cfg.sys.params.C)
        d = np.mat(cfg.sys.params.D)
        self.sys = StateSpace(a, b, c, d, self.dt)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError("TODO")

        return np.array(self.state), reward, done, {}

    def reset(self):
        raise NotImplementedError("TODO")
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

