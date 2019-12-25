import logging
import os
import sys
import hydra
import gym

import numpy as np
from past.utils import old_div

from control.matlab import *
from samples.control_ex import *
from sippy import system_identification
from sippy import functionset as fset
from sippy import functionsetSIM as fsetSIM

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())
log = logging.getLogger(__name__)


def control(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    dt = cfg.sys.params.dt

    # System matrices
    A = np.mat(cfg.sys.params.A)
    B = np.mat(cfg.sys.params.B)
    C = np.mat(cfg.sys.params.C)
    D = np.mat(cfg.sys.params.D)

    system = StateSpace(A, B, C, D, dt)
    dx = np.shape(A)[0]

    # Check controllability
    Wc = ctrb(A, B)
    print("Wc = ", Wc)
    print(f"Rank of controllability matrix is {np.linalg.matrix_rank(Wc)}")

    # Check Observability
    Wo = obsv(A, C)
    print("Wo = ", Wo)
    print(f"Rank of observability matrix is {np.linalg.matrix_rank(Wo)}")

    fdbk_poles = np.random.uniform(0, .01, size=(np.shape(A)[0]))
    obs_poles = np.random.uniform(0, .001, size=(np.shape(A)[0]))

    observer = full_obs(system, obs_poles)
    K = place(system.A, system.B, fdbk_poles)

    low = -np.ones((dx, 1))
    high = np.ones((dx, 1))
    x0 = np.random.uniform(low, high)

    env = gym.make(cfg.sys.name)
    env.setup(cfg)
    y = env.reset()
    x_obs = x0

    # code for testing observers... they work!
    # for t in range(100):
    #     action = controller(K).get_action(env._get_state())
    #     y, rew, done, _ = env.step(action)
    #     x_obs = np.matmul(observer.A, x_obs) + np.matmul(observer.B, np.concatenate((action, y)))
    #     print(f"Time {t}, mean sq state error is {np.mean(env._get_state()-x_obs)**2}")

    log.info(f"Running rollouts on system {cfg.sys.name}")
    states_r, actions_r, reward_r = rollout(env, random_controller(env), 500)

    log.info(f"Random rollout, reward {np.sum(reward_r)}")
    system_initial = sys_id(actions_r, states_r, dt, method='N4SID', order=dx)
    system_initial.dt = dt
    observer = full_obs(system_initial, obs_poles)
    K = place(system_initial.A, system_initial.B, fdbk_poles)

    print(np.array(cfg.sys.params.A))
    # TODO integrate observer so that we can use state feedback
    log.info(f"Running {cfg.experiment.num_trials} trials with SI and feedback control")
    for i in range(cfg.experiment.num_trials):
        # Rollout at this step
        # states, actions, reward = rollout(env, controller(K), cfg.experiment.trial_timesteps)
        states, actions, reward = rollout_observer(env, controller(K), observer, cfg.experiment.trial_timesteps)

        [states_r.append(s) for s in states]
        [actions_r.append(a) for a in actions]
        [reward_r.append(r) for r in reward]

        # Run system ID on the dataset
        # system = sys_id(actions, states, order=dx) # partial
        system = sys_id(actions_r, states_r, dt, order=dx)  # full
        system.dt = dt
        observer = full_obs(system, obs_poles)
        print(system.A.round(2))

        # Create controller
        K = place(system.A, system.B, fdbk_poles)

        log.info(f"Rollout {i}: reward {np.sum(reward)}")


def sys_id(actions, measurements, dt, method='N4SID', order=None):
    actions = np.stack(actions)[:, :, 0]
    measurements = np.stack(measurements)[:, :, 0]
    assert np.shape(measurements)[1] == np.shape(actions)[1]
    system = system_identification(measurements.T, actions.T, method, SS_f=5,
                                   SS_fixed_order=order, SS_threshold=0.01,
                                   tsample=dt)
    return system


class random_controller:
    def __init__(self, env):
        self.env = env

    def get_action(self, x):
        # return self.env.sampl
        return np.random.uniform(0, 1.0, size=(self.env.du, 1))


class controller:
    def __init__(self, K):
        self.K = K

    def get_action(self, x):
        return -np.matmul(self.K, x)


def rollout_observer(env, controller, observer, num_steps):
    done = False
    ys = []
    actions = []
    rews = []
    y = env.reset()
    state_est = np.zeros((observer.states, 1)) #np.random.uniform(0, 1.0, size=(observer.states, 1))
    for t in range(num_steps):
        if done:
            break
        action = controller.get_action(state_est)
        state_est = np.matmul(observer.A, state_est) + np.matmul(observer.B, np.concatenate((action, y)))

        y, rew, done, _ = env.step(action)
        ys.append(y)
        actions.append(np.array(action))
        rews.append(rew)

    return ys, actions, rews


def rollout(env, controller, num_steps):
    done = False
    states = []
    actions = []
    rews = []
    state = env.reset()
    for t in range(num_steps):
        if done:
            break
        action = controller.get_action(state)

        state, rew, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rews.append(rew)

    return states, actions, rews


@hydra.main(config_path='conf/config.yaml')
def experiment(cfg):
    control(cfg)

    ts = 1.0

    A = np.array([[0.89, 0.], [0., 0.45]])
    B = np.array([[0.3], [2.5]])
    C = np.array([[0.7, 1.]])
    D = np.array([[0.0]])

    tfin = 500
    npts = int(old_div(tfin, ts)) + 1
    Time = np.linspace(0, tfin, npts)

    # Input sequence
    U = np.zeros((1, npts))
    U[0] = fset.GBN_seq(npts, 0.05)

    ##Output
    x, yout = fsetSIM.SS_lsim_process_form(A, B, C, D, U)

    # measurement noise
    noise = fset.white_noise_var(npts, [0.15])

    # Output with noise
    y_tot = yout + noise

    ##System identification
    method = 'N4SID'
    sys_id = system_identification(y_tot, U, method, SS_fixed_order=2)


if __name__ == '__main__':
    sys.exit(experiment())
