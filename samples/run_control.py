import logging
import os
import sys
import hydra
import gym

import numpy as np

from control.matlab import *
from samples.control_ex import *

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())
log = logging.getLogger(__name__)


def control(cfg):
    dt = .01

    # System matrices
    A = np.mat(cfg.sys.A)
    B = np.mat(cfg.sys.B)
    C = np.mat(cfg.sys.C)
    D = np.mat(cfg.sys.D)

    system = StateSpace(A, B, C, D, dt)

    # Check controllability
    Wc = ctrb(A, B)
    print("Wc = ", Wc)
    print(f"Rank of controllability matrix is {np.linalg.matrix_rank(Wc)}")

    # Check Observability
    Wo = obsv(A, C)
    print("Wo = ", Wo)
    print(f"Rank of observability matrix is {np.linalg.matrix_rank(Wo)}")

    observer = full_obs(system, (.01, .011, .012, .013))
    k_x = place(observer.A, observer.B, (.1, .11, .12, .13))

    low = np.array([-1, -1, -1])
    high = np.array([1, 1, 1])
    x0 = np.random.uniform(low, high)

    env = gym.make(cfg.sys.name)
    env.setup(cfg)
    for i in range(cfg.num_trials):
        x0 = env.reset()

@hydra.main(config_path='conf/config.yaml')
def experiment(cfg):
    control(cfg)


if __name__ == '__main__':
    sys.exit(experiment())
