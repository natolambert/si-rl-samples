import logging
import os
import sys
import hydra

import numpy as np  # Load the scipy functions

from control.matlab import *  # Load the controls systems libra
from samples.control_ex import *

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())
log = logging.getLogger(__name__)


def control(cfg):
    dt = .01

    # System matrices
    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [.5, 1.5, 3]])

    B = np.array([[0],
                  [0],
                  [1]])

    C = np.array([[1., 0, 0]])

    system = StateSpace(A, B, C, 0, dt)

    # Check controllability
    Wc = ctrb(A, B)
    print("Wc = ", Wc)
    print(f"Rank of controllability matrix is {np.linalg.matrix_rank(Wc)}")

    # Check Observability
    Wo = obsv(A, C)
    print("Wo = ", Wo)
    print(f"Rank of observability matrix is {np.linalg.matrix_rank(Wo)}")

    observer = full_obs(system, (.01, .011, .012))
    k_x = place(observer.A, observer.B, (.1, .11, .12))

    low = np.array([-1, -1, -1])
    high = np.array([1, 1, 1])
    x0 = np.random.uniform(low, high)


@hydra.main(config_path='config.yaml')
def experiment(cfg):
    control(cfg)

if __name__ == '__main__':
    sys.exit(experiment())

