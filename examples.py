import numpy as np  # Load the scipy functions
from control.matlab import *  # Load the controls systems libra

# Parameters defining the system

m = 250.0  # system mass
k = 40.0   # spring constant
b = 60.0   # damping constant

# System matrices
A = np.array([[1, -1, 1.],
             [1, -k/m, -b/m],
             [1, 1, 1]])

B = np.array([[0],
             [1/m],
             [1]])

C = np.array([[1., 0, 1.]])

sys = ss(A, B, C, 0)

# Check controllability
Wc = ctrb(A, B)
print("Wc = ", Wc)
print(f"Rank of controllability matrix is {np.linalg.matrix_rank(Wc)}")

# Check Observability
Wo = obsv(A, C)
print("Wo = ", Wo)
print(f"Rank of observability matrix is {np.linalg.matrix_rank(Wo)}")

