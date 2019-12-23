import numpy as np
from control import ctrlutil, ControlArgument, ControlDimension, dare

def dlqr(*args, **keywords):
    """Linear quadratic regulator design for discrete systems
    Usage
    =====
    [K, S, E] = dlqr(A, B, Q, R, [N])
    [K, S, E] = dlqr(sys, Q, R, [N])
    The dlqr() function computes the optimal state feedback controller
    that minimizes the quadratic cost
        J = \sum_0^\infty x' Q x + u' R u + 2 x' N u
    Inputs
    ------
    A, B: 2-d arrays with dynamics and input matrices
    sys: linear I/O system
    Q, R: 2-d array with state and input weight matrices
    N: optional 2-d array with cross weight matrix
    Outputs
    -------
    K: 2-d array with state feedback gains
    S: 2-d array with solution to Riccati equation
    E: 1-d array with eigenvalues of the closed loop system
    """

    #
    # Process the arguments and figure out what inputs we received
    #

    # Get the system description
    if (len(args) < 3):
        raise ControlArgument("not enough input arguments")

    elif (ctrlutil.issys(args[0])):
        # We were passed a system as the first argument; extract A and B
        A = np.array(args[0].A, ndmin=2, dtype=float);
        B = np.array(args[0].B, ndmin=2, dtype=float);
        index = 1;
        if args[0].dt == 0.0:
            print
            "dlqr works only for discrete systems!"
            return
    else:
        # Arguments should be A and B matrices
        A = np.array(args[0], ndmin=2, dtype=float);
        B = np.array(args[1], ndmin=2, dtype=float);
        index = 2;

    # Get the weighting matrices (converting to matrices, if needed)
    Q = np.array(args[index], ndmin=2, dtype=float);
    R = np.array(args[index + 1], ndmin=2, dtype=float);
    if (len(args) > index + 2):
        N = np.array(args[index + 2], ndmin=2, dtype=float);
        Nflag = 1;
    else:
        N = np.zeros((Q.shape[0], R.shape[1]));
        Nflag = 0;

    # Check dimensions for consistency
    nstates = B.shape[0];
    ninputs = B.shape[1];
    if (A.shape[0] != nstates or A.shape[1] != nstates):
        raise ControlDimension("inconsistent system dimensions")

    elif (Q.shape[0] != nstates or Q.shape[1] != nstates or
          R.shape[0] != ninputs or R.shape[1] != ninputs or
          N.shape[0] != nstates or N.shape[1] != ninputs):
        raise ControlDimension("incorrect weighting matrix dimensions")

    if Nflag == 1:
        Ao = A - B * np.inv(R) * N.T
        Qo = Q - N * np.inv(R) * N.T
    else:
        Ao = A
        Qo = Q

    # Solve the riccati equation
    (X, L, G) = dare(Ao, B, Qo, R)
    #    X = bb_dare(Ao,B,Qo,R)

    # Now compute the return value
    Phi = np.mat(A)
    H = np.mat(B)
    K = np.inv(H.T * X * H + R) * (H.T * X * Phi + N.T)
    L = np.eig(Phi - H * K)
    return K, X, L
