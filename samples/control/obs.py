import numpy as np
from control import TransferFunction, StateSpace, place


def full_obs(sys, poles):
    """Full order observer of the system sys
    Call:
    obs=full_obs(sys,poles)
    Parameters
    ----------
    sys : System in State Space form
    poles: desired observer poles
    Returns
    -------
    obs: ss
    Observer
    """
    if isinstance(sys, TransferFunction):
        "System must be in state space form"
        return
    a = np.mat(sys.A)
    b = np.mat(sys.B)
    c = np.mat(sys.C)
    d = np.mat(sys.D)
    L = place(a.T, c.T, poles)
    L = np.mat(L).T
    Ao = a - L * c
    Bo = np.hstack((b - L * d, L))
    n = np.shape(Ao)
    m = np.shape(Bo)
    Co = np.eye(n[0], n[1])
    Do = np.zeros((n[0], m[1]))
    obs = StateSpace(Ao, Bo, Co, Do, sys.dt)
    return obs


def red_obs(sys, T, poles):
    """Reduced order observer of the system sys
    Call:
    obs=red_obs(sys,T,poles)
    Parameters
    ----------
    sys : System in State Space form
    T: Complement matrix
    poles: desired observer poles
    Returns
    -------
    obs: ss
    Reduced order Observer
    """
    if isinstance(sys, TransferFunction):
        "System must be in state space form"
        return
    a = np.mat(sys.A)
    b = np.mat(sys.B)
    c = np.mat(sys.C)
    d = np.mat(sys.D)
    T = np.mat(T)
    P = np.mat(np.vstack((c, T)))
    invP = np.inv(P)
    AA = P * a * invP
    ny = np.shape(c)[0]
    nx = np.shape(a)[0]
    nu = np.shape(b)[1]

    A11 = AA[0:ny, 0:ny]
    A12 = AA[0:ny, ny:nx]
    A21 = AA[ny:nx, 0:ny]
    A22 = AA[ny:nx, ny:nx]

    L1 = place(A22.T, A12.T, poles)
    L1 = np.mat(L1).T

    nn = nx - ny

    tmp1 = np.mat(np.hstack((-L1, np.eye(nn, nn))))
    tmp2 = np.mat(np.vstack((np.zeros((ny, nn)), np.eye(nn, nn))))
    Ar = tmp1 * P * a * invP * tmp2

    tmp3 = np.vstack((np.eye(ny, ny), L1))
    tmp3 = np.mat(np.hstack((P * b, P * a * invP * tmp3)))
    tmp4 = np.hstack((np.eye(nu, nu), np.zeros((nu, ny))))
    tmp5 = np.hstack((-d, np.eye(ny, ny)))
    tmp4 = np.mat(np.vstack((tmp4, tmp5)))

    Br = tmp1 * tmp3 * tmp4

    Cr = invP * tmp2

    tmp5 = np.hstack((np.zeros((ny, nu)), np.eye(ny, ny)))
    tmp6 = np.hstack((np.zeros((nn, nu)), L1))
    tmp5 = np.mat(np.vstack((tmp5, tmp6)))
    Dr = invP * tmp5 * tmp4

    obs = StateSpace(Ar, Br, Cr, Dr, sys.dt)
    return obs
