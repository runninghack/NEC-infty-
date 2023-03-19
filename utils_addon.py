from utils import projection_norm_inf
import numpy as np
from numpy.linalg import inv


def projection_nec_dgt(A, B, rho=1, kappa=0.99):
    # A is W_v, B is W_e
    A = A.clone().detach().cpu().numpy()
    B = B.clone().detach().cpu().numpy()
    X = A.clone().detach().cpu().numpy()
    Y = B.clone().detach().cpu().numpy()
    C = A @ B
    p = A.shape[0]
    q = A.shape[1]
    LAMBDA = np.random.randn(p, p)
    horizontalI = np.concatenate((np.eye(p), np.eye(p)), axis=1)
    blockedI = np.concatenate((horizontalI, horizontalI), axis=0)
    term_y3 = inv(np.eye(2 * p) + blockedI)

    num_iterations = 10
    for iter in range(num_iterations):
        # Update X
        term_x1 = 2*A - LAMBDA @ horizontalI @ Y.T + rho * C @ horizontalI @ Y.T
        term_x2 = 2*np.eye(p) + rho * Y @ blockedI @ Y.T
        X = term_x1 @ inv(term_x2)

        # Update Y
        term_y1 = 2 * np.eye(q) + rho * X.T @ X
        term_y2 = 2 * B - X.T @ LAMBDA @ horizontalI + rho * X.T @ C @ horizontalI  # [De_r, 2p]
        Y = inv(term_y1) @ term_y2 @ term_y3

        # Update C
        C = projection_norm_inf(X @ Y @  np.concatenate((np.eye(p), np.eye(p)), axis=0), kappa)

        # Update LAMBDA
        LAMBDA = LAMBDA + rho * (X @ Y @  np.concatenate((np.eye(p), np.eye(p))) - C)


    return X, Y
